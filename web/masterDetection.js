import { generateUUID } from './constants.js';

export async function detectMasterIP(extension) {
    try {
        const isRunpod = window.location.hostname.endsWith('.proxy.runpod.net');
        if (isRunpod) {
            extension.log("Detected Runpod environment", "info");
        }

        const data = await extension.api.getNetworkInfo();
        extension.log("Network info: " + JSON.stringify(data), "debug");

        if (data.cuda_device !== null && data.cuda_device !== undefined) {
            extension.masterCudaDevice = data.cuda_device;

            if (!extension.config.master) {
                extension.config.master = {};
            }
            if (extension.config.master.cuda_device === undefined || extension.config.master.cuda_device !== data.cuda_device) {
                extension.config.master.cuda_device = data.cuda_device;
                try {
                    await extension.api.updateMaster({ cuda_device: data.cuda_device });
                    extension.log(`Stored master CUDA device: ${data.cuda_device}`, "debug");
                } catch (error) {
                    extension.log(`Error storing master CUDA device: ${error.message}`, "error");
                }
            }

            extension.ui.updateMasterDisplay(extension);
        }

        if (data.cuda_device_count > 0) {
            extension.cudaDeviceCount = data.cuda_device_count;
            extension.log(`Detected ${extension.cudaDeviceCount} CUDA devices`, "info");

            const shouldAutoPopulate =
                !extension.config.settings.has_auto_populated_workers &&
                (!extension.config.workers || extension.config.workers.length === 0);

            extension.log(`Auto-population check: has_populated=${extension.config.settings.has_auto_populated_workers}, workers=${extension.config.workers ? extension.config.workers.length : 'null'}, should_populate=${shouldAutoPopulate}`, "debug");

            if (shouldAutoPopulate) {
                extension.log(`Auto-populating workers based on ${extension.cudaDeviceCount} CUDA devices (excluding master on CUDA ${extension.masterCudaDevice})`, "info");

                const workerCount = extension.cudaDeviceCount - (
                    Number.isInteger(extension.masterCudaDevice) &&
                    extension.masterCudaDevice >= 0 &&
                    extension.masterCudaDevice < extension.cudaDeviceCount ? 1 : 0
                );
                const workerPorts = data.local_worker_ports;
                if (!Array.isArray(workerPorts) || workerPorts.length !== workerCount) {
                    throw new Error("Could not allocate enough available local worker ports; check the server log");
                }
                const newWorkers = [];
                let workerNum = 1;
                let portOffset = 0;

                for (let i = 0; i < extension.cudaDeviceCount; i++) {
                    if (i === extension.masterCudaDevice) {
                        extension.log(`Skipping CUDA ${i} (used by master)`, "debug");
                        continue;
                    }

                    const worker = {
                        id: generateUUID(),
                        name: `Worker ${workerNum}`,
                        host: isRunpod ? null : "localhost",
                        port: workerPorts[portOffset],
                        cuda_device: i,
                        enabled: true,
                        extra_args: isRunpod ? "--listen" : "",
                    };
                    newWorkers.push(worker);
                    workerNum += 1;
                    portOffset += 1;
                }

                if (newWorkers.length > 0) {
                    extension.log(`Auto-populating ${newWorkers.length} workers`, "info");

                    extension.config.workers = newWorkers;
                    extension.config.settings.has_auto_populated_workers = true;

                    for (const worker of newWorkers) {
                        try {
                            await extension.api.updateWorker(worker.id, worker);
                        } catch (error) {
                            extension.log(`Error saving worker ${worker.name}: ${error.message}`, "error");
                        }
                    }

                    try {
                        await extension.api.updateSetting('has_auto_populated_workers', true);
                    } catch (error) {
                        extension.log(`Error saving auto-population flag: ${error.message}`, "error");
                    }

                    extension.log(`Auto-populated ${newWorkers.length} workers and saved config`, "info");

                    if (extension.app.extensionManager?.toast) {
                        extension.app.extensionManager.toast.add({
                            severity: "success",
                            summary: "Workers Auto-populated",
                            detail: `Automatically created ${newWorkers.length} workers based on detected CUDA devices`,
                            life: 5000,
                        });
                    }

                    await extension.loadConfig();
                } else {
                    extension.log("No additional CUDA devices available for workers (all used by master)", "debug");
                }
            }
        }

        if (extension.config?.master?.host) {
            extension.log(`Master host already configured: ${extension.config.master.host}`, "debug");
            return;
        }

        if (isRunpod) {
            const runpodHost = window.location.hostname;
            extension.log(`Setting Runpod master host: ${runpodHost}`, "info");

            await extension.api.updateMaster({ host: runpodHost });

            if (!extension.config.master) {
                extension.config.master = {};
            }
            extension.config.master.host = runpodHost;

            if (extension.app.extensionManager?.toast) {
                extension.app.extensionManager.toast.add({
                    severity: "info",
                    summary: "Runpod Auto-Configuration",
                    detail: `Master host set to ${runpodHost} with --listen flag for workers`,
                    life: 5000,
                });
            }
            return;
        }

        if (data.recommended_ip && data.recommended_ip !== '127.0.0.1') {
            extension.log(`Auto-detected master IP: ${data.recommended_ip}`, "info");

            await extension.api.updateMaster({ host: data.recommended_ip });

            if (!extension.config.master) {
                extension.config.master = {};
            }
            extension.config.master.host = data.recommended_ip;
        }
    } catch (error) {
        extension.log("Error detecting master IP: " + error.message, "error");
    }
}
