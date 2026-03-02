import { TIMEOUTS, STATUS_COLORS } from './constants.js';
import { buildWorkerUrl, normalizeWorkerUrl } from './urlUtils.js';
import { isRemoteWorker } from './workerSettings.js';
import { applyProbeResultToWorkerDot } from './workerUtils.js';

export { normalizeWorkerUrl };

let _statusCheckRunning = false;

function setStatusDotClass(dot, statusClass) {
    if (!dot) {
        return;
    }
    const classes = [
        "worker-status--online",
        "worker-status--offline",
        "worker-status--unknown",
        "worker-status--processing",
    ];
    dot.classList.remove(...classes);
    if (statusClass) {
        dot.classList.add(statusClass);
    }
}

function setButtonClass(button, className) {
    if (!button) {
        return;
    }
    button.classList.remove("btn--stop", "btn--launch", "btn--log", "btn--working", "btn--success", "btn--error");
    if (className) {
        button.classList.add(className);
    }
}

function setButtonVisibility(button, visible) {
    if (!button) {
        return;
    }
    button.classList.toggle("is-hidden", !visible);
    button.style.display = visible ? "" : "none";
}

export function shouldShowRemoteLogButton(worker, status = {}, now = Date.now()) {
    if (!worker?.enabled) {
        return false;
    }
    if (status?.online) {
        return true;
    }
    const lastSeenOnlineAt = Number(status?.lastSeenOnlineAt || 0);
    if (!Number.isFinite(lastSeenOnlineAt) || lastSeenOnlineAt <= 0) {
        return false;
    }
    return (now - lastSeenOnlineAt) <= TIMEOUTS.REMOTE_LOG_AVAILABILITY_WINDOW;
}

export async function checkAllWorkerStatuses(extension) {
    if (_statusCheckRunning || !extension.panelElement) {
        return;
    }
    _statusCheckRunning = true;
    let nextInterval = 5000;

    try {
        // Create a fresh AbortController for this poll cycle.
        extension.statusCheckAbortController = new AbortController();

        await checkMasterStatus(extension);

        if (extension.config?.workers) {
            await Promise.all(
                extension.config.workers.map(async (worker) => {
                    if (worker.enabled || extension.state.isWorkerLaunching(worker.id)) {
                        await checkWorkerStatus(extension, worker);
                    }
                })
            );
        }

        let isActive = extension.state.getMasterStatus() === "processing";
        extension.config?.workers?.forEach((worker) => {
            const workerState = extension.state.getWorker(worker.id);
            if (workerState.launching || workerState.status?.processing) {
                isActive = true;
            }
        });

        nextInterval = isActive ? 1000 : 5000;
    } finally {
        _statusCheckRunning = false;
        if (extension.panelElement) {
            extension.statusCheckTimeout = setTimeout(() => checkAllWorkerStatuses(extension), nextInterval);
        }
    }
}

export async function checkMasterStatus(extension) {
    try {
        const signal = extension.statusCheckAbortController?.signal || null;
        const probeResult = await extension.api.probeWorker(
            window.location.origin,
            TIMEOUTS.STATUS_CHECK,
            signal,
        );
        if (!probeResult.ok) {
            throw new Error(`HTTP ${probeResult.status}`);
        }

        const queueRemaining = probeResult.queueRemaining || 0;
        const isProcessing = queueRemaining > 0;

        // Update master status in state
        extension.state.setMasterStatus(isProcessing ? "processing" : "online");

        // Update master status dot
        const statusDot = document.getElementById("master-status");
        if (statusDot) {
            if (!extension.isMasterParticipating()) {
                if (isProcessing) {
                    setStatusDotClass(statusDot, "worker-status--processing");
                    statusDot.title = `Orchestrating (${queueRemaining} in queue)`;
                } else {
                    setStatusDotClass(statusDot, "worker-status--unknown");
                    statusDot.title = "Master orchestrator only";
                }
            } else if (isProcessing) {
                setStatusDotClass(statusDot, "worker-status--processing");
                statusDot.title = `Processing (${queueRemaining} in queue)`;
            } else {
                setStatusDotClass(statusDot, "worker-status--online");
                statusDot.title = "Online";
            }
        }
    } catch (error) {
        if (error?.name === "AbortError") {
            return;
        }
        // Master is always online (we're running on it), so keep it green
        const statusDot = document.getElementById("master-status");
        if (statusDot) {
            setStatusDotClass(
                statusDot,
                extension.isMasterParticipating() ? "worker-status--online" : "worker-status--unknown"
            );
            statusDot.title = extension.isMasterParticipating() ? "Online" : "Master orchestrator only";
        }
    }
}

// Helper to build worker URL
export function getWorkerUrl(extension, worker, endpoint = "") {
    return buildWorkerUrl(worker, endpoint, window.location);
}

export async function checkWorkerStatus(extension, worker) {
    // Assume caller ensured enabled; proceed with check
    const workerUrl = getWorkerUrl(extension, worker);
    const previousStatus = extension.state.getWorkerStatus(worker.id);

    try {
        const signal = extension.statusCheckAbortController?.signal || null;
        const probeResult = await extension.api.probeWorker(
            workerUrl,
            TIMEOUTS.STATUS_CHECK,
            signal,
        );
        if (!probeResult.ok) {
            throw new Error(`HTTP ${probeResult.status}`);
        }

        const queueRemaining = probeResult.queueRemaining || 0;
        const isProcessing = queueRemaining > 0;

        // Update status
        extension.state.setWorkerStatus(worker.id, {
            online: true,
            processing: isProcessing,
            queueCount: queueRemaining,
            lastSeenOnlineAt: Date.now(),
        });

        // Update status dot based on probe result
        applyProbeResultToWorkerDot(worker.id, probeResult);

        // Clear launching state since worker is now online
        if (extension.state.isWorkerLaunching(worker.id)) {
            extension.state.setWorkerLaunching(worker.id, false);
            clearLaunchingFlag(extension, worker.id);
        }
    } catch (error) {
        // Don't process aborted requests
        if (error.name === "AbortError") {
            return;
        }

        // Worker is offline or unreachable
        extension.state.setWorkerStatus(worker.id, {
            online: false,
            processing: false,
            queueCount: 0,
            lastSeenOnlineAt: previousStatus?.lastSeenOnlineAt || 0,
        });

        // Check if worker is launching
        if (extension.state.isWorkerLaunching(worker.id)) {
            extension.ui.updateStatusDot(worker.id, STATUS_COLORS.PROCESSING_YELLOW, "Launching...", true);
        } else if (worker.enabled) {
            // Only update to red if not currently launching AND still enabled.
            applyProbeResultToWorkerDot(worker.id, { ok: false });
        }
        // If disabled, don't update the dot (leave it gray)

        extension.log(`Worker ${worker.id} status check failed: ${error.message}`, "debug");
    }

    // Update control buttons based on new status
    const updatedInPlace = extension.updateWorkerCard?.(worker.id, extension.state.getWorkerStatus(worker.id));
    if (!updatedInPlace) {
        updateWorkerControls(extension, worker.id);
    }
}

export async function launchWorker(extension, workerId) {
    const worker = extension.config.workers.find((w) => w.id === workerId);

    // If worker is disabled, enable it first
    if (!worker.enabled) {
        await extension.updateWorkerEnabled(workerId, true);

        // Update the checkbox UI
        const checkbox = document.getElementById(`gpu-${workerId}`);
        if (checkbox) {
            checkbox.checked = true;
        }
    }

    // Re-query button AFTER updateWorkerEnabled (which may re-render sidebar)
    const launchBtn = document.querySelector(`#controls-${workerId} button`);

    extension.ui.updateStatusDot(workerId, STATUS_COLORS.PROCESSING_YELLOW, "Launching...", true);
    extension.state.setWorkerLaunching(workerId, true);

    // Allow 90 seconds for worker to launch (model loading can take time)
    setTimeout(() => {
        extension.state.setWorkerLaunching(workerId, false);
    }, TIMEOUTS.LAUNCH);

    if (!launchBtn) {
        return;
    }

    try {
        // Disable button immediately
        launchBtn.disabled = true;

        const result = await extension.api.launchWorker(workerId);
        if (result) {
            extension.log(`Launched ${worker.name} (PID: ${result.pid})`, "info");
            if (result.log_file) {
                extension.log(`Log file: ${result.log_file}`, "debug");
            }

            extension.state.setWorkerManaged(workerId, {
                pid: result.pid,
                log_file: result.log_file,
                started_at: Date.now(),
            });

            // Update controls immediately to hide launch button and show stop/log buttons
            updateWorkerControls(extension, workerId);
            setTimeout(() => checkWorkerStatus(extension, worker), TIMEOUTS.STATUS_CHECK);
        }
    } catch (error) {
        // Check if worker was already running
        if (error.message && error.message.includes("already running")) {
            extension.log(`Worker ${worker.name} is already running`, "info");
            updateWorkerControls(extension, workerId);
            setTimeout(() => checkWorkerStatus(extension, worker), TIMEOUTS.STATUS_CHECK_DELAY);
        } else {
            extension.log(`Error launching worker: ${error.message || error}`, "error");

            // Re-enable button on error
            if (launchBtn) {
                launchBtn.disabled = false;
            }
        }
    }
}

export async function stopWorker(extension, workerId) {
    const worker = extension.config.workers.find((w) => w.id === workerId);
    const stopBtn = document.querySelectorAll(`#controls-${workerId} button`)[1];

    // Provide immediate feedback
    if (stopBtn) {
        stopBtn.disabled = true;
        stopBtn.textContent = "Stopping...";
        setButtonClass(stopBtn, "btn--working");
    }

    try {
        const result = await extension.api.stopWorker(workerId);
        if (result) {
            extension.log(`Stopped worker: ${result.message}`, "info");
            extension.state.setWorkerManaged(workerId, null);

            // Immediately update status to offline
            extension.ui.updateStatusDot(workerId, STATUS_COLORS.OFFLINE_RED, "Offline");
            extension.state.setWorkerStatus(workerId, { online: false });

            // Flash success feedback
            if (stopBtn) {
                setButtonClass(stopBtn, "btn--success");
                stopBtn.textContent = "Stopped!";
                setTimeout(() => {
                    updateWorkerControls(extension, workerId);
                }, TIMEOUTS.FLASH_SHORT);
            }

            // Verify status after a short delay
            setTimeout(() => checkWorkerStatus(extension, worker), TIMEOUTS.STATUS_CHECK);
        } else {
            extension.log(`Failed to stop worker: ${result.message}`, "error");

            // Flash error feedback
            if (stopBtn) {
                setButtonClass(stopBtn, "btn--error");
                stopBtn.textContent = result.message.includes("already stopped") ? "Not Running" : "Failed";

                // If already stopped, update status immediately
                if (result.message.includes("already stopped")) {
                    extension.ui.updateStatusDot(workerId, STATUS_COLORS.OFFLINE_RED, "Offline");
                    extension.state.setWorkerStatus(workerId, { online: false });
                }

                setTimeout(() => {
                    updateWorkerControls(extension, workerId);
                }, TIMEOUTS.FLASH_MEDIUM);
            }
        }
    } catch (error) {
        extension.log(`Error stopping worker: ${error}`, "error");

        // Reset button on error
        if (stopBtn) {
            setButtonClass(stopBtn, "btn--error");
            stopBtn.textContent = "Error";
            setTimeout(() => {
                updateWorkerControls(extension, workerId);
            }, TIMEOUTS.FLASH_MEDIUM);
        }
    }
}

export async function clearLaunchingFlag(extension, workerId) {
    try {
        await extension.api.clearLaunchingFlag(workerId);
        extension.log(`Cleared launching flag for worker ${workerId}`, "debug");
    } catch (error) {
        extension.log(`Error clearing launching flag: ${error.message || error}`, "error");
    }
}

export async function loadManagedWorkers(extension) {
    try {
        const result = await extension.api.getManagedWorkers();

        // Check for launching workers
        for (const [workerId, info] of Object.entries(result.managed_workers)) {
            extension.state.setWorkerManaged(workerId, info);

            // If worker is marked as launching, add to launchingWorkers set
            if (info.launching) {
                extension.state.setWorkerLaunching(workerId, true);
                extension.log(`Worker ${workerId} is in launching state`, "debug");
            }
        }

        // Update UI for all workers
        if (extension.config?.workers) {
            extension.config.workers.forEach((w) => updateWorkerControls(extension, w.id));
        }
    } catch (error) {
        extension.log(`Error loading managed workers: ${error}`, "error");
    }
}

export function updateWorkerControls(extension, workerId) {
    const controlsDiv = document.getElementById(`controls-${workerId}`);

    if (!controlsDiv) {
        return;
    }

    const worker = extension.config.workers.find((w) => w.id === workerId);
    if (!worker) {
        return;
    }

    // Update button states - buttons are now inside a wrapper div
    const launchBtn = document.getElementById(`launch-${workerId}`);
    const stopBtn = document.getElementById(`stop-${workerId}`);
    const logBtn = document.getElementById(`log-${workerId}`);
    const status = extension.state.getWorkerStatus(workerId);

    if (isRemoteWorker(extension, worker)) {
        setButtonVisibility(launchBtn, false);
        setButtonVisibility(stopBtn, false);
        if (logBtn) {
            const showRemoteLog = shouldShowRemoteLogButton(worker, status);
            setButtonVisibility(logBtn, showRemoteLog);
            logBtn.disabled = !showRemoteLog;
            logBtn.textContent = "View Log";
            setButtonClass(logBtn, "btn--log");
        }
        return;
    }

    // Ensure we check for string ID
    const managedInfo = extension.state.getWorker(workerId).managed;
    // Show log button immediately if we have log file info (even if worker is still starting)
    if (logBtn) {
        const showLog = Boolean(managedInfo?.log_file);
        setButtonVisibility(logBtn, showLog);
        if (showLog) {
            setButtonClass(logBtn, "btn--log");
        }
    }

    if (status?.online || managedInfo) {
        // Worker is running or we just launched it
        setButtonVisibility(launchBtn, false);

        if (managedInfo) {
            // Only show stop button if we manage this worker
            setButtonVisibility(stopBtn, true);
            stopBtn.disabled = false;
            stopBtn.textContent = "Stop";
            setButtonClass(stopBtn, "btn--stop");
        } else {
            // Hide stop button for workers launched outside UI
            setButtonVisibility(stopBtn, false);
        }
    } else {
        // Worker is not running
        setButtonVisibility(launchBtn, true);
        launchBtn.disabled = false;
        launchBtn.textContent = "Launch";
        setButtonClass(launchBtn, "btn--launch");

        setButtonVisibility(stopBtn, false);
    }
}

export async function viewWorkerLog(extension, workerId, isRemote = false) {
    const worker = extension.config.workers.find((w) => w.id === workerId);
    const isRemoteLog = isRemote || (worker ? isRemoteWorker(extension, worker) : false);
    const managedInfo = extension.state.getWorker(workerId).managed;
    if (!isRemoteLog && !managedInfo?.log_file) {
        return;
    }

    const logBtn = document.getElementById(`log-${workerId}`);

    // Provide immediate feedback
    if (logBtn) {
        logBtn.disabled = true;
        logBtn.textContent = "Loading...";
        setButtonClass(logBtn, "btn--working");
    }

    try {
        const fetchLog = isRemoteLog
            ? async () => extension.api.getRemoteWorkerLog(workerId, 300)
            : async () => extension.api.getWorkerLog(workerId, 1000);
        const data = await fetchLog();

        // Create modal dialog
        extension.ui.showLogModal(extension, workerId, data, fetchLog);

        // Restore button
        if (logBtn) {
            logBtn.disabled = false;
            logBtn.textContent = "View Log";
            setButtonClass(logBtn, "btn--log");
        }
    } catch (error) {
        extension.log("Error viewing log: " + error.message, "error");
        extension.app.extensionManager.toast.add({
            severity: "error",
            summary: "Error",
            detail: `Failed to load log: ${error.message}`,
            life: 5000,
        });

        // Flash error and restore button
        if (logBtn) {
            setButtonClass(logBtn, "btn--error");
            logBtn.textContent = "Error";
            setTimeout(() => {
                logBtn.disabled = false;
                logBtn.textContent = "View Log";
                setButtonClass(logBtn, "btn--log");
            }, TIMEOUTS.FLASH_LONG);
        }
    }
}

export async function refreshLog(extension, workerId, silent = false) {
    const logContent = document.getElementById("distributed-log-content");
    if (!logContent) {
        return;
    }

    try {
        const worker = extension.config.workers.find((w) => w.id === workerId);
        const isRemoteLog = worker ? isRemoteWorker(extension, worker) : false;
        const data = isRemoteLog
            ? await extension.api.getRemoteWorkerLog(workerId, 300)
            : await extension.api.getWorkerLog(workerId, 1000);

        // Update content
        const shouldAutoScroll = logContent.scrollTop + logContent.clientHeight >= logContent.scrollHeight - 50;
        logContent.textContent = data.content;

        // Auto-scroll if was at bottom
        if (shouldAutoScroll) {
            logContent.scrollTop = logContent.scrollHeight;
        }

        // Only show toast if not in silent mode (manual refresh)
        if (!silent) {
            extension.app.extensionManager.toast.add({
                severity: "success",
                summary: "Log Refreshed",
                detail: "Log content updated",
                life: 2000,
            });
        }
    } catch (error) {
        // Only show error toast if not in silent mode
        if (!silent) {
            extension.app.extensionManager.toast.add({
                severity: "error",
                summary: "Refresh Failed",
                detail: error.message,
                life: 3000,
            });
        }
    }
}

export function startLogAutoRefresh(extension, workerId) {
    // Stop any existing auto-refresh
    stopLogAutoRefresh(extension);

    // Refresh every 2 seconds
    extension.logAutoRefreshInterval = setInterval(() => {
        refreshLog(extension, workerId, true); // silent mode
    }, TIMEOUTS.LOG_REFRESH);
}

export function stopLogAutoRefresh(extension) {
    if (extension.logAutoRefreshInterval) {
        clearInterval(extension.logAutoRefreshInterval);
        extension.logAutoRefreshInterval = null;
    }
}

export function toggleWorkerExpanded(extension, workerId) {
    const gpuDiv = document.querySelector(`[data-worker-id="${workerId}"]`);
    const settingsDiv = gpuDiv?.querySelector(`#settings-${workerId}`) || document.getElementById(`settings-${workerId}`);
    const settingsArrow = gpuDiv?.querySelector(".settings-arrow");

    if (!settingsDiv) {
        return;
    }

    if (extension.state.isWorkerExpanded(workerId)) {
        extension.state.setWorkerExpanded(workerId, false);
        settingsDiv.classList.remove("expanded");
        settingsDiv.style.padding = "0 12px";
        settingsDiv.style.marginTop = "0";
        settingsDiv.style.marginBottom = "0";
        if (settingsArrow) {
            settingsArrow.classList.remove("settings-arrow--expanded");
        }
    } else {
        extension.state.setWorkerExpanded(workerId, true);
        settingsDiv.classList.add("expanded");
        settingsDiv.style.padding = "12px";
        settingsDiv.style.marginTop = "8px";
        settingsDiv.style.marginBottom = "8px";
        if (settingsArrow) {
            settingsArrow.classList.add("settings-arrow--expanded");
        }
    }
}
