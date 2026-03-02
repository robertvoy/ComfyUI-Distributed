import { api } from "../../scripts/api.js";
import { applyProbeResultToWorkerDot, findNodesByClass } from './workerUtils.js';
import { TIMEOUTS, NODE_CLASSES, generateUUID } from './constants.js';
import { checkAllWorkerStatuses, getWorkerUrl } from './workerLifecycle.js';

export function setupInterceptor(extension) {
    api.queuePrompt = async (number, prompt) => {
        if (extension.isEnabled) {
            const hasCollector = findNodesByClass(prompt.output, NODE_CLASSES.DISTRIBUTED_COLLECTOR).length > 0;
            const hasListSplitter = findNodesByClass(prompt.output, NODE_CLASSES.DISTRIBUTED_LIST_SPLITTER).length > 0;
            const hasListCollector = findNodesByClass(prompt.output, NODE_CLASSES.DISTRIBUTED_LIST_COLLECTOR).length > 0;
            const hasBranch = findNodesByClass(prompt.output, NODE_CLASSES.DISTRIBUTED_BRANCH).length > 0;
            const hasBranchCollector = findNodesByClass(prompt.output, NODE_CLASSES.DISTRIBUTED_BRANCH_COLLECTOR).length > 0;
            const hasDistUpscale = findNodesByClass(prompt.output, NODE_CLASSES.UPSCALE_DISTRIBUTED).length > 0;

            if (hasCollector || hasListSplitter || hasListCollector || hasBranch || hasBranchCollector || hasDistUpscale) {
                const result = await executeParallelDistributed(extension, prompt);
                // Immediate status check for instant feedback
                checkAllWorkerStatuses(extension);
                // Another check after a short delay to catch state changes
                setTimeout(() => checkAllWorkerStatuses(extension), TIMEOUTS.POST_ACTION_DELAY);
                return result;
            }
        }
        return extension.originalQueuePrompt(number, prompt);
    };
}

export async function executeParallelDistributed(extension, promptWrapper) {
    const traceExecutionId = `exec_${Date.now()}_${generateUUID().slice(0, 6)}`;
    try {
        const enabledWorkers = extension.enabledWorkers;
        extension.log(`[exec:${traceExecutionId}] Starting distributed execution`, "debug");
        
        // Pre-flight health check on all enabled workers
        const activeWorkers = await performPreflightCheck(extension, enabledWorkers);
        
        // Case: Enabled workers but all offline
        if (activeWorkers.length === 0 && enabledWorkers.length > 0) {
            extension.log("No active workers found. All enabled workers are offline.");
            if (extension.ui?.showToast) {
                extension.ui.showToast(extension.app, "error", "All Workers Offline", 
                    `${enabledWorkers.length} worker(s) enabled but all are offline or unreachable. Check worker connections and try again.`, 5000);
            }
            // Fall back to master-only execution
            return extension.originalQueuePrompt(0, promptWrapper);
        }
        
        extension.log(`Pre-flight check: ${activeWorkers.length} of ${enabledWorkers.length} workers are active`, "debug");

        // Check if master host might be unreachable by workers (cloudflare tunnel down)
        const masterHost = extension.config?.master?.host || '';
        const isCloudflareHost = /\.(trycloudflare\.com|cloudflare\.dev)$/i.test(masterHost);

        if (isCloudflareHost && activeWorkers.length > 0) {
            // Try to verify if the cloudflare tunnel is actually up
            try {
                const testUrl = `${window.location.protocol}//${masterHost}/prompt`;
                const response = await fetch(testUrl, {
                    method: 'GET',
                    mode: 'cors',
                    cache: 'no-cache',
                    signal: AbortSignal.timeout(3000) // 3 second timeout
                });

                if (!response.ok) {
                    throw new Error('Master not reachable');
                }
            } catch (error) {
                // Cloudflare tunnel appears to be down
                extension.log(`Master host ${masterHost} is not reachable - cloudflare tunnel may be down`, "error");

                if (extension.ui?.showCloudflareWarning) {
                    extension.ui.showCloudflareWarning(extension, masterHost);
                }

                // Stop execution - workers won't be able to send results back
                extension.log("Blocking execution - workers cannot reach master at cloudflare domain", "error");
                return null; // This will prevent the workflow from running
            }
        }

        const queueResponse = await extension.api.queueDistributed({
            prompt: promptWrapper.output,
            workflow: promptWrapper.workflow,
            enabled_worker_ids: activeWorkers.map((worker) => worker.id),
            workers: activeWorkers.map((worker) => ({ id: worker.id })),
            client_id: api.clientId,
            delegate_master: Boolean(extension.config?.settings?.master_delegate_only),
            auto_prepare: true,
            trace_execution_id: traceExecutionId,
        });
        if (queueResponse?.prompt_id) {
            extension.log(
                `[exec:${traceExecutionId}] Distributed queue accepted by backend (prompt_id=${queueResponse.prompt_id}, workers=${queueResponse.worker_count ?? activeWorkers.length})`,
                "debug"
            );
            return queueResponse;
        }
        throw new Error(
            `[exec:${traceExecutionId}] Backend did not return a prompt_id for distributed queue.`
        );
    } catch (error) {
        extension.log(`[exec:${traceExecutionId}] Distributed execution failed: ${error.message}`, "error");
        if (extension.ui?.showToast) {
            extension.ui.showToast(extension.app, "error", "Distributed Failed", error.message, 5000);
        }
        return null;
    }
}

export async function performPreflightCheck(extension, workers) {
    if (workers.length === 0) return [];
    
    extension.log(`Performing pre-flight health check on ${workers.length} workers...`, "debug");
    const startTime = Date.now();
    
    const checkPromises = workers.map(async (worker) => {
        const workerUrl = getWorkerUrl(extension, worker);
        
        extension.log(`Pre-flight checking ${worker.name} at: ${workerUrl}`, "debug");
        
        try {
            const probeResult = await extension.api.probeWorker(workerUrl, TIMEOUTS.STATUS_CHECK);

            if (probeResult.ok) {
                extension.log(`Worker ${worker.name} is active`, "debug");
                return { worker, active: true };
            } else {
                extension.log(`Worker ${worker.name} returned ${probeResult.status}`, "debug");
                return { worker, active: false };
            }
        } catch (error) {
            if (error?.name === 'AbortError') {
                extension.log(`Worker ${worker.name} pre-flight check timed out; assuming active`, "debug");
                return { worker, active: true, uncertain: true };
            }
            extension.log(`Worker ${worker.name} is offline or unreachable: ${error.message}`, "debug");
            return { worker, active: false };
        }
    });
    
    const results = await Promise.all(checkPromises);
    const activeWorkers = results.filter(r => r.active).map(r => r.worker);
    
    const elapsed = Date.now() - startTime;
    extension.log(`Pre-flight check completed in ${elapsed}ms. Active workers: ${activeWorkers.length}/${workers.length}`, "debug");
    
    // Update UI status indicators for inactive workers
    results.filter(r => !r.active).forEach(r => {
        applyProbeResultToWorkerDot(r.worker.id, { ok: false });
    });
    
    return activeWorkers;
}
