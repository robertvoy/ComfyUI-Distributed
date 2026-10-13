import { TIMEOUTS } from './constants.js';
import { normalizeWorkerUrl } from './urlUtils.js';

export function createApiClient(baseUrl) {
    const normalizedBaseUrl = normalizeWorkerUrl(baseUrl);

    const request = async (
        endpoint,
        options = {},
        { retries = TIMEOUTS.MAX_RETRIES, retry = true, redactErrors = false } = {},
    ) => {
        const maxAttempts = retry ? retries : 1;
        let lastError;
        let delay = TIMEOUTS.RETRY_DELAY; // Initial delay for exponential backoff

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
                const headers = {
                    'Content-Type': 'application/json',
                    ...(options.headers || {}),
                };
                const response = await fetch(`${normalizedBaseUrl}${endpoint}`, {
                    ...options,
                    headers,
                });
                
                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    const message = error.message
                        || error.error
                        || (Array.isArray(error.errors) ? error.errors.join('; ') : null)
                        || `HTTP ${response.status}`;
                    throw new Error(message);
                }
                
                return await response.json();
            } catch (error) {
                lastError = error;
                console.log(`API Error (attempt ${attempt + 1}/${maxAttempts}): ${endpoint} - ${redactErrors ? "[redacted]" : error.message}`);
                if (attempt < maxAttempts - 1) {
                    await new Promise(resolve => setTimeout(resolve, delay));
                    delay *= 2; // Exponential backoff
                }
            }
        }
        throw lastError;
    };

    const requestUrl = async (
        url,
        options = {},
        { retries = TIMEOUTS.MAX_RETRIES, retry = true } = {},
    ) => {
        const maxAttempts = retry ? retries : 1;
        let lastError;
        let delay = TIMEOUTS.RETRY_DELAY;

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
                const response = await fetch(url, options);
                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    const message = error.message
                        || error.error
                        || (Array.isArray(error.errors) ? error.errors.join('; ') : null)
                        || `HTTP ${response.status}`;
                    throw new Error(message);
                }
                return await response.json();
            } catch (error) {
                lastError = error;
                console.log(`API Error (attempt ${attempt + 1}/${maxAttempts}): ${url} - ${error.message}`);
                if (attempt < maxAttempts - 1) {
                    await new Promise(resolve => setTimeout(resolve, delay));
                    delay *= 2;
                }
            }
        }
        throw lastError;
    };
    
    return {
        // Config endpoints
        async getConfig() {
            return request('/distributed/config');
        },
        
        async updateWorker(workerId, data) {
            const apiKey = data?.api_key;
            try {
                return await request('/distributed/config/update_worker', {
                    method: 'POST',
                    body: JSON.stringify({ worker_id: workerId, ...data })
                }, { retry: false, redactErrors: Boolean(apiKey) });
            } catch (error) {
                if (apiKey && typeof error?.message === "string") {
                    error.message = error.message.split(apiKey).join("[redacted]");
                }
                throw error;
            }
        },

        async testServerlessWorker(workerId) {
            return request('/distributed/serverless/test', {
                method: 'POST',
                body: JSON.stringify({ worker_id: workerId }),
            }, { retry: false });
        },

        async getServerlessStatus(workerId) {
            return request(`/distributed/serverless/status/${encodeURIComponent(workerId)}`);
        },

        async cancelServerlessWorker(workerId) {
            return request('/distributed/serverless/cancel', {
                method: 'POST',
                body: JSON.stringify({ worker_id: workerId }),
            }, { retry: false });
        },
        
        async deleteWorker(workerId) {
            return request('/distributed/config/delete_worker', {
                method: 'POST',
                body: JSON.stringify({ worker_id: workerId })
            }, { retry: false });
        },
        
        async updateSetting(key, value) {
            return request('/distributed/config/update_setting', {
                method: 'POST',
                body: JSON.stringify({ key, value })
            }, { retry: false });
        },
        
        async updateMaster(data) {
            return request('/distributed/config/update_master', {
                method: 'POST',
                body: JSON.stringify(data)
            }, { retry: false });
        },
        
        // Worker management endpoints
        async launchWorker(workerId) {
            return request('/distributed/launch_worker', {
                method: 'POST',
                body: JSON.stringify({ worker_id: workerId })
            }, { retry: false });
        },
        
        async stopWorker(workerId) {
            return request('/distributed/stop_worker', {
                method: 'POST',
                body: JSON.stringify({ worker_id: workerId })
            }, { retry: false });
        },
        
        async getManagedWorkers() {
            return request('/distributed/managed_workers');
        },
        
        async getWorkerLog(workerId, lines = 1000) {
            return request(`/distributed/worker_log/${workerId}?lines=${lines}`);
        },

        async getRemoteWorkerLog(workerId, lines = 300) {
            return request(`/distributed/remote_worker_log/${workerId}?lines=${lines}`);
        },
        
        async clearLaunchingFlag(workerId) {
            return request('/distributed/worker/clear_launching', {
                method: 'POST',
                body: JSON.stringify({ worker_id: workerId })
            }, { retry: false });
        },
        
        async queueDistributed(payload) {
            return request('/distributed/queue', {
                method: 'POST',
                headers: {
                    ...(payload?.trace_execution_id
                        ? { 'X-Idempotency-Key': payload.trace_execution_id }
                        : {}),
                },
                body: JSON.stringify(payload)
            }, { retry: false });
        },

        async probeWorker(workerUrl, timeoutMs = TIMEOUTS.STATUS_CHECK, signal = null) {
            const normalizedWorkerUrl = normalizeWorkerUrl(workerUrl);
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
            const effectiveSignal = signal
                ? AbortSignal.any([controller.signal, signal])
                : controller.signal;
            try {
                const response = await fetch(`${normalizedWorkerUrl}/prompt`, {
                    method: 'GET',
                    mode: 'cors',
                    cache: 'no-store',
                    signal: effectiveSignal,
                });

                if (!response.ok) {
                    return { ok: false, status: response.status, queueRemaining: null };
                }

                let data;
                try {
                    data = await response.json();
                } catch {
                    return { ok: false, status: response.status, queueRemaining: null };
                }

                if (!data || typeof data !== "object" || Array.isArray(data)) {
                    return { ok: false, status: response.status, queueRemaining: null };
                }

                const execInfo = data.exec_info;
                if (!execInfo || typeof execInfo !== "object" || Array.isArray(execInfo)) {
                    return { ok: false, status: response.status, queueRemaining: null };
                }

                const rawQueueRemaining = execInfo.queue_remaining;
                const queueRemaining = Number(rawQueueRemaining);
                if (!Number.isFinite(queueRemaining)) {
                    return { ok: false, status: response.status, queueRemaining: null };
                }

                return {
                    ok: true,
                    status: response.status,
                    queueRemaining: Math.max(0, queueRemaining),
                };
            } finally {
                clearTimeout(timeoutId);
            }
        },

        async dispatchToWorker(workerUrl, promptPayload) {
            const normalizedWorkerUrl = normalizeWorkerUrl(workerUrl);
            return requestUrl(`${normalizedWorkerUrl}/prompt`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                mode: 'cors',
                body: JSON.stringify(promptPayload),
            }, { retry: false });
        },
        
        // Network info
        async getNetworkInfo() {
            return request('/distributed/network_info');
        },
        
        // Status checking (with timeout)
        async checkStatus(url, timeout = TIMEOUTS.DEFAULT_FETCH) {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeout);
            
            try {
                const response = await fetch(url, {
                    method: 'GET',
                    mode: 'cors',
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
                
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return await response.json();
            } catch (error) {
                clearTimeout(timeoutId);
                throw error;
            }
        },
        
        // Batch status checking
        async checkMultipleStatuses(urls) {
            return Promise.allSettled(
                urls.map(url => this.checkStatus(url))
            );
        },

        // Cloudflare tunnel management
        async startTunnel() {
            return request('/distributed/tunnel/start', {
                method: 'POST',
                body: JSON.stringify({})
            }, { retry: false });
        },

        async stopTunnel() {
            return request('/distributed/tunnel/stop', {
                method: 'POST',
                body: JSON.stringify({})
            }, { retry: false });
        },

        async getTunnelStatus() {
            return request('/distributed/tunnel/status');
        }
    };
}
