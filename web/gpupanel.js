import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

class MultiGPUExtension {
    // Button styling constants
    static BUTTON_STYLES = {
        base: "width: 100%; padding: 8px; font-size: 12px; color: white; border: none; border-radius: 4px; cursor: pointer; transition: background-color 0.2s;",
        clearMemory: "background-color: #555;",
        success: "background-color: #3ca03c;",
        error: "background-color: #c04c4c;"
    };

    constructor() {
        this.config = null;
        this.originalQueuePrompt = api.queuePrompt.bind(api);

        this.loadConfig().then(() => {
            this.registerSidebarTab();
            this.setupInterceptor();
        });
    }

    // --- State & Config Management (Single Source of Truth) ---

    get enabledWorkers() {
        return this.config?.workers?.filter(w => w.enabled) || [];
    }

    get isEnabled() {
        return this.enabledWorkers.length > 0;
    }

    async loadConfig() {
        try {
            const response = await fetch(`${window.location.origin}/multigpu/config`);
            if (response.ok) {
                this.config = await response.json();
                console.log("[MultiGPU] Loaded config:", this.config);
            } else {
                console.error("[MultiGPU] Failed to load config");
                this.config = { workers: [], settings: {} };
            }
        } catch (error) {
            console.error("[MultiGPU] Error loading config:", error);
            this.config = { workers: [], settings: {} };
        }
    }

    async updateWorkerEnabled(workerId, enabled) {
        const worker = this.config.workers.find(w => w.id === workerId);
        if (worker) {
            worker.enabled = enabled;
        }
        
        try {
            await fetch(`${window.location.origin}/multigpu/config/update_worker`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ worker_id: workerId, enabled: enabled })
            });
        } catch (error) {
            console.error("[MultiGPU] Error updating worker:", error);
        }
    }
    
    async _updateSetting(key, value) {
        try {
            await fetch(`${window.location.origin}/multigpu/config/update_setting`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key, value })
            });
        } catch (error) {
            console.error(`[MultiGPU] Error updating setting '${key}':`, error);
        }
    }

    // --- UI Rendering ---

    registerSidebarTab() {
        app.extensionManager.registerSidebarTab({
            id: "multiGPU",
            icon: "pi pi-server",
            title: "Multi-GPU",
            tooltip: "Multi-GPU Control Panel",
            type: "custom",
            render: (el) => this.renderSidebarContent(el)
        });
    }

    _createButton(text, onClick, style) {
        const button = document.createElement("button");
        button.textContent = text;
        button.style.cssText = MultiGPUExtension.BUTTON_STYLES.base + style;
        button.onclick = onClick;
        return button;
    }

    async _handleClearMemory(button) {
        const originalText = button.textContent;
        button.textContent = "Clearing...";
        button.disabled = true;
        
        try {
            const urlsToClear = this.enabledWorkers.map(w => ({ 
                name: w.name, 
                url: `http://${w.host || window.location.hostname}:${w.port}` 
            }));
            
            if (urlsToClear.length === 0) {
                button.textContent = "No Workers";
                button.style.backgroundColor = MultiGPUExtension.BUTTON_STYLES.error.split(':')[1].trim().replace(';', '');
                setTimeout(() => {
                    button.textContent = originalText;
                    button.style.cssText = MultiGPUExtension.BUTTON_STYLES.base + MultiGPUExtension.BUTTON_STYLES.clearMemory;
                }, 3000);
                return;
            }
            
            const promises = urlsToClear.map(target =>
                fetch(`${target.url}/multigpu/clear_memory`, { 
                    method: 'POST', 
                    mode: 'cors'
                })
                    .then(response => ({ ok: response.ok, name: target.name }))
                    .catch(() => ({ ok: false, name: target.name }))
            );
            
            const results = await Promise.all(promises);
            const failures = results.filter(r => !r.ok);
            
            if (failures.length === 0) {
                button.textContent = "Success!";
                button.style.backgroundColor = MultiGPUExtension.BUTTON_STYLES.success.split(':')[1].trim().replace(';', '');
            } else {
                button.textContent = "Error! See Console";
                button.style.backgroundColor = MultiGPUExtension.BUTTON_STYLES.error.split(':')[1].trim().replace(';', '');
                console.error("[MultiGPU] Failed to clear memory on:", failures.map(f => f.name).join(", "));
            }
            
            setTimeout(() => {
                button.textContent = originalText;
                button.style.cssText = MultiGPUExtension.BUTTON_STYLES.base + MultiGPUExtension.BUTTON_STYLES.clearMemory;
            }, 3000);
        } finally {
            button.disabled = false;
        }
    }


    renderSidebarContent(el) {
        el.innerHTML = '';
        const container = document.createElement("div");
        container.style.cssText = "padding: 15px; display: flex; flex-direction: column; height: 100%;";
        
        const title = document.createElement("h3");
        title.textContent = "Multi-GPU Control";
        title.style.cssText = "margin: 0 0 15px 0; padding-bottom: 10px; border-bottom: 1px solid #444;";
        container.appendChild(title);
        
        const gpuSection = document.createElement("div");
        gpuSection.style.cssText = "flex: 1; overflow-y: auto; margin-bottom: 15px;";
        const gpuTitle = document.createElement("h4");
        gpuTitle.textContent = "Available Workers";
        gpuTitle.style.cssText = "margin: 0 0 10px 0; font-size: 14px;";
        gpuSection.appendChild(gpuTitle);
        
        const gpuList = document.createElement("div");
        (this.config?.workers || []).forEach(worker => {
            const gpuDiv = document.createElement("div");
            gpuDiv.style.cssText = "margin-bottom: 8px; padding: 8px; background: #2a2a2a; border-radius: 4px;";
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.id = `gpu-${worker.id}`;
            checkbox.checked = worker.enabled;
            checkbox.onchange = async (e) => {
                await this.updateWorkerEnabled(worker.id, e.target.checked);
                this.updateSummary();
            };
            const label = document.createElement("label");
            label.htmlFor = `gpu-${worker.id}`;
            label.style.cssText = "cursor: pointer; display: flex; align-items: center; gap: 8px;";
            const gpuInfo = document.createElement("span");
            const hostInfo = worker.host ? ` • ${worker.host}` : '';
            const cudaInfo = worker.cuda_device !== undefined ? `CUDA ${worker.cuda_device} • ` : '';
            gpuInfo.innerHTML = `<strong>${worker.name}</strong><br><small style="color: #888;">${cudaInfo}Port ${worker.port}${hostInfo}</small>`;
            label.appendChild(checkbox);
            label.appendChild(gpuInfo);
            gpuDiv.appendChild(label);
            gpuList.appendChild(gpuDiv);
        });
        gpuSection.appendChild(gpuList);
        container.appendChild(gpuSection);
        
        const actionsSection = document.createElement("div");
        actionsSection.style.cssText = "padding-top: 10px; margin-bottom: 15px; border-top: 1px solid #444;";
        
        const clearMemButton = this._createButton("Clear Worker VRAM", (e) => this._handleClearMemory(e.target), MultiGPUExtension.BUTTON_STYLES.clearMemory);
        clearMemButton.title = "Clear VRAM on all enabled worker GPUs (not master)";
        actionsSection.appendChild(clearMemButton);
        
        container.appendChild(actionsSection);

        const summarySection = document.createElement("div");
        summarySection.style.cssText = "border-top: 1px solid #444; padding-top: 10px;";
        const summary = document.createElement("div");
        summary.id = "multigpu-summary";
        summary.style.cssText = "font-size: 11px; color: #888;";
        summarySection.appendChild(summary);
        container.appendChild(summarySection);
        el.appendChild(container);
        this.updateSummary();
    }

    updateSummary() {
        const summaryEl = document.getElementById('multigpu-summary');
        if (summaryEl) {
            const totalGPUs = this.enabledWorkers.length + 1;
            if (this.isEnabled) {
                summaryEl.textContent = `If Collector node is present, total generation = (${totalGPUs} GPUs × Batch Size)`;
            } else {
                summaryEl.textContent = "Only the master GPU will be used.";
            }
        }
    }

    // --- Helper Methods ---

    // --- Core Logic & Execution ---

    setupInterceptor() {
        api.queuePrompt = async (number, prompt) => {
            if (this.isEnabled && this.findNodesByClass(prompt.output, "MultiGPUCollector").length > 0) {
                return await this.executeParallelMultiGPU(prompt);
            }
            return this.originalQueuePrompt(number, prompt);
        };
    }

    async executeParallelMultiGPU(promptWrapper) {
        try {
            const executionPrefix = "exec_" + Date.now(); // Unique ID for this specific execution
            const enabledWorkers = this.enabledWorkers;
            
            // Find all collector nodes in the workflow
            const collectorNodes = this.findNodesByClass(promptWrapper.output, "MultiGPUCollector");
            
            // Map original node IDs to truly unique job IDs for this specific run
            const job_id_map = new Map(collectorNodes.map(node => [node.id, `${executionPrefix}_${node.id}`]));
            
            // Prepare a separate job queue on the backend for each unique job ID
            const preparePromises = Array.from(job_id_map.values()).map(uniqueId => this._prepareMultiGpuJob(uniqueId));
            await Promise.all(preparePromises);

            const jobs = [];
            const participants = ['master', ...enabledWorkers.map(w => w.id)];

            for (const participantId of participants) {
                const jobApiPrompt = this._prepareApiPromptForParticipant(
                    promptWrapper.output, participantId,
                    { 
                        enabled_worker_ids: enabledWorkers.map(w => w.id), 
                        workflow: promptWrapper.workflow,
                        job_id_map: job_id_map // Pass the map of unique IDs
                    }
                );
                
                if (participantId === 'master') {
                    jobs.push({ type: 'master', promptWrapper: { ...promptWrapper, output: jobApiPrompt } });
                } else {
                    const worker = this.config.workers.find(w => w.id === participantId);
                    if (worker) jobs.push({ type: 'worker', worker, prompt: jobApiPrompt, workflow: promptWrapper.workflow });
                }
            }
            
            const result = await this._executeJobs(jobs);
            return result;
        } catch (error) {
            console.error("[MultiGPU] Parallel execution failed:", error);
            alert(`[MultiGPU] Parallel execution failed: ${error.message}`);
            throw error;
        }
    }


    async _executeJobs(jobs) {
        let masterPromptId = null;
        const promises = jobs.map(job => {
            if (job.type === 'master') {
                return this.originalQueuePrompt(0, job.promptWrapper).then(result => {
                    masterPromptId = result;
                    return result;
                });
            } else {
                return this._dispatchToWorker(job.worker, job.prompt, job.workflow);
            }
        });
        await Promise.all(promises);
        return masterPromptId || { "prompt_id": "multi-gpu-job-dispatched" };
    }
    
    // --- Helper Methods ---

    findNodesByClass(apiPrompt, className) {
        return Object.entries(apiPrompt)
            .filter(([, nodeData]) => nodeData.class_type === className)
            .map(([nodeId, nodeData]) => ({ id: nodeId, data: nodeData }));
    }

    /**
     * Find all nodes connected to collector nodes (both upstream and downstream)
     * @param {Object} apiPrompt - The workflow API prompt
     * @param {Array<string>} collectorIds - Array of collector node IDs
     * @returns {Set<string>} Set of node IDs connected to any collector
     */
    findCollectorConnectedNodes(apiPrompt, collectorIds) {
        const connected = new Set(collectorIds); // Include all collectors
        const toProcess = [...collectorIds];
        
        // First, build a map of which nodes output to which nodes
        const outputMap = new Map(); // sourceNode -> Set of target nodes
        for (const [nodeId, node] of Object.entries(apiPrompt)) {
            if (node.inputs) {
                for (const [inputName, inputValue] of Object.entries(node.inputs)) {
                    if (Array.isArray(inputValue) && inputValue.length === 2) {
                        const sourceNodeId = String(inputValue[0]);
                        if (!outputMap.has(sourceNodeId)) {
                            outputMap.set(sourceNodeId, new Set());
                        }
                        outputMap.get(sourceNodeId).add(nodeId);
                    }
                }
            }
        }
        
        // Traverse both upstream and downstream
        while (toProcess.length > 0) {
            const nodeId = toProcess.pop();
            const node = apiPrompt[nodeId];
            
            // Traverse upstream (inputs)
            if (node && node.inputs) {
                for (const [inputName, inputValue] of Object.entries(node.inputs)) {
                    if (Array.isArray(inputValue) && inputValue.length === 2) {
                        const sourceNodeId = String(inputValue[0]);
                        if (!connected.has(sourceNodeId)) {
                            connected.add(sourceNodeId);
                            toProcess.push(sourceNodeId);
                        }
                    }
                }
            }
            
            // Traverse downstream (outputs)
            if (outputMap.has(nodeId)) {
                for (const targetNodeId of outputMap.get(nodeId)) {
                    if (!connected.has(targetNodeId)) {
                        connected.add(targetNodeId);
                        toProcess.push(targetNodeId);
                    }
                }
            }
        }
        
        return connected;
    }

    /**
     * Prune workflow to only include nodes connected to collectors
     * @param {Object} apiPrompt - The full workflow API prompt
     * @param {Array} collectorNodes - Array of collector nodes (optional, will find if not provided)
     * @returns {Object} Pruned API prompt with only required nodes
     */
    pruneWorkflowForWorker(apiPrompt, collectorNodes = null) {
        // Find all collector nodes if not provided
        if (!collectorNodes) {
            collectorNodes = this.findNodesByClass(apiPrompt, "MultiGPUCollector");
        }
        
        if (collectorNodes.length === 0) {
            // No collectors, return full workflow
            return apiPrompt;
        }
        
        // Get all nodes connected to collectors (both upstream and downstream)
        const collectorIds = collectorNodes.map(node => node.id);
        const connectedNodes = this.findCollectorConnectedNodes(apiPrompt, collectorIds);
        
        console.log(`[MultiGPU] Pruning workflow: keeping ${connectedNodes.size} of ${Object.keys(apiPrompt).length} nodes`);
        
        // Create pruned prompt with only required nodes
        const prunedPrompt = {};
        for (const nodeId of connectedNodes) {
            prunedPrompt[nodeId] = JSON.parse(JSON.stringify(apiPrompt[nodeId]));
        }
        
        return prunedPrompt;
    }


    _prepareApiPromptForParticipant(baseApiPrompt, participantId, options = {}) {
        let jobApiPrompt = JSON.parse(JSON.stringify(baseApiPrompt));
        const isMaster = participantId === 'master';
        
        // Find collector nodes once (before pruning)
        const collectorNodes = this.findNodesByClass(jobApiPrompt, "MultiGPUCollector");
        
        // For workers, prune the workflow to only include collector dependencies
        if (!isMaster && collectorNodes.length > 0) {
            jobApiPrompt = this.pruneWorkflowForWorker(jobApiPrompt, collectorNodes);
        }
        
        // Handle MultiGPU distributor nodes
        const distributorNodes = this.findNodesByClass(jobApiPrompt, "MultiGPUDistributor");
        if (distributorNodes.length > 0) {
            console.log(`[MultiGPU] Found ${distributorNodes.length} distributor node(s)`);
        }
        
        for (const distributor of distributorNodes) {
            const { inputs } = jobApiPrompt[distributor.id];
            inputs.is_worker = !isMaster;
            if (!isMaster) {
                const workerIndex = options.enabled_worker_ids.indexOf(participantId);
                inputs.worker_id = `worker_${workerIndex}`;
                console.log(`[MultiGPU] Set distributor ${distributor.id} for worker ${workerIndex}`);
            }
        }
        
        // Handle MultiGPU collector nodes (already found above)
        for (const collector of collectorNodes) {
            const { inputs } = jobApiPrompt[collector.id];
            
            // Get the unique job ID from the map created for this execution
            const uniqueJobId = options.job_id_map ? options.job_id_map.get(collector.id) : collector.id;
            
            // Use the truly unique ID for this execution
            inputs.multi_job_id = uniqueJobId;
            inputs.is_worker = !isMaster;
            if (isMaster) {
                inputs.enabled_worker_ids = JSON.stringify(options.enabled_worker_ids || []);
            } else {
                inputs.master_url = window.location.origin;
                // Also make the worker_job_id unique to prevent potential caching issues
                inputs.worker_job_id = `${uniqueJobId}_worker_${participantId}`;
                inputs.worker_id = participantId;
            }
        }
        
        return jobApiPrompt;
    }

    async _prepareMultiGpuJob(multi_job_id) {
        try {
            await fetch(`${window.location.origin}/multigpu/prepare_job`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ multi_job_id })
            });
        } catch (error) {
            console.error("[MultiGPU] Error preparing job:", error);
            throw error;
        }
    }

    async _dispatchToWorker(worker, prompt, workflow) {
        const workerUrl = `http://${worker.host || window.location.hostname}:${worker.port}`;
        
        const promptToSend = {
            prompt,
            extra_data: { extra_pnginfo: { workflow } },
            client_id: api.clientId
        };
        
        
        try {
            await fetch(`${workerUrl}/prompt`, { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                mode: 'cors', 
                body: JSON.stringify(promptToSend) 
            });
        } catch (e) {
            console.error(`[MultiGPU] Failed to connect to worker ${worker.name} at ${workerUrl}`, e);
        }
    }
}

app.registerExtension({
    name: "MultiGPU.Panel",
    async setup() {
        new MultiGPUExtension();
    }
});