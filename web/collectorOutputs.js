import { app } from "/scripts/app.js";
import { ENDPOINTS, NODE_CLASSES } from "./constants.js";

// worker_N previews the Nth configured worker (see nodes/collector.py).
const WORKER_OUTPUT_PREFIX = "worker_";
const WORKERS_CHANGED_EVENT = "distributed:workers-changed";

const trackedNodes = new Set();
let workers = [];
let listenerAttached = false;

async function fetchWorkers() {
    try {
        const resp = await fetch(ENDPOINTS.CONFIG);
        if (!resp.ok) return [];
        const config = await resp.json();
        return Array.isArray(config.workers) ? config.workers : [];
    } catch {
        return [];
    }
}

// Only the displayed label changes; slot names stay fixed so links survive.
function labelWorkerOutputs(node, configWorkers) {
    for (const output of node.outputs || []) {
        if (!output?.name?.startsWith(WORKER_OUTPUT_PREFIX)) continue;
        const index = parseInt(output.name.slice(WORKER_OUTPUT_PREFIX.length), 10) - 1;
        const worker = configWorkers[index];
        output.label = worker?.name ? `${output.name} (${worker.name})` : output.name;
    }
    node.setDirtyCanvas?.(true, true);
}

function relabelAll() {
    for (const node of trackedNodes) labelWorkerOutputs(node, workers);
}

function attachListener() {
    if (listenerAttached || typeof window === "undefined") return;
    window.addEventListener(WORKERS_CHANGED_EVENT, (event) => {
        if (Array.isArray(event?.detail?.workers)) {
            workers = event.detail.workers;
            relabelAll();
        }
    });
    listenerAttached = true;
}

app.registerExtension({
    name: "Distributed.CollectorOutputs",
    async nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASSES.DISTRIBUTED_COLLECTOR) return;
        try {
            attachListener();
            trackedNodes.add(node);

            const originalConfigure = node.configure;
            node.configure = function (data) {
                const result = originalConfigure ? originalConfigure.call(this, data) : undefined;
                labelWorkerOutputs(this, workers);
                return result;
            };

            const originalOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                trackedNodes.delete(this);
                return originalOnRemoved ? originalOnRemoved.call(this) : undefined;
            };

            if (workers.length === 0) workers = await fetchWorkers();
            labelWorkerOutputs(node, workers);
        } catch (error) {
            console.error("Error in DistributedCollector outputs extension:", error);
        }
    },
});
