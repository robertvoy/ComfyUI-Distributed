import { app } from "/scripts/app.js";
import { ENDPOINTS } from "./constants.js";

const NODE_CLASS = "DistributedValue";
const DYNAMIC_DEFAULT_WIDGET = "_dv_default";
const DYNAMIC_WORKER_WIDGET_PREFIX = "_dv_worker_";
const WORKERS_CHANGED_EVENT = "distributed:workers-changed";

const trackedNodes = new Set();
let workersChangedListenerAttached = false;

function filterEnabledWorkers(workers) {
    if (!Array.isArray(workers)) return [];
    return workers.filter((worker) => Boolean(worker?.enabled));
}

async function fetchWorkers() {
    try {
        const resp = await fetch(ENDPOINTS.CONFIG);
        if (!resp.ok) return [];
        const config = await resp.json();
        return filterEnabledWorkers(config.workers);
    } catch {
        return [];
    }
}

function getRawDefaultWidget(node) {
    return node.widgets?.find((w) => w.name === "default_value");
}

function getRawWorkerValuesWidget(node) {
    return node.widgets?.find((w) => w.name === "worker_values");
}

function getDynamicDefaultWidget(node) {
    return node.widgets?.find((w) => w.name === DYNAMIC_DEFAULT_WIDGET);
}

function getDynamicWorkerWidgets(node) {
    return (node.widgets || []).filter((w) => w.name.startsWith(DYNAMIC_WORKER_WIDGET_PREFIX));
}

function hideWidgetForGood(widget) {
    if (!widget) return;
    // These are serialized backing fields, not widgets converted into sockets.
    // Keep their native type/serializer and mutate the shared options in place
    // so both the classic canvas and Nodes 2.0 suppress the whole widget row.
    widget.options ??= {};
    widget.options.hidden = true;

    // Hide any attached DOM element (multiline widgets).
    if (widget.element) widget.element.style.display = "none";
    if (widget.inputEl) widget.inputEl.style.display = "none";

    if (widget.linkedWidgets) {
        for (const linked of widget.linkedWidgets) {
            hideWidgetForGood(linked);
        }
    }
}

function hideRawWidgets(node) {
    hideWidgetForGood(getRawDefaultWidget(node));
    hideWidgetForGood(getRawWorkerValuesWidget(node));
}

function removeDynamicDefaultWidget(node) {
    const idx = node.widgets?.findIndex((w) => w.name === DYNAMIC_DEFAULT_WIDGET);
    if (idx != null && idx >= 0) {
        node.widgets.splice(idx, 1);
    }
}

function removeDynamicWorkerWidgets(node) {
    if (!node.widgets) return;
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        if (node.widgets[i].name.startsWith(DYNAMIC_WORKER_WIDGET_PREFIX)) {
            node.widgets.splice(i, 1);
        }
    }
}

function readWorkerStore(node) {
    const raw = getRawWorkerValuesWidget(node);
    if (!raw) return {};
    try {
        const parsed = JSON.parse(raw.value || "{}");
        return typeof parsed === "object" && parsed !== null ? parsed : {};
    } catch {
        return {};
    }
}

function writeWorkerStore(node, store) {
    const raw = getRawWorkerValuesWidget(node);
    if (!raw) return;
    raw.value = JSON.stringify(store);
}

function normalizeComboOptions(options) {
    if (!options) return null;
    if (Array.isArray(options)) return options;
    if (Array.isArray(options.values)) return options.values;
    return null;
}

function resolveGraphLink(graph, linkId) {
    const links = graph.links || graph._links;
    if (!links) return null;
    const link = links[linkId] ?? (typeof links.get === "function" ? links.get(linkId) : null);
    if (!link) return null;
    if (Array.isArray(link)) {
        return {
            target_id: link[2],
            target_slot: link[3],
        };
    }
    return link;
}

function detectTargetType(node) {
    const out = node.outputs?.[0];
    const linkIds = out?.links || [];
    if (!linkIds.length) {
        return { connected: false, type: "STRING", options: null };
    }

    const graph = node.graph || app.graph;
    if (!graph) {
        return { connected: false, type: "STRING", options: null };
    }

    const link = resolveGraphLink(graph, linkIds[0]);
    if (!link) {
        return { connected: false, type: "STRING", options: null };
    }

    const targetNode = graph.getNodeById(link.target_id);
    if (!targetNode) {
        return { connected: false, type: "STRING", options: null };
    }

    const targetInputName = targetNode.inputs?.[link.target_slot]?.name;
    if (!targetInputName) {
        return { connected: false, type: "STRING", options: null };
    }

    const targetWidget = targetNode.widgets?.find((w) => w.name === targetInputName);
    if (targetWidget) {
        if (targetWidget.type === "combo") {
            const comboOptions = normalizeComboOptions(targetWidget.options);
            return { connected: true, type: "COMBO", options: comboOptions };
        }
        if (targetWidget.type === "number") {
            const step = targetWidget.options?.step;
            const precision = targetWidget.options?.precision;
            const isInt = Number.isInteger(step) && (precision === 0 || precision == null);
            return { connected: true, type: isInt ? "INT" : "FLOAT", options: null };
        }
    }

    const nodeDef = targetNode.constructor?.nodeData;
    const inputDef = nodeDef?.input?.required?.[targetInputName] || nodeDef?.input?.optional?.[targetInputName];
    if (inputDef) {
        const defType = inputDef[0];
        if (Array.isArray(defType)) {
            return { connected: true, type: "COMBO", options: defType };
        }
        if (defType === "INT") return { connected: true, type: "INT", options: null };
        if (defType === "FLOAT") return { connected: true, type: "FLOAT", options: null };
    }

    return { connected: true, type: "STRING", options: null };
}

function normalizeNumber(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function getDefaultInitialValue(node, inputType, comboOptions) {
    const rawDefault = getRawDefaultWidget(node);
    const current = rawDefault?.value;

    if (inputType === "INT") {
        return Math.trunc(normalizeNumber(current, 0));
    }
    if (inputType === "FLOAT") {
        return normalizeNumber(current, 0);
    }
    if (inputType === "COMBO" && Array.isArray(comboOptions) && comboOptions.length) {
        const currentText = current == null ? "" : String(current);
        return comboOptions.includes(currentText) ? currentText : comboOptions[0];
    }
    return current == null ? "" : String(current);
}

function setRawDefaultValue(node, value) {
    const rawDefault = getRawDefaultWidget(node);
    if (!rawDefault) return;
    rawDefault.value = value;
}

function serializeWorkerStoreFromWidgets(node, inputType, comboOptions) {
    const nextStore = { _type: inputType };
    if (inputType === "COMBO" && Array.isArray(comboOptions)) {
        nextStore._options = comboOptions;
    }
    const valuesByWorkerId = {};

    for (const widget of getDynamicWorkerWidgets(node)) {
        const key = widget.name.slice(DYNAMIC_WORKER_WIDGET_PREFIX.length);
        if (widget.value !== "" && widget.value !== null && widget.value !== undefined) {
            const value = String(widget.value);
            nextStore[key] = value;
            if (widget._dvWorkerId) {
                valuesByWorkerId[widget._dvWorkerId] = value;
            }
        }
    }
    if (Object.keys(valuesByWorkerId).length) {
        nextStore._by_worker_id = valuesByWorkerId;
    }

    writeWorkerStore(node, nextStore);
}

function updateWorkerStoreTypeMetadata(node, inputType, comboOptions) {
    const store = readWorkerStore(node);
    store._type = inputType;
    if (inputType === "COMBO" && Array.isArray(comboOptions)) {
        store._options = comboOptions;
    } else {
        delete store._options;
    }
    writeWorkerStore(node, store);
}

function createDynamicDefaultWidget(node, inputType, comboOptions) {
    removeDynamicDefaultWidget(node);
    const initial = getDefaultInitialValue(node, inputType, comboOptions);
    let widget;

    if (inputType === "COMBO" && Array.isArray(comboOptions) && comboOptions.length) {
        widget = node.addWidget(
            "combo",
            DYNAMIC_DEFAULT_WIDGET,
            initial,
            (value) => {
                widget.value = value;
                setRawDefaultValue(node, String(value));
            },
            { values: comboOptions }
        );
    } else if (inputType === "INT") {
        widget = node.addWidget(
            "number",
            DYNAMIC_DEFAULT_WIDGET,
            initial,
            (value) => {
                widget.value = Math.trunc(normalizeNumber(value, 0));
                setRawDefaultValue(node, widget.value);
            },
            { min: -Infinity, max: Infinity, step: 1, precision: 0 }
        );
    } else if (inputType === "FLOAT") {
        widget = node.addWidget(
            "number",
            DYNAMIC_DEFAULT_WIDGET,
            initial,
            (value) => {
                widget.value = normalizeNumber(value, 0);
                setRawDefaultValue(node, widget.value);
            },
            { min: -Infinity, max: Infinity, step: 0.1, precision: 3 }
        );
    } else {
        widget = node.addWidget(
            "string",
            DYNAMIC_DEFAULT_WIDGET,
            initial,
            (value) => {
                widget.value = value ?? "";
                setRawDefaultValue(node, widget.value);
            },
            {}
        );
    }

    widget.label = "default_value";
}

function getWorkerInitialValue(store, key, workerId, inputType, comboOptions) {
    const byWorkerId = store?._by_worker_id;
    const saved = (byWorkerId && workerId && byWorkerId[workerId] != null)
        ? byWorkerId[workerId]
        : store[key];
    if (saved == null) {
        if (inputType === "INT" || inputType === "FLOAT") return 0;
        if (inputType === "COMBO" && Array.isArray(comboOptions) && comboOptions.length) {
            return comboOptions[0];
        }
        return "";
    }

    if (inputType === "INT") return Math.trunc(normalizeNumber(saved, 0));
    if (inputType === "FLOAT") return normalizeNumber(saved, 0);
    if (inputType === "COMBO" && Array.isArray(comboOptions) && comboOptions.length) {
        const savedText = String(saved);
        return comboOptions.includes(savedText) ? savedText : comboOptions[0];
    }
    return String(saved);
}

function createWorkerWidgets(node, workers, inputType, comboOptions) {
    removeDynamicWorkerWidgets(node);
    const store = readWorkerStore(node);

    for (let i = 0; i < workers.length; i++) {
        const key = String(i + 1);
        const worker = workers[i];
        const label = worker.name || worker.id || `Worker ${key}`;
        const widgetName = `${DYNAMIC_WORKER_WIDGET_PREFIX}${key}`;
        const initial = getWorkerInitialValue(store, key, worker.id, inputType, comboOptions);
        let widget;

        if (inputType === "COMBO" && Array.isArray(comboOptions) && comboOptions.length) {
            widget = node.addWidget(
                "combo",
                widgetName,
                initial,
                (value) => {
                    widget.value = value;
                    serializeWorkerStoreFromWidgets(node, inputType, comboOptions);
                },
                { values: comboOptions }
            );
        } else if (inputType === "INT") {
            widget = node.addWidget(
                "number",
                widgetName,
                initial,
                (value) => {
                    widget.value = Math.trunc(normalizeNumber(value, 0));
                    serializeWorkerStoreFromWidgets(node, inputType, comboOptions);
                },
                { min: -Infinity, max: Infinity, step: 1, precision: 0 }
            );
        } else if (inputType === "FLOAT") {
            widget = node.addWidget(
                "number",
                widgetName,
                initial,
                (value) => {
                    widget.value = normalizeNumber(value, 0);
                    serializeWorkerStoreFromWidgets(node, inputType, comboOptions);
                },
                { min: -Infinity, max: Infinity, step: 0.1, precision: 3 }
            );
        } else {
            widget = node.addWidget(
                "string",
                widgetName,
                initial,
                (value) => {
                    widget.value = value ?? "";
                    serializeWorkerStoreFromWidgets(node, inputType, comboOptions);
                },
                {}
            );
        }

        widget.label = label;
        widget._dvWorkerId = worker.id;
    }

    serializeWorkerStoreFromWidgets(node, inputType, comboOptions);
}

function rebuildWidgets(node) {
    hideRawWidgets(node);
    const workers = node._dvWorkers || [];
    const store = readWorkerStore(node);
    const detected = detectTargetType(node);
    const disconnected = !detected.connected;
    const inputType = disconnected ? "STRING" : detected.type;
    const comboOptions = disconnected ? null : detected.options;

    if (disconnected) {
        // Reset disconnected node back to the neutral default state.
        setRawDefaultValue(node, "");
        writeWorkerStore(node, { _type: "STRING" });
    }

    createDynamicDefaultWidget(node, inputType, comboOptions);
    if (workers.length > 0) {
        createWorkerWidgets(node, workers, inputType, comboOptions);
    } else {
        removeDynamicWorkerWidgets(node);
        updateWorkerStoreTypeMetadata(node, inputType, comboOptions);
    }

    const size = node.computeSize();
    size[0] = Math.max(size[0], node.size?.[0] || 0);
    node.setSize(size);
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
}

function refreshNodeWorkers(node, workers) {
    if (!node || !node.graph) return;
    node._dvWorkers = workers;
    rebuildWidgets(node);
}

async function refreshTrackedNodes(workers = null) {
    const nextWorkers = workers || (await fetchWorkers());
    for (const node of trackedNodes) {
        refreshNodeWorkers(node, nextWorkers);
    }
}

function attachWorkersChangedListener() {
    if (workersChangedListenerAttached) return;
    if (typeof window === "undefined" || typeof window.addEventListener !== "function") return;

    window.addEventListener(WORKERS_CHANGED_EVENT, (event) => {
        const changedWorkers = filterEnabledWorkers(event?.detail?.workers);
        if (changedWorkers.length > 0 || Array.isArray(event?.detail?.workers)) {
            void refreshTrackedNodes(changedWorkers);
            return;
        }
        void refreshTrackedNodes();
    });

    workersChangedListenerAttached = true;
}

app.registerExtension({
    name: "Distributed.DistributedValue",
    async nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;

        try {
            attachWorkersChangedListener();
            trackedNodes.add(node);
            node._dvWorkers = await fetchWorkers();
            rebuildWidgets(node);

            const originalOnConnectionsChange = node.onConnectionsChange;
            node.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
                if (originalOnConnectionsChange) {
                    originalOnConnectionsChange.call(this, type, index, connected, linkInfo, ioSlot);
                }
                if (type === 2 && index === 0) {
                    setTimeout(() => rebuildWidgets(this), 20);
                }
            };

            const originalConfigure = node.configure;
            node.configure = function (data) {
                const result = originalConfigure ? originalConfigure.call(this, data) : undefined;
                setTimeout(() => rebuildWidgets(this), 20);
                return result;
            };

            const originalOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                trackedNodes.delete(this);
                if (originalOnRemoved) {
                    return originalOnRemoved.call(this);
                }
            };
        } catch (error) {
            console.error("Error in DistributedValue extension:", error);
        }
    },
});
