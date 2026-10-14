import { app } from "/scripts/app.js";

// End's Python schema owns all ten optional override slots. Only presentation
// grows here: unlike Start's Autogrow inputs, End keys must never compact, since
// value_3 always overrides source value_3, even if value_2 is disconnected.
const START = "DistributedImageListStart";
const END = "DistributedImageListEnd";
const GET = "DistributedImageListGetValue";
const PIPE = "DISTRIBUTED_IMAGE_LIST_VALUES";
const VALUE_TYPES = "IMAGE,MASK,STRING,INT,FLOAT,BOOLEAN";
const MAX_VALUES = 10;
const pendingGraphs = new WeakSet();

function ordinal(name) {
    const match = /^value_([1-9]|10)$/.exec(name ?? "");
    return match ? Number(match[1]) : null;
}

function getLink(graph, id) {
    if (id == null) return undefined;
    if (typeof graph?.getLink === "function") return graph.getLink(id);
    return graph?.links?.get?.(id) ?? graph?.links?.[id];
}

function sourceOf(node, inputName) {
    const input = node.inputs?.find(slot => slot.name === inputName);
    const link = getLink(node.graph, input?.link);
    const source = link && node.graph?.getNodeById?.(link.origin_id);
    return source ? { node: source, slot: link.origin_slot, link } : null;
}

function isReroute(node) {
    return [node?.comfyClass, node?.type].includes("Reroute");
}

function throughReroutes(source) {
    const seen = new Set();
    while (source && isReroute(source.node)) {
        if (seen.has(source.node)) return null;
        seen.add(source.node);
        source = sourceOf(source.node, source.node.inputs?.[0]?.name);
    }
    return source;
}

function sourceType(source, seen) {
    source = throughReroutes(source);
    if (!source) return "*";
    const { node, slot, link } = source;
    if (node.comfyClass === GET) return getterType(node, seen);
    const type = node.outputs?.[slot]?.type;
    return type && type !== "*" ? type : (link.type || "*");
}

function getterType(node, seen = new Set()) {
    if (seen.has(node)) return "*";
    seen.add(node);
    const name = node.widgets?.find(widget => widget.name === "value_name")?.value;
    if (!ordinal(name)) return "*";
    const endSource = throughReroutes(sourceOf(node, "values"));
    if (endSource?.node.comfyClass !== END || endSource.slot !== 3) return "*";
    const end = endSource.node;
    const override = sourceOf(end, name);
    if (override) return sourceType(override, seen);
    const startSource = throughReroutes(sourceOf(end, "context"));
    if (startSource?.node.comfyClass !== START || startSource.slot !== 3) return "*";
    return sourceType(sourceOf(startSource.node, `values.${name}`), seen);
}

function compatible(outputType, inputType) {
    if (!outputType || !inputType || outputType === "*" || inputType === "*") return true;
    return String(outputType).split(",").some(type => String(inputType).split(",").includes(type));
}

function syncGetter(node) {
    const output = node.outputs?.[0];
    if (!output) return;
    const type = getterType(node);
    if (output.type === type) return;
    output.type = type;
    for (const id of [...(output.links ?? [])]) {
        const link = getLink(node.graph, id);
        if (!link) continue;
        const target = node.graph?.getNodeById?.(link.target_id);
        if (target && !compatible(type, target.inputs?.[link.target_slot]?.type)) {
            target.disconnectInput(link.target_slot);
        } else {
            link.type = type;
        }
    }
    node.setDirtyCanvas?.(true, true);
}

function syncEnd(node) {
    if (!Array.isArray(node.inputs) || !Array.isArray(node.outputs)) return;
    // Old saved End nodes have only the three original outputs. Append rather
    // than replacing any socket, preserving every existing output index/link.
    if (node.outputs.length === 3) {
        const shape = node.outputs[0]?.shape;
        node.addOutput("values", PIPE, shape === undefined ? undefined : { shape });
    }
    const connected = node.inputs.filter(input => ordinal(input.name) && input.link != null);
    const highest = Math.max(0, ...connected.map(input => ordinal(input.name)));
    const visible = Math.min(MAX_VALUES, Math.max(highest + 1, node._distributedVisibleOverrides ?? 1));
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const input = node.inputs[i];
        if (ordinal(input.name) > visible && input.link == null) node.removeInput(i);
    }
    for (let i = 1; i <= visible; i++) {
        if (!node.inputs.some(input => input.name === `value_${i}`)) {
            node.addInput(`value_${i}`, VALUE_TYPES);
        }
    }
    node.setDirtyCanvas?.(true, true);
}

function schedule(graph) {
    if (!graph || pendingGraphs.has(graph)) return;
    pendingGraphs.add(graph);
    const run = () => {
        // Keep the guard while syncing, since disconnectInput can re-enter the
        // connection hook. A fresh user event schedules a fresh traversal.
        try {
            for (const node of graph._nodes ?? []) if (node.comfyClass === END) syncEnd(node);
            for (const node of graph._nodes ?? []) if (node.comfyClass === GET) syncGetter(node);
        } catch (error) {
            console.error("[Distributed] Image List values presentation failed:", error);
        } finally {
            pendingGraphs.delete(graph);
        }
    };
    // Let native Start Autogrow and its existing output-sync finish first.
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => setTimeout(run, 0));
    else setTimeout(run, 0);
}

app.registerExtension({
    name: "Distributed.ImageListEndValues",
    nodeCreated(node) {
        if (![START, END, GET].includes(node.comfyClass) && !isReroute(node)) return;
        if (node.comfyClass === END) {
            node.addWidget("button", "Add value override", null, () => {
                const shown = Math.max(0, ...(node.inputs ?? []).map(input => ordinal(input.name) ?? 0));
                node._distributedVisibleOverrides = Math.min(MAX_VALUES, shown + 1);
                schedule(node.graph);
            }, { serialize: false });
        }
        for (const hook of ["onConnectionsChange", "onConfigure", "onAdded", "onWidgetChanged"]) {
            const original = node[hook];
            node[hook] = function (...args) {
                const result = original?.apply(this, args);
                schedule(this.graph);
                return result;
            };
        }
        // Widget callbacks are also used by frontend versions that do not call
        // the node's onWidgetChanged hook for a combo selection.
        if (node.comfyClass === GET) {
            const widget = node.widgets?.find(widget => widget.name === "value_name");
            if (widget) {
                const callback = widget.callback;
                widget.callback = function (...args) {
                    const result = callback?.apply(this, args);
                    schedule(node.graph);
                    return result;
                };
            }
        }
        schedule(node.graph);
    },
    loadedGraphNode(node) {
        schedule(node.graph);
    },
});
