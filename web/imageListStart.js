import { app } from "/scripts/app.js";

// Distributed Image List Start declares value_1..value_10 outputs in Python so
// saved links and prompt output indexes stay valid. ComfyUI grows the matching
// `values.value_N` inputs natively (Autogrow) but has no growable outputs, so
// this extension only presents them: it shows outputs for connected inputs,
// copies each input's type onto its output, and moves output links when
// Autogrow compacts inputs after a middle input is disconnected.
// If this script fails, the node still runs; all outputs simply stay visible.

const START_CLASS = "DistributedImageListStart";
const FIXED_OUTPUTS = 4; // image, mask, item_index, context
const MAX_VALUES = 10;
const INPUT_PREFIX = "values.value_";

function valueOrdinal(name) {
    if (typeof name !== "string" || !name.startsWith(INPUT_PREFIX)) return null;
    const ordinal = Number.parseInt(name.slice(INPUT_PREFIX.length), 10);
    return ordinal >= 1 && ordinal <= MAX_VALUES ? ordinal : null;
}

function getLink(graph, id) {
    if (id == null || !graph) return undefined;
    if (typeof graph.getLink === "function") return graph.getLink(id) ?? undefined;
    const links = graph.links;
    return (typeof links?.get === "function" ? links.get(id) : links?.[id]) ?? undefined;
}

function connectedValues(node) {
    const result = new Map();
    for (const input of node.inputs ?? []) {
        const ordinal = valueOrdinal(input.name);
        if (!ordinal) continue;
        const link = getLink(node.graph, input.link);
        if (!link) continue;
        const origin = node.graph.getNodeById?.(link.origin_id);
        const originType = origin?.outputs?.[link.origin_slot]?.type;
        const type = originType && originType !== "*" ? originType : (link.type ?? "*");
        result.set(ordinal, { type, key: `${link.origin_id}:${link.origin_slot}` });
    }
    return result;
}

function outputLinks(node, ordinal) {
    return node.outputs?.[FIXED_OUTPUTS + ordinal - 1]?.links ?? [];
}

function typesCompatible(outputType, inputType) {
    if (!outputType || !inputType || outputType === "*" || inputType === "*") return true;
    const accepted = String(inputType).split(",");
    return String(outputType).split(",").some(type => accepted.includes(type));
}

function moveOutputLinks(node, from, to) {
    const graph = node.graph;
    const targets = outputLinks(node, from)
        .map(id => getLink(graph, id))
        .filter(Boolean)
        .map(link => [graph.getNodeById?.(link.target_id), link.target_slot])
        .filter(([target]) => target);
    // The moved links carry the same value, so give the destination output its
    // type first; LiteGraph refuses connections to a mismatched output type.
    node.outputs[FIXED_OUTPUTS + to - 1].type = node.outputs[FIXED_OUTPUTS + from - 1].type;
    // Connecting a target input replaces its existing link from `from`.
    for (const [target, slot] of targets) node.connect(FIXED_OUTPUTS + to - 1, target, slot);
}

function disconnectValueOutput(node, ordinal) {
    if (outputLinks(node, ordinal).length) node.disconnectOutput(FIXED_OUTPUTS + ordinal - 1);
}

// Autogrow removes a disconnected middle input and shifts later input links
// down by one. Shift the matching outputs' links the same way, and drop the
// removed value's downstream links, so no consumer silently receives a
// different value.
function followCompactedInputs(node, current) {
    const previous = node._distributedValueSources;
    if (!previous) return;
    const now = [];
    for (const [ordinal, info] of current) now[ordinal - 1] = info.key;
    const before = previous;
    if (now.length !== before.length - 1) return;
    let removed = 0;
    while (removed < now.length && now[removed] === before[removed]) removed++;
    // Removing the last value shifts nothing; keep its links so a temporary
    // rewire does not silently drop downstream connections.
    if (removed >= now.length) return;
    for (let i = removed; i < now.length; i++) {
        if (now[i] !== before[i + 1]) return; // Not a single Autogrow compaction.
    }
    disconnectValueOutput(node, removed + 1);
    for (let i = removed + 1; i < before.length; i++) moveOutputLinks(node, i + 1, i);
}

function syncImageListStartOutputs(node) {
    if (!node?.graph || !Array.isArray(node.outputs) || node.outputs.length < FIXED_OUTPUTS) return;
    const current = connectedValues(node);
    followCompactedInputs(node, current);
    const sources = [];
    for (const [ordinal, info] of current) sources[ordinal - 1] = info.key;
    node._distributedValueSources = Array.from(sources);

    let visible = Math.max(0, ...current.keys());
    for (let ordinal = MAX_VALUES; ordinal > visible; ordinal--) {
        // Never drop a wired output; its links keep it visible until removed.
        if (outputLinks(node, ordinal).length) visible = ordinal;
    }
    const listShape = node.outputs[0]?.shape;
    while (node.outputs.length < FIXED_OUTPUTS + visible) {
        const ordinal = node.outputs.length - FIXED_OUTPUTS + 1;
        node.addOutput(`value_${ordinal}`, "*", listShape === undefined ? undefined : { shape: listShape });
    }
    while (node.outputs.length > FIXED_OUTPUTS + visible) node.removeOutput(node.outputs.length - 1);

    for (let ordinal = 1; ordinal <= visible; ordinal++) {
        const output = node.outputs[FIXED_OUTPUTS + ordinal - 1];
        const type = current.get(ordinal)?.type ?? "*";
        output.name = `value_${ordinal}`;
        if (output.type !== type) {
            output.type = type;
            for (const id of [...(output.links ?? [])]) {
                const link = getLink(node.graph, id);
                if (!link) continue;
                const target = node.graph.getNodeById?.(link.target_id);
                const targetType = target?.inputs?.[link.target_slot]?.type;
                if (target && !typesCompatible(type, targetType)) target.disconnectInput(link.target_slot);
                else link.type = type;
            }
        }
    }
    node.setDirtyCanvas?.(true, true);
}

function scheduleSync(node) {
    if (node._distributedValueSyncPending) return;
    node._distributedValueSyncPending = true;
    const run = () => {
        node._distributedValueSyncPending = false;
        try {
            syncImageListStartOutputs(node);
        } catch (error) {
            console.error("[Distributed] Image List Start output sync failed:", error);
        }
    };
    // Run after Autogrow's own animation-frame input compaction.
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => setTimeout(run, 0));
    else setTimeout(run, 0);
}

app.registerExtension({
    name: "Distributed.ImageListStartValues",
    nodeCreated(node) {
        if (node.comfyClass !== START_CLASS) return;
        const onConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (...args) {
            const result = onConnectionsChange?.apply(this, args);
            scheduleSync(this);
            return result;
        };
        const onConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const result = onConfigure?.apply(this, args);
            // A loaded graph is the source of truth for prior wiring.
            this._distributedValueSources = undefined;
            scheduleSync(this);
            return result;
        };
        scheduleSync(node);
    },
});
