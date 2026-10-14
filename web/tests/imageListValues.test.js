import { readFileSync } from "node:fs";
import vm from "node:vm";
import { describe, expect, it } from "vitest";

const source = readFileSync(new URL("../imageListValues.js", import.meta.url), "utf8")
    .replace(/^import .*;\r?\n/gm, "");
const PIPE = "DISTRIBUTED_IMAGE_LIST_VALUES";

function setup() {
    let extension;
    let nextId = 1;
    const tasks = [];
    const graph = {
        _nodes: [], links: new Map(),
        getNodeById(id) { return this._nodes.find(node => node.id === id); },
        getLink(id) { return this.links.get(id); },
    };
    vm.runInNewContext(source, {
        app: { registerExtension: ext => { extension = ext; } },
        setTimeout: callback => tasks.push(callback), console,
    });
    function node(comfyClass, inputs, outputs, widgets = []) {
        const n = {
            id: nextId++, comfyClass, graph, widgets,
            inputs: inputs.map(name => ({ name, type: "*", link: null })),
            outputs: outputs.map(([name, type]) => ({ name, type, links: [], shape: 6 })),
            addWidget(type, name, value, callback, options) { this.widgets.push({ type, name, value, callback, options }); },
            addInput(name, type) { this.inputs.push({ name, type, link: null }); },
            addOutput(name, type, extra) { this.outputs.push({ name, type, links: [], ...extra }); },
            removeInput(slot) {
                if (this.inputs[slot].link != null) throw new Error("Removed wired input");
                this.inputs.splice(slot, 1);
                this.inputs.forEach((input, index) => {
                    if (input.link != null) graph.links.get(input.link).target_slot = index;
                });
            },
            disconnectInput(slot) {
                const input = this.inputs[slot];
                const link = graph.links.get(input.link);
                if (link) {
                    const out = graph.getNodeById(link.origin_id).outputs[link.origin_slot];
                    out.links = out.links.filter(id => id !== link.id);
                    graph.links.delete(link.id);
                }
                input.link = null;
                this.onConnectionsChange?.();
            },
            setDirtyCanvas() {},
        };
        graph._nodes.push(n);
        extension.nodeCreated(n);
        return n;
    }
    const flush = () => { while (tasks.length) tasks.shift()(); };
    const connect = (from, slot, to, name) => {
        const targetSlot = to.inputs.findIndex(input => input.name === name);
        if (targetSlot < 0) throw new Error(`Missing input ${name}`);
        to.disconnectInput(targetSlot);
        const id = nextId++;
        graph.links.set(id, { id, origin_id: from.id, origin_slot: slot, target_id: to.id,
            target_slot: targetSlot, type: from.outputs[slot].type });
        from.outputs[slot].links.push(id);
        to.inputs[targetSlot].link = id;
        from.onConnectionsChange?.();
        to.onConnectionsChange?.();
        return id;
    };
    const start = node("DistributedImageListStart", ["images", "mask", "values.value_1", "values.value_2", "values.value_3"],
        [["image", "IMAGE"], ["mask", "MASK"], ["item_index", "INT"], ["context", "DISTRIBUTED_IMAGE_LIST_CONTEXT"]]);
    const end = node("DistributedImageListEnd", ["images", "masks", "texts", "context", ...Array.from({ length: 10 }, (_, i) => `value_${i + 1}`)],
        [["image_list", "IMAGE"], ["masks", "MASK"], ["texts", "STRING"], ["values", PIPE]]);
    const get = node("DistributedImageListGetValue", ["values"], [["value", "*"]], [{ name: "value_name", value: "value_1" }]);
    connect(start, 3, end, "context");
    connect(end, 3, get, "values");
    flush();
    return { node, graph, start, end, get, flush, connect, extension };
}

const valueNames = node => node.inputs.filter(input => input.name.startsWith("value_")).map(input => input.name);
const addOverride = h => { h.end.widgets.find(widget => widget.name === "Add value override").callback(); h.flush(); };

describe("Image List End context values", () => {
    it("keeps original sockets and shows only the first optional override initially", () => {
        const h = setup();
        expect(h.end.inputs.map(input => input.name)).toEqual(["images", "masks", "texts", "context", "value_1"]);
        expect(h.end.outputs.map(output => output.name)).toEqual(["image_list", "masks", "texts", "values"]);
        expect(h.end.widgets[0].options.serialize).toBe(false);
    });

    it("appends the values output when loading an old three-output End", () => {
        const h = setup();
        const old = h.node("DistributedImageListEnd", ["images", "masks", "texts", "context"],
            [["image_list", "IMAGE"], ["masks", "MASK"], ["texts", "STRING"]]);
        h.flush();
        expect(old.outputs[3]).toMatchObject({ name: "values", type: PIPE, shape: 6 });
        expect(valueNames(old)).toEqual(["value_1"]);
    });

    it("grows overrides without compacting field identities or existing links", () => {
        const h = setup();
        addOverride(h);
        addOverride(h);
        const src = h.node("PrimitiveInt", [], [["INT", "INT"]]);
        const thirdLink = h.connect(src, 0, h.end, "value_3");
        h.connect(src, 0, h.end, "value_2");
        h.flush();
        h.end.disconnectInput(h.end.inputs.findIndex(input => input.name === "value_2"));
        h.flush();
        expect(valueNames(h.end)).toEqual(["value_1", "value_2", "value_3", "value_4"]);
        const link = h.graph.links.get(thirdLink);
        expect(h.end.inputs[link.target_slot].name).toBe("value_3");
        for (let i = 0; i < 20; i++) addOverride(h);
        expect(valueNames(h.end)).toHaveLength(10);
    });

    it.each(["IMAGE", "MASK", "STRING", "INT", "FLOAT", "BOOLEAN"])("infers %s from the original Start value through context", type => {
        const h = setup();
        const source = h.node("Source", [], [["value", type]]);
        h.connect(source, 0, h.start, "values.value_1");
        h.flush();
        expect(h.get.outputs[0].type).toBe(type);
    });

    it("prefers the End override type, then returns to the original after disconnect", () => {
        const h = setup();
        h.connect(h.node("Source", [], [["value", "INT"]]), 0, h.start, "values.value_1");
        h.connect(h.node("Source", [], [["value", "STRING"]]), 0, h.end, "value_1");
        h.flush();
        expect(h.get.outputs[0].type).toBe("STRING");
        h.end.disconnectInput(h.end.inputs.findIndex(input => input.name === "value_1"));
        h.flush();
        expect(h.get.outputs[0].type).toBe("INT");
    });

    it("updates a selected field and disconnects incompatible consumers", () => {
        const h = setup();
        h.connect(h.node("Source", [], [["value", "STRING"]]), 0, h.start, "values.value_1");
        h.connect(h.node("Source", [], [["value", "INT"]]), 0, h.start, "values.value_2");
        h.flush();
        const consumer = h.node("Save", ["filename"], []);
        consumer.inputs[0].type = "STRING";
        h.connect(h.get, 0, consumer, "filename");
        h.get.widgets[0].value = "value_2";
        h.get.widgets[0].callback();
        h.flush();
        expect(h.get.outputs[0].type).toBe("INT");
        expect(consumer.inputs[0].link).toBe(null);
    });

    it("keeps compatible consumer links and uses wildcard for unavailable graph metadata", () => {
        const h = setup();
        h.connect(h.node("Source", [], [["value", "INT"]]), 0, h.start, "values.value_1");
        h.flush();
        const consumer = h.node("Consumer", ["value"], []);
        consumer.inputs[0].type = "INT,FLOAT";
        const id = h.connect(h.get, 0, consumer, "value");
        h.get.disconnectInput(0);
        h.flush();
        expect(h.get.outputs[0].type).toBe("*");
        expect(h.graph.links.has(id)).toBe(true);
    });

    it("follows reroutes for context and values, and terminates on a cycle", () => {
        const h = setup();
        h.connect(h.node("Source", [], [["value", "FLOAT"]]), 0, h.start, "values.value_1");
        const reroute = h.node("Reroute", ["input"], [["output", "*"]]);
        h.connect(h.end, 3, reroute, "input");
        h.connect(reroute, 0, h.get, "values");
        h.flush();
        expect(h.get.outputs[0].type).toBe("FLOAT");
        h.connect(reroute, 0, reroute, "input");
        h.flush();
        expect(h.get.outputs[0].type).toBe("*");
    });

    it("recomputes types across sequential regions", () => {
        const h = setup();
        const secondStart = h.node("DistributedImageListStart", ["values.value_1"],
            [["image", "IMAGE"], ["mask", "MASK"], ["item_index", "INT"], ["context", "DISTRIBUTED_IMAGE_LIST_CONTEXT"]]);
        const secondEnd = h.node("DistributedImageListEnd", ["images", "masks", "texts", "context"],
            [["image_list", "IMAGE"], ["masks", "MASK"], ["texts", "STRING"], ["values", PIPE]]);
        const secondGet = h.node("DistributedImageListGetValue", ["values"], [["value", "*"]], [{ name: "value_name", value: "value_1" }]);
        h.connect(h.node("Source", [], [["value", "BOOLEAN"]]), 0, h.start, "values.value_1");
        h.connect(h.get, 0, secondStart, "values.value_1");
        h.connect(secondStart, 3, secondEnd, "context");
        h.connect(secondEnd, 3, secondGet, "values");
        h.flush();
        expect(secondGet.outputs[0].type).toBe("BOOLEAN");
    });
});
