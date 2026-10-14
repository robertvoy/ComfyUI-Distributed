import { readFileSync } from "node:fs";
import vm from "node:vm";
import { describe, expect, it } from "vitest";

const source = readFileSync(new URL("../imageListStart.js", import.meta.url), "utf8")
    .replace(/^import .*;\r?\n/gm, "");

// Minimal LiteGraph-like graph: enough link bookkeeping to exercise output
// add/remove, retyping, and link moves the extension performs.
function setup() {
    let extension;
    const tasks = [];
    const links = new Map();
    let nextLink = 1;
    const nodes = new Map();
    const graph = {
        links,
        getLink: id => links.get(id),
        getNodeById: id => nodes.get(id),
    };
    const makeSource = (id, type) => {
        const node = { id, outputs: [{ type, links: [] }] };
        nodes.set(id, node);
        return node;
    };
    const makeTarget = (id, type) => {
        const node = {
            id, inputs: [{ type, link: null }],
            disconnectInput(slot) {
                const link = links.get(this.inputs[slot].link);
                if (!link) return;
                const origin = nodes.get(link.origin_id);
                origin.outputs[link.origin_slot].links =
                    origin.outputs[link.origin_slot].links.filter(l => l !== link.id);
                links.delete(link.id);
                this.inputs[slot].link = null;
            },
        };
        nodes.set(id, node);
        return node;
    };
    const fixed = ["image", "mask", "item_index", "context"].map(name => ({ name, type: "IMAGE", links: [], shape: 6 }));
    const start = {
        id: 1, comfyClass: "DistributedImageListStart", graph,
        inputs: [{ name: "images", link: null }, { name: "mask", link: null }, { name: "values.value_1", link: null }],
        outputs: [...fixed, ...Array.from({ length: 10 }, (_, i) => ({ name: `value_${i + 1}`, type: "*", links: [] }))],
        addOutput(name, type, extra) { this.outputs.push({ name, type, links: [], ...(extra ?? {}) }); },
        removeOutput(slot) {
            if (this.outputs[slot].links.length) throw new Error("removed a wired output");
            this.outputs.splice(slot, 1);
        },
        connect(slot, target, targetSlot) {
            // LiteGraph refuses a link whose output type does not fit the input.
            const outType = this.outputs[slot].type;
            const inType = target.inputs[targetSlot].type;
            if (outType !== "*" && inType !== "*" && !String(inType).split(",").includes(outType)) return null;
            target.disconnectInput(targetSlot);
            const id = nextLink++;
            links.set(id, { id, origin_id: this.id, origin_slot: slot, target_id: target.id,
                            target_slot: targetSlot, type: this.outputs[slot].type });
            this.outputs[slot].links.push(id);
            target.inputs[targetSlot].link = id;
            return links.get(id);
        },
        disconnectOutput(slot) {
            for (const id of [...this.outputs[slot].links]) {
                const link = links.get(id);
                nodes.get(link.target_id).disconnectInput(link.target_slot);
            }
        },
        setDirtyCanvas() {},
    };
    nodes.set(1, start);
    vm.runInNewContext(source, {
        app: { registerExtension: ext => { extension = ext; } },
        setTimeout: callback => { tasks.push(callback); },
        console,
    }, { filename: "imageListStart.js" });
    extension.nodeCreated(start);
    const flush = () => { while (tasks.length) tasks.shift()(); };
    // Wire a source output into Start's value input, as Autogrow would.
    const connectValue = (ordinal, sourceNode) => {
        const name = `values.value_${ordinal}`;
        let input = start.inputs.find(i => i.name === name);
        if (!input) { input = { name, link: null }; start.inputs.push(input); }
        const id = nextLink++;
        links.set(id, { id, origin_id: sourceNode.id, origin_slot: 0, target_id: 1,
                        target_slot: start.inputs.indexOf(input), type: sourceNode.outputs[0].type });
        sourceNode.outputs[0].links.push(id);
        input.link = id;
        if (!start.inputs.some(i => i.name === `values.value_${ordinal + 1}`)) {
            start.inputs.push({ name: `values.value_${ordinal + 1}`, link: null });
        }
        start.onConnectionsChange(1, start.inputs.indexOf(input), true);
        flush();
    };
    // Mimic Autogrow's compaction after disconnecting a middle input.
    const disconnectValueAndCompact = ordinal => {
        const valueInputs = start.inputs.filter(i => i.name.startsWith("values."));
        const linksBefore = valueInputs.map(i => i.link);
        links.delete(linksBefore[ordinal - 1]);
        const remaining = linksBefore.filter((l, i) => i !== ordinal - 1 && l != null);
        start.inputs = start.inputs.filter(i => !i.name.startsWith("values."));
        remaining.forEach((l, i) => start.inputs.push({ name: `values.value_${i + 1}`, link: l }));
        start.inputs.push({ name: `values.value_${remaining.length + 1}`, link: null });
        start.onConnectionsChange(1, 0, false);
        flush();
    };
    flush();
    return { start, makeSource, makeTarget, connectValue, disconnectValueAndCompact, links, flush };
}

const valueOutputs = start => start.outputs.slice(4).map(o => `${o.name}:${o.type}`);

describe("Distributed Image List Start value outputs", () => {
    it("shows no value outputs until a value input is connected", () => {
        const h = setup();
        expect(h.start.outputs.map(o => o.name)).toEqual(["image", "mask", "item_index", "context"]);
    });

    it("adds one typed output per connected value input", () => {
        const h = setup();
        h.connectValue(1, h.makeSource(10, "IMAGE"));
        h.connectValue(2, h.makeSource(11, "STRING"));
        expect(valueOutputs(h.start)).toEqual(["value_1:IMAGE", "value_2:STRING"]);
        expect(h.start.outputs[4].shape).toBe(6);
    });

    it("keeps downstream wiring on the same value when a middle input is removed", () => {
        const h = setup();
        h.connectValue(1, h.makeSource(10, "IMAGE"));
        h.connectValue(2, h.makeSource(11, "STRING"));
        h.connectValue(3, h.makeSource(12, "INT"));
        const imageUser = h.makeTarget(20, "IMAGE");
        const stringUser = h.makeTarget(21, "STRING");
        const intUser = h.makeTarget(22, "INT");
        h.start.connect(4, imageUser, 0);
        h.start.connect(5, stringUser, 0);
        h.start.connect(6, intUser, 0);

        h.disconnectValueAndCompact(2);

        expect(valueOutputs(h.start)).toEqual(["value_1:IMAGE", "value_2:INT"]);
        expect(h.links.get(imageUser.inputs[0].link).origin_slot).toBe(4);
        expect(h.links.get(intUser.inputs[0].link).origin_slot).toBe(5);
        expect(stringUser.inputs[0].link).toBe(null);
    });

    it("disconnects a consumer whose type no longer matches the input", () => {
        const h = setup();
        h.connectValue(1, h.makeSource(10, "IMAGE"));
        const imageUser = h.makeTarget(20, "IMAGE");
        h.start.connect(4, imageUser, 0);
        // Replace value_1's source with a STRING source.
        const input = h.start.inputs.find(i => i.name === "values.value_1");
        h.links.delete(input.link);
        input.link = null;
        h.connectValue(1, h.makeSource(11, "STRING"));
        expect(valueOutputs(h.start)).toEqual(["value_1:STRING"]);
        expect(imageUser.inputs[0].link).toBe(null);
    });

    it("never removes an output that still has links", () => {
        const h = setup();
        h.connectValue(1, h.makeSource(10, "IMAGE"));
        h.start.connect(4, h.makeTarget(20, "*"), 0);
        const input = h.start.inputs.find(i => i.name === "values.value_1");
        h.links.delete(input.link);
        input.link = null;
        h.start.onConnectionsChange(1, 2, false);
        h.flush();
        expect(valueOutputs(h.start)).toEqual(["value_1:*"]);
    });
});
