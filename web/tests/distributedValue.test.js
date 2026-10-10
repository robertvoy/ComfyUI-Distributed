import { readFileSync } from "node:fs";
import vm from "node:vm";
import { describe, expect, it, vi } from "vitest";

// Evaluate this repository's browser entrypoint without needing a ComfyUI server.
// The real-renderer acceptance additionally checks the input rows in Chromium.
const source = readFileSync(new URL("../distributedValue.js", import.meta.url), "utf8")
    .replace(/^import .*;\r?\n/gm, "");

function setup({ connected = false, targetType = "STRING" } = {}) {
    let extension;
    const listeners = new Map();
    const tasks = [];
    const workers = [
        { id: "gpu-a", name: "GPU A", enabled: true },
        { id: "gpu-b", name: "GPU B", enabled: true },
        { id: "off", name: "Disabled", enabled: false },
    ];
    const target = {
        inputs: [{ name: "value" }],
        widgets: [{ name: "value", type: targetType === "INT" ? "number" : "string", options: { step: 1, precision: 0 } }],
    };
    const graph = { links: { 1: { target_id: 2, target_slot: 0 } }, getNodeById: () => target };
    const raw = ["default_value", "worker_values"].map((name, index) => ({
        name,
        type: "text",
        value: index === 0 ? "saved default" : '{"1":"saved worker"}',
        options: { multiline: false },
        computeSize: vi.fn(() => [200, 20]),
        serializeValue: vi.fn(function () { return this.value; }),
    }));
    const node = {
        comfyClass: "DistributedValue", graph, widgets: [...raw], size: [200, 100],
        inputs: raw.map(w => ({ name: w.name, widget: { name: w.name }, link: null })),
        outputs: [{ links: connected ? [1] : [] }],
        computeSize: () => [200, 100], setSize: vi.fn(), setDirtyCanvas: vi.fn(),
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget);
            return widget;
        },
        configure(data) {
            data.widgets_values.forEach((value, i) => { this.widgets[i].value = value; });
            return "configured";
        },
    };
    vm.runInNewContext(source, {
        app: { graph, registerExtension: ext => { extension = ext; } },
        ENDPOINTS: { CONFIG: "/distributed/config" },
        fetch: async () => ({ ok: true, json: async () => ({ workers }) }),
        window: { addEventListener: (name, handler) => listeners.set(name, handler) },
        setTimeout: callback => { tasks.push(callback); }, console,
    }, { filename: "distributedValue.js" });
    return { node, raw, extension, listeners, tasks, flush: () => { while (tasks.length) tasks.shift()(); } };
}

describe("Distributed Value internal widget visibility", () => {
    it("hides raw fields natively without converting their types or replacing their options", async () => {
        const h = setup();
        const options = h.raw.map(w => w.options);
        const sizes = h.raw.map(w => w.computeSize);
        const serializers = h.raw.map(w => w.serializeValue);
        const slots = h.node.inputs;
        await h.extension.nodeCreated(h.node);

        for (const [i, widget] of h.raw.entries()) {
            expect(widget.type).toBe("text");
            expect(widget.options).toBe(options[i]);
            expect(widget.options.hidden).toBe(true);
            expect(widget.computeSize).toBe(sizes[i]);
            expect(widget.serializeValue).toBe(serializers[i]);
            expect(h.node.widgets[i]).toBe(widget);
        }
        expect(h.node.inputs).toBe(slots);
        expect(h.node.widgets.slice(2).map(w => w.label)).toEqual(["default_value", "GPU A", "GPU B"]);
        expect(h.node.widgets.slice(2).every(w => !w.options.hidden)).toBe(true);
    });

    it("keeps typed edits in the original serialized fields across rebuild/configure", async () => {
        const h = setup({ connected: true, targetType: "INT" });
        await h.extension.nodeCreated(h.node);
        h.node.widgets.find(w => w.name === "_dv_default").callback(42);
        h.node.widgets.find(w => w.name === "_dv_worker_1").callback(7);
        h.node.widgets.find(w => w.name === "_dv_worker_2").callback(9);
        const values = h.raw.map(w => w.serializeValue());
        expect(values[0]).toBe(42);
        expect(JSON.parse(values[1])).toMatchObject({
            _type: "INT", 1: "7", 2: "9", _by_worker_id: { "gpu-a": "7", "gpu-b": "9" },
        });
        expect(h.node.configure({ widgets_values: values })).toBe("configured");
        h.flush();
        expect(h.raw.map(w => w.serializeValue())).toEqual(values);
        expect(h.raw.every(w => w.options.hidden && w.type === "text")).toBe(true);
        expect(h.node.widgets.slice(2).map(w => w.value)).toEqual([42, 7, 9]);
    });

    it("reapplies visibility on worker refresh without duplicating raw or dynamic fields", async () => {
        const h = setup();
        await h.extension.nodeCreated(h.node);
        h.raw[0].options.hidden = false;
        await h.listeners.get("distributed:workers-changed")({ detail: { workers: [
            { id: "gpu-b", name: "Renamed B", enabled: true },
        ] } });
        expect(h.raw.every(w => w.options.hidden)).toBe(true);
        expect(h.node.widgets.map(w => w.name)).toEqual(["default_value", "worker_values", "_dv_default", "_dv_worker_1"]);
        expect(h.node.widgets[3].label).toBe("Renamed B");
    });

    it("hides linked backing widgets and handles absent options without altering serializers", async () => {
        const h = setup();
        delete h.raw[1].options;
        const linked = { type: "text", options: {}, serializeValue: vi.fn(() => "linked") };
        h.raw[0].linkedWidgets = [linked];
        await h.extension.nodeCreated(h.node);
        expect(h.raw[1].options.hidden).toBe(true);
        expect(linked.options.hidden).toBe(true);
        expect(linked.type).toBe("text");
        expect(linked.serializeValue()).toBe("linked");
    });
});
