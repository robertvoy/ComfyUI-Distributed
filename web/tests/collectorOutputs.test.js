import { readFileSync } from "node:fs";
import vm from "node:vm";
import { describe, expect, it } from "vitest";

const source = readFileSync(new URL("../collectorOutputs.js", import.meta.url), "utf8")
    .replace(/^import .*;\r?\n/gm, "");

function setup(workers) {
    let extension;
    const listeners = new Map();
    vm.runInNewContext(source, {
        app: { registerExtension: (ext) => { extension = ext; } },
        ENDPOINTS: { CONFIG: "/distributed/config" },
        NODE_CLASSES: { DISTRIBUTED_COLLECTOR: "DistributedCollector" },
        fetch: async () => ({ ok: true, json: async () => ({ workers }) }),
        window: { addEventListener: (name, fn) => listeners.set(name, fn) },
        parseInt,
        Array,
        console,
    });
    const node = {
        comfyClass: "DistributedCollector",
        outputs: ["images", "audio", "master", "worker_1", "worker_2", "worker_3"].map((name) => ({ name })),
    };
    return { extension, listeners, node };
}

describe("DistributedCollector per-GPU outputs", () => {
    it("labels worker_N with the Nth configured worker and keeps slot names", async () => {
        const { extension, node } = setup([{ id: "a", name: "GPU A", enabled: false }, { id: "b", name: "GPU B" }]);
        await extension.nodeCreated(node);
        expect(node.outputs.map((o) => o.name)).toEqual(
            ["images", "audio", "master", "worker_1", "worker_2", "worker_3"]);
        expect(node.outputs.slice(3).map((o) => o.label)).toEqual(
            ["worker_1 (GPU A)", "worker_2 (GPU B)", "worker_3"]);
        expect(node.outputs[2].label).toBeUndefined();
    });

    it("relabels when the worker list changes", async () => {
        const { extension, listeners, node } = setup([{ id: "a", name: "GPU A" }]);
        await extension.nodeCreated(node);
        listeners.get("distributed:workers-changed")({ detail: { workers: [{ id: "c", name: "Renamed" }] } });
        expect(node.outputs[3].label).toBe("worker_1 (Renamed)");
    });
});
