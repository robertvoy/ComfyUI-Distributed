import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getWorkerUrl, stopWorker } from "../workerLifecycle.js";


describe("workerLifecycle URL wiring", () => {
    let originalWindow;

    beforeEach(() => {
        originalWindow = globalThis.window;
        globalThis.window = {
            location: {
                hostname: "127.0.0.1",
                protocol: "http:",
                port: "8190",
                origin: "http://127.0.0.1:8190",
            },
        };
    });

    afterEach(() => {
        globalThis.window = originalWindow;
    });

    // URL variants are covered in urlUtils.test.js; keep the wrapper wiring here.
    it("forwards the worker and endpoint using window.location", () => {
        const worker = { id: "w1", port: 8189, type: "local" };
        expect(getWorkerUrl({}, worker, "/prompt")).toBe("http://127.0.0.1:8189/prompt");
    });
});

describe("stopWorker failure replies", () => {
    let originalDocument;

    beforeEach(() => {
        vi.useFakeTimers();
        originalDocument = globalThis.document;
    });

    afterEach(() => {
        vi.useRealTimers();
        globalThis.document = originalDocument;
    });

    // The backend answers a failed stop with an HTTP error, which apiClient
    // turns into a rejected promise carrying the backend's message.
    function setup(stopErrorMessage) {
        const stopBtn = {
            disabled: false,
            textContent: "Stop",
            classList: { add: vi.fn(), remove: vi.fn() },
        };
        globalThis.document = {
            getElementById: (id) => (id === "stop-w1" ? stopBtn : null),
            querySelectorAll: () => [],
        };
        const managed = new Map([["w1", { pid: 1234 }]]);
        const statuses = new Map();
        const extension = {
            config: { workers: [{ id: "w1", name: "GPU 1", type: "local", enabled: true }] },
            api: { stopWorker: vi.fn().mockRejectedValue(new Error(stopErrorMessage)) },
            state: {
                setWorkerManaged: (id, info) => managed.set(id, info),
                setWorkerStatus: (id, status) => statuses.set(id, status),
            },
            ui: { updateStatusDot: vi.fn() },
            log: vi.fn(),
        };
        return { extension, stopBtn, managed, statuses };
    }

    it("treats 409 'Worker already stopped' as not running and marks the worker offline", async () => {
        const h = setup("Worker already stopped");
        await stopWorker(h.extension, "w1");
        expect(h.stopBtn.textContent).toBe("Not Running");
        expect(h.managed.get("w1")).toBeNull();
        expect(h.statuses.get("w1")).toEqual({ online: false });
        expect(h.extension.ui.updateStatusDot).toHaveBeenCalledWith("w1", expect.any(String), "Offline");
    });

    it("drops stale managed state on 404 'Worker not managed by UI' without assuming offline", async () => {
        const h = setup("Worker not managed by UI");
        await stopWorker(h.extension, "w1");
        expect(h.stopBtn.textContent).toBe("Not Running");
        expect(h.managed.get("w1")).toBeNull();
        expect(h.statuses.has("w1")).toBe(false);
    });

    it("reports other failures as errors and keeps the managed state", async () => {
        const h = setup("Error stopping worker: access denied");
        await stopWorker(h.extension, "w1");
        expect(h.stopBtn.textContent).toBe("Error");
        expect(h.managed.get("w1")).toEqual({ pid: 1234 });
    });
});
