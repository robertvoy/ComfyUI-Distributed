import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../scripts/app.js", () => ({ app: { registerExtension: vi.fn() } }), { virtual: true });
vi.mock("../../scripts/api.js", () => ({ api: { clientId: "test-client", queuePrompt: vi.fn() } }), { virtual: true });

import { DistributedExtension } from "../main.js";

function makeExtension(worker, updateWorker) {
    return {
        config: { workers: [worker] },
        state: { updateWorker: vi.fn() },
        api: { updateWorker },
        app: { extensionManager: { toast: { add: vi.fn() } } },
        ui: { updateStatusDot: vi.fn() },
        log: vi.fn(),
        _emitWorkersChanged: vi.fn(),
        panelElement: null,
    };
}

describe("worker enable toggle", () => {
    let originalDocument;
    beforeEach(() => {
        originalDocument = globalThis.document;
        globalThis.document = { getElementById: () => null };
    });
    afterEach(() => {
        globalThis.document = originalDocument;
    });

    it("reverts the toggle and explains why when the backend rejects the change", async () => {
        const worker = { id: "w1", type: "serverless", enabled: false };
        const extension = makeExtension(worker, vi.fn().mockRejectedValue(new Error("Save an API key before enabling this worker")));

        await DistributedExtension.prototype.updateWorkerEnabled.call(extension, "w1", true);

        expect(worker.enabled).toBe(false);
        expect(extension.state.updateWorker).toHaveBeenLastCalledWith("w1", { enabled: false });
        expect(extension.app.extensionManager.toast.add).toHaveBeenCalledWith(expect.objectContaining({
            severity: "error", summary: "Could not enable worker", detail: "Save an API key before enabling this worker",
        }));
    });

    it("keeps the new state when the backend accepts it", async () => {
        const worker = { id: "w2", type: "serverless", enabled: false };
        const extension = makeExtension(worker, vi.fn().mockResolvedValue({ status: "success" }));

        await DistributedExtension.prototype.updateWorkerEnabled.call(extension, "w2", true);

        expect(worker.enabled).toBe(true);
        expect(extension.state.updateWorker).toHaveBeenLastCalledWith("w2", { enabled: true });
        expect(extension.app.extensionManager.toast.add).not.toHaveBeenCalled();
    });
});
