import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { detectMasterIP } from "../masterDetection.js";

describe("automatic local worker ports", () => {
    let originalWindow;
    beforeEach(() => {
        originalWindow = globalThis.window;
        globalThis.window = { location: { hostname: "localhost", port: "443" } };
    });
    afterEach(() => { globalThis.window = originalWindow; });

    function makeExtension(overrides = {}) {
        return {
            config: { workers: [], settings: {}, master: {} },
            log: vi.fn(),
            api: {
                getNetworkInfo: vi.fn().mockResolvedValue({
                    cuda_device: 0, cuda_device_count: 3,
                    master_port: 8189, local_worker_ports: [8191, 8193],
                    ...overrides,
                }),
                updateMaster: vi.fn().mockResolvedValue({}),
                updateWorker: vi.fn().mockResolvedValue({}),
                updateSetting: vi.fn().mockResolvedValue({}),
            },
            ui: { updateMasterDisplay: vi.fn() },
            app: {},
            loadConfig: vi.fn().mockResolvedValue(),
        };
    }

    it("uses server-allocated ports, not hardcoded ports or the browser proxy port", async () => {
        const extension = makeExtension();
        await detectMasterIP(extension);
        expect(extension.config.workers.map(worker => worker.port)).toEqual([8191, 8193]);
        expect(extension.config.workers.map(worker => worker.cuda_device)).toEqual([1, 2]);
        expect(extension.api.updateWorker).toHaveBeenCalledTimes(2);
    });

    it("does not overwrite an existing manually configured worker", async () => {
        const extension = makeExtension();
        const worker = { id: "manual", host: "localhost", port: 8189 };
        extension.config.workers = [worker];
        await detectMasterIP(extension);
        expect(extension.config.workers).toEqual([worker]);
        expect(extension.api.updateWorker).not.toHaveBeenCalled();
    });

    it("fails without partially saving workers if not enough ports are available", async () => {
        const extension = makeExtension({ local_worker_ports: [8191] });
        await detectMasterIP(extension);
        expect(extension.config.workers).toEqual([]);
        expect(extension.api.updateWorker).not.toHaveBeenCalled();
        expect(extension.api.updateSetting).not.toHaveBeenCalled();
        expect(extension.log).toHaveBeenCalledWith(expect.stringContaining("port"), "error");
    });

    it("preserves Runpod worker host and launch settings", async () => {
        globalThis.window.location.hostname = "pod-8189.proxy.runpod.net";
        const extension = makeExtension();
        await detectMasterIP(extension);
        expect(extension.config.workers.map(worker => worker.port)).toEqual([8191, 8193]);
        expect(extension.config.workers.every(worker => worker.host === null && worker.extra_args === "--listen")).toBe(true);
    });
});
