import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createApiClient } from "../apiClient.js";

describe("apiClient probeWorker", () => {
    let originalFetch;

    beforeEach(() => {
        originalFetch = globalThis.fetch;
        globalThis.fetch = vi.fn();
    });

    afterEach(() => {
        globalThis.fetch = originalFetch;
        vi.restoreAllMocks();
    });

    it("returns ok=true when /prompt returns valid exec_info payload", async () => {
        globalThis.fetch.mockResolvedValue({
            ok: true,
            status: 200,
            json: vi.fn().mockResolvedValue({ exec_info: { queue_remaining: 2 } }),
        });

        const client = createApiClient("http://127.0.0.1:8188");
        const result = await client.probeWorker("http://worker.local:8190", 1000);

        expect(result).toEqual({ ok: true, status: 200, queueRemaining: 2 });
    });

    it("returns ok=false on non-200 responses", async () => {
        globalThis.fetch.mockResolvedValue({
            ok: false,
            status: 503,
            json: vi.fn(),
        });

        const client = createApiClient("http://127.0.0.1:8188");
        const result = await client.probeWorker("http://worker.local:8190", 1000);

        expect(result).toEqual({ ok: false, status: 503, queueRemaining: null });
    });

    it("returns ok=false when response JSON is invalid", async () => {
        globalThis.fetch.mockResolvedValue({
            ok: true,
            status: 200,
            json: vi.fn().mockRejectedValue(new Error("invalid json")),
        });

        const client = createApiClient("http://127.0.0.1:8188");
        const result = await client.probeWorker("http://worker.local:8190", 1000);

        expect(result).toEqual({ ok: false, status: 200, queueRemaining: null });
    });

    it("returns ok=false when exec_info is missing", async () => {
        globalThis.fetch.mockResolvedValue({
            ok: true,
            status: 200,
            json: vi.fn().mockResolvedValue({}),
        });

        const client = createApiClient("http://127.0.0.1:8188");
        const result = await client.probeWorker("http://worker.local:8190", 1000);

        expect(result).toEqual({ ok: false, status: 200, queueRemaining: null });
    });

    it("returns ok=false when queue_remaining is not numeric", async () => {
        globalThis.fetch.mockResolvedValue({
            ok: true,
            status: 200,
            json: vi.fn().mockResolvedValue({ exec_info: { queue_remaining: "n/a" } }),
        });

        const client = createApiClient("http://127.0.0.1:8188");
        const result = await client.probeWorker("http://worker.local:8190", 1000);

        expect(result).toEqual({ ok: false, status: 200, queueRemaining: null });
    });

    it("clamps negative queue_remaining to zero", async () => {
        globalThis.fetch.mockResolvedValue({
            ok: true,
            status: 200,
            json: vi.fn().mockResolvedValue({ exec_info: { queue_remaining: -5 } }),
        });

        const client = createApiClient("http://127.0.0.1:8188");
        const result = await client.probeWorker("http://worker.local:8190", 1000);

        expect(result).toEqual({ ok: true, status: 200, queueRemaining: 0 });
    });
});

describe("apiClient request retries", () => {
    let originalFetch;

    beforeEach(() => {
        originalFetch = globalThis.fetch;
        globalThis.fetch = vi.fn();
        vi.spyOn(console, "log").mockImplementation(() => {});
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
        globalThis.fetch = originalFetch;
        vi.restoreAllMocks();
    });

    const reply = (status, body = {}) => ({
        ok: status >= 200 && status < 300,
        status,
        json: vi.fn().mockResolvedValue(body),
    });

    it("does not retry a 4xx answer and keeps its message and status", async () => {
        globalThis.fetch.mockResolvedValue(reply(404, { ok: false, message: "Serverless worker not found" }));
        const client = createApiClient("http://127.0.0.1:8188");

        const error = await client.getServerlessStatus("gone").catch((err) => err);

        expect(error.message).toBe("Serverless worker not found");
        expect(error.status).toBe(404);
        expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    });

    it("still retries server errors with backoff", async () => {
        globalThis.fetch
            .mockResolvedValueOnce(reply(503))
            .mockResolvedValueOnce(reply(200, { workers: [] }));
        const client = createApiClient("http://127.0.0.1:8188");

        const pending = client.getConfig();
        await vi.runAllTimersAsync();

        await expect(pending).resolves.toEqual({ workers: [] });
        expect(globalThis.fetch).toHaveBeenCalledTimes(2);
    });
});
