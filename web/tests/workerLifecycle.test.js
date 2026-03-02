import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { TIMEOUTS } from "../constants.js";
import { getWorkerUrl, shouldShowRemoteLogButton } from "../workerLifecycle.js";


describe("workerLifecycle URL construction", () => {
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

    it("builds local worker URL with explicit local port", () => {
        const worker = { id: "w1", port: 8189, type: "local" };
        expect(getWorkerUrl({}, worker, "/prompt")).toBe("http://127.0.0.1:8189/prompt");
    });

    it("builds remote worker URL with host:port", () => {
        const worker = { id: "w2", host: "worker.example.com", port: 9000, type: "remote" };
        expect(getWorkerUrl({}, worker, "/prompt")).toBe("http://worker.example.com:9000/prompt");
    });

    it("builds cloud worker URL as https", () => {
        const worker = { id: "w3", host: "cloud.example.com", port: 443, type: "cloud" };
        expect(getWorkerUrl({}, worker, "/prompt")).toBe("https://cloud.example.com/prompt");
    });

    it("rewrites runpod proxy hostname for local worker ports", () => {
        globalThis.window = {
            location: {
                hostname: "podabc.proxy.runpod.net",
                protocol: "https:",
                port: "",
                origin: "https://podabc.proxy.runpod.net",
            },
        };
        const worker = { id: "w4", port: 8189, type: "local" };
        expect(getWorkerUrl({}, worker, "/prompt")).toBe("https://podabc-8189.proxy.runpod.net/prompt");
    });
});

describe("workerLifecycle remote log visibility", () => {
    it("shows remote log when worker is online", () => {
        const worker = { enabled: true };
        const status = { online: true };
        expect(shouldShowRemoteLogButton(worker, status, 100_000)).toBe(true);
    });

    it("shows remote log shortly after last online detection", () => {
        const worker = { enabled: true };
        const now = 100_000;
        const status = {
            online: false,
            lastSeenOnlineAt: now - (TIMEOUTS.REMOTE_LOG_AVAILABILITY_WINDOW - 1),
        };
        expect(shouldShowRemoteLogButton(worker, status, now)).toBe(true);
    });

    it("hides remote log when worker has been unavailable past the freshness window", () => {
        const worker = { enabled: true };
        const now = 100_000;
        const status = {
            online: false,
            lastSeenOnlineAt: now - (TIMEOUTS.REMOTE_LOG_AVAILABILITY_WINDOW + 1),
        };
        expect(shouldShowRemoteLogButton(worker, status, now)).toBe(false);
    });

    it("hides remote log when worker is disabled", () => {
        const worker = { enabled: false };
        const status = { online: true, lastSeenOnlineAt: 99_000 };
        expect(shouldShowRemoteLogButton(worker, status, 100_000)).toBe(false);
    });
});
