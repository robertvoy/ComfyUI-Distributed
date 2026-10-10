import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
    buildWorkerUrl,
    buildWorkerWebSocketUrl,
    getMasterUrl,
    normalizeWorkerUrl,
    parseHostInput,
} from "../urlUtils.js";


// ---------------------------------------------------------------------------
// normalizeWorkerUrl
// ---------------------------------------------------------------------------

describe("normalizeWorkerUrl", () => {
    it("returns empty string for empty input", () => {
        expect(normalizeWorkerUrl("")).toBe("");
    });

    it("returns empty string for null input", () => {
        expect(normalizeWorkerUrl(null)).toBe("");
    });

    it("returns empty string for non-string input", () => {
        expect(normalizeWorkerUrl(42)).toBe("");
    });

    it("preserves https protocol", () => {
        expect(normalizeWorkerUrl("https://example.com")).toBe("https://example.com");
    });

    it("strips trailing slash", () => {
        expect(normalizeWorkerUrl("http://example.com/")).toBe("http://example.com");
    });

    it("prepends http when protocol is missing", () => {
        const result = normalizeWorkerUrl("worker.local:8188");
        expect(result).toMatch(/^http:\/\//);
    });

    it("trims leading and trailing whitespace", () => {
        expect(normalizeWorkerUrl("  http://example.com  ")).toBe("http://example.com");
    });
});


// ---------------------------------------------------------------------------
// parseHostInput
// ---------------------------------------------------------------------------

describe("parseHostInput", () => {
    it("returns empty host and null port for null", () => {
        expect(parseHostInput(null)).toEqual({ host: "", port: null });
    });

    it("returns empty host and null port for empty string", () => {
        expect(parseHostInput("")).toEqual({ host: "", port: null });
    });

    it("parses hostname without port", () => {
        const result = parseHostInput("worker.example.com");
        expect(result.host).toBe("worker.example.com");
        expect(result.port).toBeNull();
    });

    it("parses hostname with port", () => {
        const result = parseHostInput("worker.example.com:9000");
        expect(result.host).toBe("worker.example.com");
        expect(result.port).toBe(9000);
    });

    it("strips http:// protocol prefix", () => {
        const result = parseHostInput("http://worker.example.com:8188");
        expect(result.host).toBe("worker.example.com");
        expect(result.port).toBe(8188);
    });

    it("strips https:// protocol prefix", () => {
        const result = parseHostInput("https://worker.example.com");
        expect(result.host).toBe("worker.example.com");
        expect(result.port).toBeNull();
    });

    it("ignores path after host:port", () => {
        const result = parseHostInput("worker.example.com:8188/some/path");
        expect(result.host).toBe("worker.example.com");
        expect(result.port).toBe(8188);
    });
});


// ---------------------------------------------------------------------------
// buildWorkerWebSocketUrl
// ---------------------------------------------------------------------------

describe("buildWorkerWebSocketUrl", () => {
    it("converts http to ws", () => {
        expect(buildWorkerWebSocketUrl("http://worker.local:8188")).toBe(
            "ws://worker.local:8188/distributed/worker_ws"
        );
    });

    it("converts https to wss", () => {
        expect(buildWorkerWebSocketUrl("https://worker.example.com")).toBe(
            "wss://worker.example.com/distributed/worker_ws"
        );
    });
});


// ---------------------------------------------------------------------------
// buildWorkerUrl (requires window.location stub)
// ---------------------------------------------------------------------------

describe("buildWorkerUrl", () => {
    let originalWindow;

    beforeEach(() => {
        originalWindow = globalThis.window;
        globalThis.window = {
            location: {
                hostname: "127.0.0.1",
                protocol: "http:",
                port: "8188",
                origin: "http://127.0.0.1:8188",
            },
        };
    });

    afterEach(() => {
        globalThis.window = originalWindow;
    });

    it("builds local worker URL using window hostname when no host set", () => {
        const worker = { id: "w1", port: 8189 };
        expect(buildWorkerUrl(worker, "/prompt")).toBe("http://127.0.0.1:8189/prompt");
    });

    it("builds remote worker URL using explicit host", () => {
        const worker = { id: "w2", host: "worker.example.com", port: 9000 };
        expect(buildWorkerUrl(worker, "/prompt")).toBe("http://worker.example.com:9000/prompt");
    });

    it("builds cloud worker URL with https when type=cloud", () => {
        const worker = { id: "w3", host: "cloud.example.com", port: 443, type: "cloud" };
        expect(buildWorkerUrl(worker, "/prompt")).toBe("https://cloud.example.com/prompt");
    });

    it("uses https for port 443 even without type=cloud", () => {
        const worker = { id: "w4", host: "worker.example.com", port: 443 };
        const result = buildWorkerUrl(worker, "");
        expect(result.startsWith("https://")).toBe(true);
    });

    it("rewrites runpod proxy hostname for local port", () => {
        globalThis.window = {
            location: {
                hostname: "podabc.proxy.runpod.net",
                protocol: "https:",
                port: "",
                origin: "https://podabc.proxy.runpod.net",
            },
        };
        const worker = { id: "w5", port: 8189 };
        expect(buildWorkerUrl(worker, "/prompt")).toBe(
            "https://podabc-8189.proxy.runpod.net/prompt"
        );
    });

    it("returns URL without trailing slash when no endpoint given", () => {
        const worker = { id: "w6", host: "worker.example.com", port: 8188 };
        const result = buildWorkerUrl(worker, "");
        expect(result.endsWith("/")).toBe(false);
    });
});


// ---------------------------------------------------------------------------
// getMasterUrl
// ---------------------------------------------------------------------------

describe("getMasterUrl", () => {
    const _loc = (hostname, protocol = "http:", port = "8188") => ({
        hostname,
        protocol,
        port,
        origin: `${protocol}//${hostname}${port ? `:${port}` : ""}`,
    });

    it("returns origin when master host not configured and hostname is non-localhost", () => {
        const loc = _loc("192.168.1.10");
        const result = getMasterUrl({}, loc);
        expect(result).toBe(loc.origin);
    });

    it("returns origin for localhost when master host not configured", () => {
        const loc = _loc("127.0.0.1");
        const result = getMasterUrl({}, loc);
        expect(result).toBe(loc.origin);
    });

    it("uses configured master host as-is when it includes http://", () => {
        const config = { master: { host: "http://master.example.com" } };
        const result = getMasterUrl(config, _loc("127.0.0.1"));
        expect(result).toBe("http://master.example.com");
    });

    it("uses configured master host as-is when it includes https://", () => {
        const config = { master: { host: "https://secure.master.com" } };
        const result = getMasterUrl(config, _loc("127.0.0.1"));
        expect(result).toBe("https://secure.master.com");
    });

    it("defaults to https for domain-name master hosts", () => {
        const config = { master: { host: "master.example.com" } };
        const result = getMasterUrl(config, _loc("127.0.0.1"));
        expect(result).toBe("https://master.example.com");
    });

    it("does not force https for IP-address master hosts", () => {
        const config = { master: { host: "192.168.1.100" } };
        const result = getMasterUrl(config, _loc("127.0.0.1"));
        expect(result.startsWith("https://")).toBe(false);
    });

    it("does not force https for localhost master host", () => {
        const config = { master: { host: "localhost" } };
        const result = getMasterUrl(config, _loc("127.0.0.1"));
        expect(result.startsWith("https://")).toBe(false);
    });

    it("accepts null log parameter without throwing", () => {
        const loc = _loc("127.0.0.1");
        expect(() => getMasterUrl({}, loc, null)).not.toThrow();
    });
});
