import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { getWorkerUrl } from "../workerLifecycle.js";


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
