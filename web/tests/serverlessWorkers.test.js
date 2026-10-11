import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../scripts/api.js", () => ({ api: { clientId: "test-client" } }), { virtual: true });

import { createApiClient } from "../apiClient.js";
import { createWorkerSettingsForm } from "../ui/settingsForm.js";
import { DistributedUI } from "../ui.js";
import { checkWorkerStatus, getWorkerUrl, launchWorker, stopWorker, viewWorkerLog } from "../workerLifecycle.js";
import { handleClearMemory, handleInterruptWorkers } from "../workerUtils.js";
import { isRemoteWorker, saveWorkerSettings, testServerlessWorker } from "../workerSettings.js";
import { performPreflightCheck } from "../executionUtils.js";

function createDocumentMock() {
    const elements = new Map();
    const makeElement = (tagName) => {
        const element = {
            tagName,
            children: [],
            style: {},
            classList: {
                values: new Set(),
                add(...names) { names.forEach((name) => this.values.add(name)); },
                remove(...names) { names.forEach((name) => this.values.delete(name)); },
                toggle(name, force) {
                    const enabled = force ?? !this.values.has(name);
                    if (enabled) this.values.add(name);
                    else this.values.delete(name);
                    return enabled;
                },
            },
            appendChild(child) { this.children.push(child); child.parentNode = this; return child; },
            dispatchEvent() { this.onchange?.({ target: this }); },
        };
        Object.defineProperty(element, "id", {
            get() { return this._id || ""; },
            set(value) { this._id = value; elements.set(value, this); },
        });
        return element;
    };
    return {
        elements,
        document: {
            createElement: makeElement,
            getElementById: (id) => elements.get(id) || null,
        },
    };
}

function createFormUi() {
    return {
        createFormGroup(label, value, id, type = "text", placeholder = "") {
            const group = document.createElement("div");
            const input = document.createElement("input");
            input.id = id;
            input.value = String(value ?? "");
            input.type = type;
            input.placeholder = placeholder;
            group.appendChild(input);
            return { group, input, label };
        },
        createButton(text, onclick) {
            const button = document.createElement("button");
            button.textContent = text;
            button.onclick = onclick;
            return button;
        },
        createButtonGroup(buttons) {
            const group = document.createElement("div");
            buttons.forEach((button) => group.appendChild(button));
            return group;
        },
    };
}

function makeSaveExtension(worker, updateWorker = vi.fn().mockResolvedValue({})) {
    return {
        config: { workers: [worker] },
        api: { updateWorker },
        state: { updateWorker: vi.fn(), setWorkerExpanded: vi.fn() },
        app: { extensionManager: { toast: { add: vi.fn() } } },
        panelElement: null,
    };
}

describe("Runpod serverless worker panel", () => {
    let originalDocument;
    let originalWindow;
    let elements;

    beforeEach(() => {
        originalDocument = globalThis.document;
        originalWindow = globalThis.window;
        const mock = createDocumentMock();
        elements = mock.elements;
        globalThis.document = mock.document;
        globalThis.window = {
            location: { hostname: "localhost", protocol: "http:", port: "8188", origin: "http://localhost:8188" },
        };
    });

    afterEach(() => {
        globalThis.document = originalDocument;
        globalThis.window = originalWindow;
        vi.restoreAllMocks();
    });

    it("classifies Runpod serverless as remote for worker allocation", () => {
        expect(isRemoteWorker({}, { type: "serverless" })).toBe(true);
    });

    it("initializes a saved serverless form before it is attached to the document", () => {
        // Real document lookups cannot see a form while its card is being built.
        vi.spyOn(document, "getElementById").mockReturnValue(null);
        const worker = {
            id: "saved", name: "Runpod", type: "serverless", endpoint_id: "ep_123",
            jobs_per_run: 2, concurrency: 1, timeout_seconds: 900, api_key_configured: true,
        };
        const form = createWorkerSettingsForm(createFormUi(), {}, worker);

        expect(form.parentNode).toBeUndefined();
        expect(elements.get("worker-type-saved").value).toBe("serverless");
        for (const field of ["host", "port", "cuda", "args"]) {
            expect(elements.get(`${field}-group-saved`).style.display).toBe("none");
        }
        expect(elements.get("runpod-text-saved").style.display).toBe("none");
        for (const field of ["endpoint-id", "api-key", "clear-api-key", "jobs-per-run", "concurrency", "timeout-seconds", "serverless-test"]) {
            expect(elements.get(`${field}-group-saved`).style.display).toBe("flex");
        }
        expect(elements.get("endpoint-id-saved").value).toBe("ep_123");
        expect(elements.get("jobs-per-run-saved").value).toBe("2");
        expect(elements.get("concurrency-saved").value).toBe("1");
        expect(elements.get("timeout-seconds-saved").value).toBe("900");
        expect(elements.get("api-key-saved").value).toBe("");
        expect(elements.get("api-key-saved").placeholder).toMatch(/leave blank/i);
    });

    it("preserves a saved cloud port on first render and applies defaults only on type changes", () => {
        vi.spyOn(document, "getElementById").mockReturnValue(null);
        createWorkerSettingsForm(createFormUi(), {}, {
            id: "cloud", name: "Cloud", type: "cloud", host: "worker.example", port: 8443,
        });
        const type = elements.get("worker-type-cloud");
        expect(type.value).toBe("cloud");
        expect(elements.get("port-cloud").value).toBe("8443");
        expect(elements.get("host-cloud").value).toBe("worker.example");
        expect(elements.get("runpod-text-cloud").style.display).toBe("block");

        type.value = "remote";
        type.dispatchEvent();
        type.value = "cloud";
        type.dispatchEvent();
        expect(elements.get("port-cloud").value).toBe("443");
    });

    it("shows only serverless settings and keeps credentials write-only", () => {
        const worker = {
            id: "w1", name: "Runpod", type: "remote", host: "worker.example", port: 8188,
            cuda_device: 0, extra_args: "--test", api_key_configured: true,
        };
        const form = createWorkerSettingsForm(createFormUi(), {}, worker);
        const type = elements.get("worker-type-w1");
        type.value = "serverless";
        type.dispatchEvent();

        expect(type.children.map((option) => [option.value, option.textContent])).toContainEqual([
            "serverless", "Serverless (Runpod)",
        ]);
        for (const id of ["host-group-w1", "port-group-w1", "cuda-group-w1", "args-group-w1"]) {
            expect(elements.get(id).style.display).toBe("none");
        }
        for (const id of ["endpoint-id-w1", "api-key-w1", "jobs-per-run-w1", "concurrency-w1", "timeout-seconds-w1"]) {
            expect(elements.has(id)).toBe(true);
        }
        for (const id of ["endpoint-id-group-w1", "api-key-group-w1", "clear-api-key-group-w1", "jobs-per-run-group-w1", "concurrency-group-w1", "timeout-seconds-group-w1", "serverless-test-group-w1"]) {
            expect(elements.get(id).style.display).toBe("flex");
        }
        expect(elements.has("serverless-help-w1")).toBe(false);
        expect(elements.get("api-key-w1").type).toBe("password");
        expect(elements.get("api-key-w1").value).toBe("");
        expect(elements.get("api-key-w1").placeholder).toMatch(/leave blank/i);
        expect(elements.get("jobs-per-run-w1").value).toBe("1");
        expect(elements.get("jobs-per-run-w1").min).toBe("1");
        expect(elements.get("jobs-per-run-w1").max).toBe("16");
        expect(elements.get("concurrency-w1").value).toBe("1");
        expect(elements.get("concurrency-w1").min).toBe("1");
        expect(elements.get("concurrency-w1").max).toBe("16");
        expect(elements.get("timeout-seconds-w1").value).toBe("1800");
        expect(elements.get("timeout-seconds-w1").min).toBe("30");
        expect(elements.get("timeout-seconds-w1").max).toBe("86400");
        expect(form.children.length).toBeGreaterThan(0);
    });

    it("saves fixed Runpod settings without enabling or retaining the API key", async () => {
        const worker = { id: "w2", name: "Old", type: "remote", enabled: false, api_key_configured: false };
        const extension = makeSaveExtension(worker);
        elements.set("name-w2", { value: "Runpod worker" });
        elements.set("worker-type-w2", { value: "serverless" });
        elements.set("endpoint-id-w2", { value: "ep_123" });
        elements.set("api-key-w2", { value: "secret-token" });
        elements.set("clear-api-key-w2", { checked: false });
        elements.set("jobs-per-run-w2", { value: "4" });
        elements.set("concurrency-w2", { value: "3" });
        elements.set("timeout-seconds-w2", { value: "2400" });

        await saveWorkerSettings(extension, "w2");

        expect(extension.api.updateWorker).toHaveBeenCalledWith("w2", {
            name: "Runpod worker", type: "serverless", provider: "runpod", endpoint_id: "ep_123",
            api_key: "secret-token", jobs_per_run: 4, concurrency: 3, timeout_seconds: 2400, enabled: false,
        });
        expect(worker).toMatchObject({ type: "serverless", provider: "runpod", enabled: false, api_key_configured: true });
        expect(worker).not.toHaveProperty("api_key");
        expect(JSON.stringify(extension.config)).not.toContain("secret-token");
        expect(elements.get("api-key-w2").value).toBe("");
        expect(extension.state.updateWorker).toHaveBeenCalledWith("w2", { enabled: false });
        expect(JSON.stringify(extension.state.updateWorker.mock.calls)).not.toContain("secret-token");
    });

    it("omits a blank API key so the saved key is preserved", async () => {
        const worker = { id: "w-blank", name: "Runpod", type: "serverless", provider: "runpod", enabled: true, api_key_configured: true };
        const extension = makeSaveExtension(worker);
        elements.set("name-w-blank", { value: "Runpod" });
        elements.set("worker-type-w-blank", { value: "serverless" });
        elements.set("endpoint-id-w-blank", { value: "ep_123" });
        elements.set("api-key-w-blank", { value: "" });
        elements.set("clear-api-key-w-blank", { checked: false });
        elements.set("jobs-per-run-w-blank", { value: "1" });
        elements.set("concurrency-w-blank", { value: "1" });
        elements.set("timeout-seconds-w-blank", { value: "1800" });

        await saveWorkerSettings(extension, "w-blank");

        const payload = extension.api.updateWorker.mock.calls[0][1];
        expect(payload).not.toHaveProperty("api_key");
        expect(payload).not.toHaveProperty("clear_api_key");
        expect(worker.api_key_configured).toBe(true);
        expect(worker).not.toHaveProperty("api_key");
    });

    it("omits a blank key to preserve it and sends an explicit clear flag when requested", async () => {
        const worker = { id: "w3", name: "Runpod", type: "serverless", provider: "runpod", enabled: true, api_key_configured: true };
        const extension = makeSaveExtension(worker);
        elements.set("name-w3", { value: "Runpod" });
        elements.set("worker-type-w3", { value: "serverless" });
        elements.set("endpoint-id-w3", { value: "ep_123" });
        elements.set("api-key-w3", { value: "" });
        elements.set("clear-api-key-w3", { checked: true });
        elements.set("jobs-per-run-w3", { value: "1" });
        elements.set("concurrency-w3", { value: "1" });
        elements.set("timeout-seconds-w3", { value: "1800" });

        await saveWorkerSettings(extension, "w3");

        expect(extension.api.updateWorker).toHaveBeenCalledWith("w3", {
            name: "Runpod", type: "serverless", provider: "runpod", endpoint_id: "ep_123",
            clear_api_key: true, jobs_per_run: 1, concurrency: 1, timeout_seconds: 1800, enabled: true,
        });
        expect(worker.api_key_configured).toBe(false);
        expect(worker).not.toHaveProperty("api_key");
    });

    it("uses provider status instead of a direct ComfyUI URL probe", async () => {
        const statusElement = { textContent: "" };
        elements.set("serverless-status-w4", statusElement);
        const status = {
            ok: true, queueRemaining: 0, message: "Ready to scale",
            jobs: [{ job_id: "job-1", status: "RUNNING", worker_id: "w4", execution_id: "exec-1" }],
        };
        const extension = {
            config: { workers: [{ id: "w4", name: "Runpod", type: "serverless", enabled: true }] },
            api: { getServerlessStatus: vi.fn().mockResolvedValue(status), probeWorker: vi.fn() },
            statusCheckAbortController: null,
            state: {
                setWorkerStatus: vi.fn(),
                updateWorker: vi.fn(),
                getWorkerStatus: vi.fn().mockReturnValue({ online: true, processing: false, queueCount: 0 }),
                getWorker: vi.fn().mockReturnValue({}),
                isWorkerLaunching: vi.fn().mockReturnValue(false),
            },
            updateWorkerCard: vi.fn().mockReturnValue(true),
            log: vi.fn(),
        };

        await checkWorkerStatus(extension, extension.config.workers[0]);

        expect(extension.api.getServerlessStatus).toHaveBeenCalledWith("w4");
        expect(extension.api.probeWorker).not.toHaveBeenCalled();
        expect(extension.state.setWorkerStatus).toHaveBeenCalledWith("w4", {
            online: true, processing: true, queueCount: 0,
        });
        expect(statusElement.textContent).toContain("Ready to scale");
        expect(statusElement.textContent).toContain("RUNNING");
    });

    it("considers a healthy zero-worker Runpod pool ready and never builds a worker URL", async () => {
        const worker = { id: "w-ready", name: "Runpod", type: "serverless", enabled: true };
        const extension = {
            api: {
                getServerlessStatus: vi.fn().mockResolvedValue({ ok: true, queueRemaining: 0, message: "Ready to scale" }),
                probeWorker: vi.fn(),
            },
            log: vi.fn(),
        };

        await expect(performPreflightCheck(extension, [worker])).resolves.toEqual([worker]);
        expect(extension.api.getServerlessStatus).toHaveBeenCalledWith("w-ready");
        expect(extension.api.probeWorker).not.toHaveBeenCalled();
        expect(getWorkerUrl(extension, worker)).toBeNull();
    });

    it("does not render or invoke direct ComfyUI launch, stop, or log controls", async () => {
        const ui = new DistributedUI();
        const worker = { id: "w-controls", name: "Runpod", type: "serverless", enabled: true };
        const controls = ui.createControlsSection({ dynamic: true }, worker, {}, true);
        const collect = (element) => [element, ...(element.children || []).flatMap(collect)];
        const elementsInControls = collect(controls);
        const extension = {
            config: { workers: [worker] },
            api: {
                launchWorker: vi.fn(), stopWorker: vi.fn(), getRemoteWorkerLog: vi.fn(),
            },
        };

        expect(elementsInControls.filter((element) => element.tagName === "button")).toHaveLength(0);
        await launchWorker(extension, worker.id);
        await stopWorker(extension, worker.id);
        await viewWorkerLog(extension, worker.id, true);
        expect(extension.api.launchWorker).not.toHaveBeenCalled();
        expect(extension.api.stopWorker).not.toHaveBeenCalled();
        expect(extension.api.getRemoteWorkerLog).not.toHaveBeenCalled();
    });

    it("uses the serverless status, test, and cancel routes with the frozen request shapes", async () => {
        const originalFetch = globalThis.fetch;
        const statusPayload = {
            ok: true,
            queueRemaining: 0,
            message: "Ready to scale",
            jobs: [{ job_id: "job-1", status: "RUNNING", worker_id: "w5", execution_id: "exec-1" }],
        };
        globalThis.fetch = vi.fn()
            .mockResolvedValueOnce({ ok: true, status: 200, json: vi.fn().mockResolvedValue({ ok: true, message: "Endpoint test passed" }) })
            .mockResolvedValueOnce({ ok: true, status: 200, json: vi.fn().mockResolvedValue(statusPayload) })
            .mockResolvedValueOnce({ ok: true, status: 200, json: vi.fn().mockResolvedValue({ ok: true }) });
        try {
            const client = createApiClient("http://localhost:8188");
            await client.testServerlessWorker("w5");
            await expect(client.getServerlessStatus("w5")).resolves.toEqual(statusPayload);
            await client.cancelServerlessWorker("w5");

            expect(globalThis.fetch).toHaveBeenNthCalledWith(1, "http://localhost:8188/distributed/serverless/test", expect.objectContaining({
                method: "POST", body: JSON.stringify({ worker_id: "w5" }),
            }));
            expect(globalThis.fetch).toHaveBeenNthCalledWith(2, "http://localhost:8188/distributed/serverless/status/w5", expect.any(Object));
            expect(globalThis.fetch).toHaveBeenNthCalledWith(3, "http://localhost:8188/distributed/serverless/cancel", expect.objectContaining({
                method: "POST", body: JSON.stringify({ worker_id: "w5" }),
            }));
        } finally {
            globalThis.fetch = originalFetch;
        }
    });

    it("tests only the saved Runpod endpoint and reports the backend message", async () => {
        const button = document.createElement("button");
        button.id = "test-serverless-w6";
        button.textContent = "Test Saved Endpoint";
        const extension = {
            api: {
                testServerlessWorker: vi.fn().mockResolvedValue({ ok: true, message: "Endpoint reachable" }),
                queueDistributed: vi.fn(),
            },
            app: { extensionManager: { toast: { add: vi.fn() } } },
        };

        await expect(testServerlessWorker(extension, "w6")).resolves.toEqual({ ok: true, message: "Endpoint reachable" });

        expect(extension.api.testServerlessWorker).toHaveBeenCalledWith("w6");
        expect(extension.api.queueDistributed).not.toHaveBeenCalled();
        expect(extension.app.extensionManager.toast.add).toHaveBeenCalledWith(expect.objectContaining({
            severity: "success", detail: "Endpoint reachable",
        }));
        expect(button.textContent).toBe("Test Saved Endpoint");
        expect(button.disabled).toBe(false);
    });

    it("redacts API-key-bearing worker update errors from logs and user-visible exceptions", async () => {
        const originalFetch = globalThis.fetch;
        const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
        const secret = "never-log-this-runpod-key";
        globalThis.fetch = vi.fn().mockResolvedValue({
            ok: false,
            status: 400,
            json: vi.fn().mockResolvedValue({ message: `Rejected key ${secret}` }),
        });
        try {
            const client = createApiClient("http://localhost:8188");
            await expect(client.updateWorker("w-secret", { api_key: secret })).rejects.toThrow("[redacted]");
            expect(logSpy.mock.calls.flat().join(" ")).not.toContain(secret);
        } finally {
            globalThis.fetch = originalFetch;
        }
    });

    it("never sends VRAM operations to serverless workers", async () => {
        const originalFetch = globalThis.fetch;
        globalThis.fetch = vi.fn();
        vi.useFakeTimers();
        const button = {
            textContent: "Clear Worker VRAM", style: { cssText: "" }, disabled: false,
            classList: { names: new Set(), add(...names) { names.forEach((name) => this.names.add(name)); }, remove(...names) { names.forEach((name) => this.names.delete(name)); } },
        };
        const extension = { enabledWorkers: [{ id: "s-vram", name: "Runpod", type: "serverless", enabled: true }] };
        try {
            await handleClearMemory(extension, button);
            expect(globalThis.fetch).not.toHaveBeenCalled();
        } finally {
            vi.runOnlyPendingTimers();
            vi.useRealTimers();
            globalThis.fetch = originalFetch;
        }
    });

    it("cancels each enabled serverless worker once without changing normal worker interrupts", async () => {
        const originalFetch = globalThis.fetch;
        globalThis.fetch = vi.fn().mockResolvedValue({ ok: true });
        vi.useFakeTimers();
        const button = {
            textContent: "Interrupt Workers", style: { cssText: "" }, disabled: false,
            classList: { names: new Set(), add(...names) { names.forEach((name) => this.names.add(name)); }, remove(...names) { names.forEach((name) => this.names.delete(name)); } },
        };
        const workers = [
            { id: "s1", name: "Serverless 1", type: "serverless", enabled: true },
            { id: "s2", name: "Serverless 2", type: "serverless", enabled: true },
            { id: "s3", name: "Disabled", type: "serverless", enabled: false },
            { id: "local", name: "Local", type: "local", enabled: true, port: 8189 },
        ];
        const extension = {
            enabledWorkers: workers.filter((worker) => worker.enabled),
            api: { cancelServerlessWorker: vi.fn().mockResolvedValue({ ok: true }) },
            log: vi.fn(),
        };
        try {
            await handleInterruptWorkers(extension, button);

            expect(extension.api.cancelServerlessWorker).toHaveBeenCalledTimes(2);
            expect(extension.api.cancelServerlessWorker).toHaveBeenNthCalledWith(1, "s1");
            expect(extension.api.cancelServerlessWorker).toHaveBeenNthCalledWith(2, "s2");
            expect(globalThis.fetch).toHaveBeenCalledTimes(1);
            expect(globalThis.fetch.mock.calls[0][0]).toBe("http://localhost:8189/interrupt");
        } finally {
            vi.runOnlyPendingTimers();
            vi.useRealTimers();
            globalThis.fetch = originalFetch;
        }
    });
});
