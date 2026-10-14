import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../scripts/api.js", () => ({ api: { clientId: "test-client" } }), { virtual: true });

import { createApiClient } from "../apiClient.js";
import { createWorkerSettingsForm } from "../ui/settingsForm.js";
import { DistributedUI } from "../ui.js";
import { checkWorkerStatus, formatServerlessStatus, getWorkerUrl, launchWorker, stopWorker, viewWorkerLog } from "../workerLifecycle.js";
import { handleClearMemory, handleInterruptWorkers } from "../workerUtils.js";
import { cancelWorkerSettings, isRemoteWorker, normalizeRunpodEndpointId, saveWorkerSettings, testServerlessWorker } from "../workerSettings.js";
import { performPreflightCheck } from "../executionUtils.js";

function createDocumentMock() {
    const elements = new Map();
    const makeElement = (tagName) => {
        const element = {
            tagName,
            children: [],
            dataset: {},
            style: {},
            setAttribute(name, value) { this[name] = value; },
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
            querySelector: () => null,
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
        for (const field of ["endpoint-id", "media-profile", "api-key", "clear-api-key", "jobs-per-run", "timeout-seconds", "serverless-test"]) {
            expect(elements.get(`${field}-group-saved`).style.display).toBe("flex");
        }
        // The separate concurrency setting was removed; a legacy value is ignored.
        expect(elements.has("concurrency-saved")).toBe(false);
        expect(elements.has("concurrency-group-saved")).toBe(false);
        expect(elements.get("endpoint-id-saved").value).toBe("ep_123");
        expect(elements.get("media-profile-saved").value).toBe("legacy-image-v1");
        expect(elements.get("jobs-per-run-saved").value).toBe("2");
        expect(elements.get("timeout-seconds-saved").value).toBe("900");
        expect(elements.get("api-key-saved").value).toBe("");
        expect(elements.get("api-key-saved").placeholder).toMatch(/leave blank/i);
    });

    it("renders explicit endpoint profiles with installation and no-auto-detection guidance", () => {
        createWorkerSettingsForm(createFormUi(), {}, {
            id: "profile", name: "Runpod", type: "serverless", media_profile: "comfy-media-v1",
        });
        const select = elements.get("media-profile-profile");
        const group = elements.get("media-profile-group-profile");
        expect(group.children[0].textContent).toBe("Endpoint media profile");
        expect(group.children[0].htmlFor).toBe(select.id);
        expect(select.children.map((option) => [option.value, option.textContent])).toEqual([
            ["legacy-image-v1", "Legacy images (existing handler)"],
            ["comfy-media-v1", "Typed media v1 (requires endpoint bridge)"],
        ]);
        expect(select.value).toBe("comfy-media-v1");
        expect(select.title).toMatch(/after installing.*endpoint bridge/i);
        expect(select.title).toMatch(/health checks do not auto-detect/i);
        expect(select.title).toMatch(/schema.*formats and limits are fixed/i);
        const type = elements.get("worker-type-profile");
        type.value = "remote";
        type.dispatchEvent();
        expect(group.style.display).toBe("none");
        type.value = "serverless";
        type.dispatchEvent();
        expect(group.style.display).toBe("flex");
        expect(select.value).toBe("comfy-media-v1");
    });

    it.each(["legacy-image-v1", "comfy-media-v1"])("round-trips an explicit %s selection through the save payload and reload", async (profile) => {
        const worker = { id: "roundtrip", name: "Runpod", type: "serverless", endpoint_id: "ep_123", enabled: false };
        let persisted;
        const extension = makeSaveExtension(worker, vi.fn(async (id, payload) => {
            persisted = JSON.parse(JSON.stringify({ id, ...payload }));
            return {};
        }));
        const ui = createFormUi();
        const form = createWorkerSettingsForm(ui, extension, worker);
        elements.get("media-profile-roundtrip").value = profile;
        const saveButton = form.children.at(-1).children.find((button) => button.textContent === "Save");
        await saveButton.onclick();
        expect(persisted.media_profile).toBe(profile);
        expect(worker.media_profile).toBe(profile);
        expect(persisted).not.toHaveProperty("api_key");
        expect(persisted).not.toHaveProperty("concurrency");
        createWorkerSettingsForm(ui, extension, persisted);
        expect(elements.get("media-profile-roundtrip").value).toBe(profile);
    });

    it.each([undefined, "comfy-media-v1"])("restores the saved profile on cancel (%s)", (profile) => {
        const worker = { id: "cancel", name: "Runpod", type: "serverless", media_profile: profile };
        const extension = makeSaveExtension(worker);
        createWorkerSettingsForm(createFormUi(), extension, worker);
        elements.get("media-profile-cancel").value = profile ? "legacy-image-v1" : "comfy-media-v1";
        cancelWorkerSettings(extension, worker.id);
        expect(elements.get("media-profile-cancel").value).toBe(profile ?? "legacy-image-v1");
        expect(extension.api.updateWorker).not.toHaveBeenCalled();
    });

    it("keeps the saved profile after a busy-guard error and rejects invalid selections", async () => {
        const worker = { id: "busy", name: "Runpod", type: "serverless", endpoint_id: "ep_123", media_profile: "comfy-media-v1" };
        const extension = makeSaveExtension(worker, vi.fn().mockRejectedValue(new Error("Finish or reconcile unresolved serverless jobs")));
        createWorkerSettingsForm(createFormUi(), extension, worker);
        elements.get("media-profile-busy").value = "legacy-image-v1";
        await saveWorkerSettings(extension, worker.id);
        expect(worker.media_profile).toBe("comfy-media-v1");
        expect(extension.app.extensionManager.toast.add).toHaveBeenLastCalledWith(
            expect.objectContaining({ summary: "Save Failed", detail: expect.stringMatching(/unresolved/) }),
        );
        elements.get("media-profile-busy").value = "auto";
        await saveWorkerSettings(extension, worker.id);
        expect(extension.api.updateWorker).toHaveBeenCalledTimes(1);
        expect(extension.app.extensionManager.toast.add).toHaveBeenLastCalledWith(
            expect.objectContaining({ summary: "Validation Error", detail: expect.stringMatching(/media profile/) }),
        );
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
        for (const id of ["endpoint-id-w1", "api-key-w1", "jobs-per-run-w1", "timeout-seconds-w1"]) {
            expect(elements.has(id)).toBe(true);
        }
        expect(elements.has("concurrency-w1")).toBe(false);
        for (const id of ["endpoint-id-group-w1", "api-key-group-w1", "clear-api-key-group-w1", "jobs-per-run-group-w1", "timeout-seconds-group-w1", "serverless-test-group-w1"]) {
            expect(elements.get(id).style.display).toBe("flex");
        }
        expect(elements.has("serverless-help-w1")).toBe(false);
        expect(elements.get("api-key-w1").type).toBe("password");
        expect(elements.get("api-key-w1").value).toBe("");
        expect(elements.get("api-key-w1").placeholder).toMatch(/leave blank/i);
        expect(elements.get("jobs-per-run-w1").value).toBe("1");
        expect(elements.get("jobs-per-run-w1").min).toBe("1");
        expect(elements.get("jobs-per-run-w1").max).toBe("16");
        expect(elements.get("jobs-per-run-w1").title).toBe(
            "Separate Runpod jobs submitted each run, all at once. Runpod's max workers decides how many GPUs actually run them.",
        );
        expect(elements.get("timeout-seconds-w1").value).toBe("1800");
        expect(elements.get("timeout-seconds-w1").min).toBe("30");
        expect(elements.get("timeout-seconds-w1").max).toBe("86400");
        expect(form.children.length).toBeGreaterThan(0);
    });

    it("disables Clear saved API key while the worker is enabled and explains why", () => {
        vi.spyOn(document, "getElementById").mockReturnValue(null);
        const base = { name: "Runpod", type: "serverless", endpoint_id: "ep_123" };
        createWorkerSettingsForm(createFormUi(), {}, { ...base, id: "on", enabled: true, api_key_configured: true });
        createWorkerSettingsForm(createFormUi(), {}, { ...base, id: "off", enabled: false, api_key_configured: true });
        createWorkerSettingsForm(createFormUi(), {}, { ...base, id: "nokey", enabled: false, api_key_configured: false });

        expect(elements.get("clear-api-key-on").disabled).toBe(true);
        expect(elements.get("clear-api-key-group-on").title).toMatch(/turn this worker off/i);
        expect(elements.get("clear-api-key-off").disabled).toBe(false);
        expect(elements.get("clear-api-key-nokey").disabled).toBe(true);
        expect(elements.get("clear-api-key-group-nokey").title).toMatch(/no api key is saved/i);
        for (const id of ["on", "off", "nokey"]) {
            expect(elements.get(`clear-api-key-group-${id}`).title).not.toMatch(/pool/i);
        }
    });

    it("saves fixed Runpod settings without enabling or retaining the API key", async () => {
        const worker = { id: "w2", name: "Old", type: "remote", enabled: false, api_key_configured: false };
        const extension = makeSaveExtension(worker);
        elements.set("name-w2", { value: "Runpod worker" });
        elements.set("worker-type-w2", { value: "serverless" });
        elements.set("endpoint-id-w2", { value: "ep_123" });
        elements.set("media-profile-w2", { value: "legacy-image-v1" });
        elements.set("api-key-w2", { value: "secret-token" });
        elements.set("clear-api-key-w2", { checked: false });
        elements.set("jobs-per-run-w2", { value: "4" });
        elements.set("timeout-seconds-w2", { value: "2400" });

        await saveWorkerSettings(extension, "w2");

        expect(extension.api.updateWorker).toHaveBeenCalledWith("w2", {
            name: "Runpod worker", type: "serverless", provider: "runpod", endpoint_id: "ep_123",
            media_profile: "legacy-image-v1",
            api_key: "secret-token", jobs_per_run: 4, timeout_seconds: 2400, enabled: false,
        });
        expect(worker).toMatchObject({ type: "serverless", provider: "runpod", enabled: false, api_key_configured: true });
        expect(worker).not.toHaveProperty("api_key");
        expect(JSON.stringify(extension.config)).not.toContain("secret-token");
        expect(elements.get("api-key-w2").value).toBe("");
        expect(extension.state.updateWorker).toHaveBeenCalledWith("w2", { enabled: false });
        expect(JSON.stringify(extension.state.updateWorker.mock.calls)).not.toContain("secret-token");
    });

    it("labels the single parallelism setting Images per run and drops a legacy concurrency value", async () => {
        const worker = { id: "legacy", name: "Runpod", type: "serverless", provider: "runpod", enabled: false,
            api_key_configured: true, jobs_per_run: 1, concurrency: 1 };
        const extension = makeSaveExtension(worker);
        elements.set("name-legacy", { value: "Runpod" });
        elements.set("worker-type-legacy", { value: "serverless" });
        elements.set("endpoint-id-legacy", { value: "ep_123" });
        elements.set("media-profile-legacy", { value: "legacy-image-v1" });
        elements.set("api-key-legacy", { value: "" });
        elements.set("clear-api-key-legacy", { checked: false });
        elements.set("jobs-per-run-legacy", { value: "2" });
        elements.set("timeout-seconds-legacy", { value: "1800" });

        await saveWorkerSettings(extension, "legacy");

        const payload = extension.api.updateWorker.mock.calls[0][1];
        expect(payload.jobs_per_run).toBe(2);
        expect(payload).not.toHaveProperty("concurrency");
        expect(worker.jobs_per_run).toBe(2);
        expect(worker).not.toHaveProperty("concurrency");

        elements.set("jobs-per-run-legacy", { value: "17" });
        await saveWorkerSettings(extension, "legacy");
        expect(extension.api.updateWorker).toHaveBeenCalledTimes(1);
        expect(extension.app.extensionManager.toast.add).toHaveBeenLastCalledWith(
            expect.objectContaining({ detail: "Images per run must be a whole number from 1 to 16" }),
        );
    });

    it("trims a pasted Runpod URL to the endpoint ID and explains other invalid IDs", async () => {
        expect(normalizeRunpodEndpointId("  abc123xyz ")).toBe("abc123xyz");
        expect(normalizeRunpodEndpointId("https://api.runpod.ai/v2/abc123xyz/run")).toBe("abc123xyz");
        expect(normalizeRunpodEndpointId("api.runpod.ai/v2/abc_12-3")).toBe("abc_12-3");
        expect(normalizeRunpodEndpointId("https://www.runpod.io/console/serverless/user/endpoint/abc")).toBe(
            "https://www.runpod.io/console/serverless/user/endpoint/abc",
        );

        const worker = { id: "url", name: "Runpod", type: "serverless", provider: "runpod", enabled: false, api_key_configured: true };
        const extension = makeSaveExtension(worker);
        elements.set("name-url", { value: "Runpod" });
        elements.set("worker-type-url", { value: "serverless" });
        elements.set("endpoint-id-url", { value: "https://api.runpod.ai/v2/abc123xyz/runsync" });
        elements.set("media-profile-url", { value: "legacy-image-v1" });
        elements.set("api-key-url", { value: "" });
        elements.set("clear-api-key-url", { checked: false });
        elements.set("jobs-per-run-url", { value: "1" });
        elements.set("timeout-seconds-url", { value: "1800" });
        await saveWorkerSettings(extension, "url");
        expect(extension.api.updateWorker.mock.calls[0][1].endpoint_id).toBe("abc123xyz");

        elements.set("endpoint-id-url", { value: "https://www.runpod.io/console/serverless/endpoint/abc" });
        await saveWorkerSettings(extension, "url");
        expect(extension.api.updateWorker).toHaveBeenCalledTimes(1);
        expect(extension.app.extensionManager.toast.add).toHaveBeenLastCalledWith(
            expect.objectContaining({ detail: expect.stringMatching(/just the Runpod endpoint ID/) }),
        );
    });

    it("asks before a type change deletes the saved API key, and cancels cleanly", async () => {
        const worker = { id: "conv", name: "Runpod", type: "serverless", provider: "runpod", enabled: false, api_key_configured: true, media_profile: "comfy-media-v1" };
        const extension = makeSaveExtension(worker);
        elements.set("name-conv", { value: "Runpod" });
        elements.set("worker-type-conv", { value: "remote" });
        elements.set("host-conv", { value: "192.168.1.20" });
        elements.set("port-conv", { value: "8188" });
        const confirmSpy = vi.fn().mockReturnValue(false);
        globalThis.confirm = confirmSpy;
        try {
            await saveWorkerSettings(extension, "conv");
            expect(confirmSpy).toHaveBeenCalledWith(expect.stringMatching(/saved Runpod API key will be deleted/));
            expect(extension.api.updateWorker).not.toHaveBeenCalled();
            expect(worker.type).toBe("serverless");
            expect(worker.media_profile).toBe("comfy-media-v1");

            confirmSpy.mockReturnValue(true);
            await saveWorkerSettings(extension, "conv");
            expect(extension.api.updateWorker).toHaveBeenCalledTimes(1);
            expect(extension.api.updateWorker.mock.calls[0][1].type).toBe("remote");
            expect(worker).not.toHaveProperty("media_profile");
        } finally {
            delete globalThis.confirm;
        }
    });

    it("does not ask for confirmation when a serverless worker without a key changes type", async () => {
        const worker = { id: "nokey2", name: "Runpod", type: "serverless", provider: "runpod", enabled: false, api_key_configured: false };
        const extension = makeSaveExtension(worker);
        elements.set("name-nokey2", { value: "Runpod" });
        elements.set("worker-type-nokey2", { value: "remote" });
        elements.set("host-nokey2", { value: "192.168.1.20" });
        elements.set("port-nokey2", { value: "8188" });
        const confirmSpy = vi.fn().mockReturnValue(false);
        globalThis.confirm = confirmSpy;
        try {
            await saveWorkerSettings(extension, "nokey2");
            expect(confirmSpy).not.toHaveBeenCalled();
            expect(extension.api.updateWorker).toHaveBeenCalledTimes(1);
        } finally {
            delete globalThis.confirm;
        }
    });

    it("omits a blank API key so the saved key is preserved", async () => {
        const worker = { id: "w-blank", name: "Runpod", type: "serverless", provider: "runpod", enabled: true, api_key_configured: true };
        const extension = makeSaveExtension(worker);
        elements.set("name-w-blank", { value: "Runpod" });
        elements.set("worker-type-w-blank", { value: "serverless" });
        elements.set("endpoint-id-w-blank", { value: "ep_123" });
        elements.set("media-profile-w-blank", { value: "legacy-image-v1" });
        elements.set("api-key-w-blank", { value: "" });
        elements.set("clear-api-key-w-blank", { checked: false });
        elements.set("jobs-per-run-w-blank", { value: "1" });
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
        elements.set("media-profile-w3", { value: "legacy-image-v1" });
        elements.set("api-key-w3", { value: "" });
        elements.set("clear-api-key-w3", { checked: true });
        elements.set("jobs-per-run-w3", { value: "1" });
        elements.set("timeout-seconds-w3", { value: "1800" });

        await saveWorkerSettings(extension, "w3");

        expect(extension.api.updateWorker).toHaveBeenCalledWith("w3", {
            name: "Runpod", type: "serverless", provider: "runpod", endpoint_id: "ep_123",
            media_profile: "legacy-image-v1",
            clear_api_key: true, jobs_per_run: 1, timeout_seconds: 1800, enabled: true,
        });
        expect(worker.api_key_configured).toBe(false);
        expect(worker).not.toHaveProperty("api_key");
    });

    it.each([
        [undefined, "Checking status…"],
        [{ ok: true, queueRemaining: 0, jobs: [] }, "Ready"],
        [{ ok: true, queueRemaining: 1, jobs: [{ status: "IN_QUEUE" }] }, "Queued"],
        [{ ok: true, jobs: [{ status: "IN_PROGRESS" }] }, "Running"],
        [{ ok: true, jobs: [{ status: "RUNNING" }, { status: "IN_QUEUE" }] }, "Running"],
        [{ ok: true, jobs: [{ status: "RUNNING" }, { status: "IN_PROGRESS" }] }, "Running · 2 jobs"],
        [{ ok: true, jobs: [{ status: "DOWNLOADING" }] }, "Downloading result"],
        [{ ok: true, jobs: [{ status: "CANCEL_REQUESTED" }] }, "Cancelling"],
        [{ ok: true, queueRemaining: 1 }, "Active · 1 job"],
        [{ ok: true, queueRemaining: 3 }, "Active · 3 jobs"],
        [{ ok: true, jobs: [{ status: "COMPLETED" }] }, "Ready"],
        [{ ok: true, jobs: [{ status: "FAILED" }] }, "Last job failed"],
        [{ ok: false, message: "Check the saved endpoint ID and API key" }, "Unavailable"],
    ])("formats provider state as one plain-language summary: %j", (status, expected) => {
        expect(formatServerlessStatus(status)).toBe(expected);
    });

    it("keeps status above a consistent endpoint info box and retains expanded settings", () => {
        const ui = new DistributedUI();
        const worker = {
            id: "compact", name: "Used Car Inventory", type: "serverless", enabled: true,
            endpoint_id: "ep_private_id", jobs_per_run: 1, concurrency: 1,
        };
        const extension = {
            config: { workers: [worker] },
            state: {
                isWorkerExpanded: () => false,
                getWorker: () => ({ serverlessStatus: { ok: true, queueRemaining: 1, jobs: [{ status: "IN_QUEUE" }] } }),
            },
        };
        const card = ui.renderEntityCard("worker", worker, extension);
        const collect = (element) => [element, ...(element.children || []).flatMap(collect)];
        const info = collect(card).find((element) => element.classList.values.has("dist-worker-info"));
        expect(info.innerHTML).not.toContain("Runpod Serverless");
        expect(info.innerHTML).not.toMatch(/ep_private_id|jobs\/run|slots/);
        expect(elements.get("serverless-status-compact").textContent).toBe("Queued");
        expect(elements.get("endpoint-id-compact").value).toBe("ep_private_id");
        expect(Number(elements.get("jobs-per-run-compact").value)).toBe(1);
        expect(elements.has("concurrency-compact")).toBe(false);
        const box = elements.get("controls-compact").children[0].children[0];
        expect(box.textContent).toBe("Runpod Serverless endpoint");
        expect(box.classList.values.has("dist-info-box")).toBe(true);
        expect(box.style.cssText).toBe(ui.styles.infoBox);
        expect(elements.get("status-compact").style.cssText).toContain("flex: 0 0 10px");
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
        expect(statusElement.textContent).toBe("Running");
        expect(statusElement.title).toBe("Ready to scale");
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

        const extension = {
            config: { workers: [worker] },
            api: {
                launchWorker: vi.fn(), stopWorker: vi.fn(), getRemoteWorkerLog: vi.fn(),
            },
        };

        const box = controls.children[0].children[0];
        expect(box.textContent).toBe("Runpod Serverless endpoint");
        expect(box.tagName).toBe("div");
        expect(controls.children[0].children).toHaveLength(1);
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
        const extension = {
            api: {
                testServerlessWorker: vi.fn().mockResolvedValue({ ok: true, message: "Endpoint reachable" }),
                queueDistributed: vi.fn(),
            },
            app: { extensionManager: { toast: { add: vi.fn() } } },
        };
        const worker = { id: "w6", name: "Runpod", type: "serverless" };
        extension.config = { workers: [worker] };
        createWorkerSettingsForm(createFormUi(), extension, worker);
        const button = elements.get("test-serverless-w6");

        await expect(testServerlessWorker(extension, "w6")).resolves.toEqual({ ok: true, message: "Endpoint reachable" });

        expect(worker).not.toHaveProperty("media_profile");
        expect(elements.get("media-profile-w6").value).toBe("legacy-image-v1");
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
