import { BUTTON_STYLES } from '../constants.js';
import { cancelWorkerSettings, deleteWorker, isRemoteWorker, saveWorkerSettings, testServerlessWorker } from '../workerSettings.js';

export function createWorkerSettingsForm(ui, extension, worker) {
    const form = document.createElement("div");
    form.style.cssText = "display: flex; flex-direction: column; gap: 8px;";

    const nameGroup = ui.createFormGroup("Name:", worker.name, `name-${worker.id}`);
    form.appendChild(nameGroup.group);

    const typeGroup = document.createElement("div");
    typeGroup.style.cssText = "display: flex; flex-direction: column; gap: 4px; margin: 5px 0;";

    const typeLabel = document.createElement("label");
    typeLabel.htmlFor = `worker-type-${worker.id}`;
    typeLabel.textContent = "Worker Type:";
    typeLabel.style.cssText = "font-size: 12px; color: var(--dist-label-text, #ccc);";

    const typeSelect = document.createElement("select");
    typeSelect.id = `worker-type-${worker.id}`;
    typeSelect.style.cssText =
        "padding: 4px 8px; background: var(--dist-input-bg, #333); color: var(--dist-input-text, #fff); border: 1px solid var(--dist-input-border, #555); border-radius: 4px; font-size: 12px;";

    const localOption = document.createElement("option");
    localOption.value = "local";
    localOption.textContent = "Local";

    const remoteOption = document.createElement("option");
    remoteOption.value = "remote";
    remoteOption.textContent = "Remote";

    const cloudOption = document.createElement("option");
    cloudOption.value = "cloud";
    cloudOption.textContent = "Cloud";

    const serverlessOption = document.createElement("option");
    serverlessOption.value = "serverless";
    serverlessOption.textContent = "Serverless (Runpod)";

    typeSelect.appendChild(localOption);
    typeSelect.appendChild(remoteOption);
    typeSelect.appendChild(cloudOption);
    typeSelect.appendChild(serverlessOption);

    const runpodText = document.createElement("a");
    runpodText.id = `runpod-text-${worker.id}`;
    runpodText.href = "https://github.com/robertvoy/ComfyUI-Distributed/blob/main/docs/worker-setup-guides.md#cloud-workers";
    runpodText.target = "_blank";
    runpodText.textContent = "Deploy Cloud Worker with Runpod";
    runpodText.style.cssText = "font-size: 12px; color: #4a90e2; text-decoration: none; margin-top: 4px; display: none; cursor: pointer;";

    const updateWorkerType = (workerType, applyDefaults = false) => {
        // The form is initialized before its card is attached to document.
        // Use this form's own elements, not document-wide ID lookups.
        const hostInput = hostGroup.input;
        const portInput = portGroup.input;
        const serverlessGroups = [
            endpointGroup.group,
            apiKeyGroup.group,
            clearApiKeyGroup,
            jobsPerRunGroup.group,
            concurrencyGroup.group,
            timeoutGroup.group,
            testServerlessGroup,
        ];

        const hideServerless = () => serverlessGroups.forEach((group) => { group.style.display = "none"; });

        if (workerType === "local") {
            hideServerless();
            hostGroup.group.style.display = "none";
            portGroup.group.style.display = "flex";
            cudaGroup.group.style.display = "flex";
            argsGroup.group.style.display = "flex";
            runpodText.style.display = "none";
        } else if (workerType === "remote") {
            hideServerless();
            hostGroup.group.style.display = "flex";
            portGroup.group.style.display = "flex";
            cudaGroup.group.style.display = "none";
            argsGroup.group.style.display = "none";
            runpodText.style.display = "none";
            hostInput.placeholder = "e.g., 192.168.1.100";
            if (applyDefaults && (hostInput.value === "localhost" || hostInput.value === "127.0.0.1")) {
                hostInput.value = "";
            }
        } else if (workerType === "cloud") {
            hideServerless();
            hostGroup.group.style.display = "flex";
            portGroup.group.style.display = "flex";
            cudaGroup.group.style.display = "none";
            argsGroup.group.style.display = "none";
            runpodText.style.display = "block";
            hostInput.placeholder = "e.g., your-cloud-worker.trycloudflare.com";
            if (applyDefaults) portInput.value = "443";
            if (applyDefaults && (hostInput.value === "localhost" || hostInput.value === "127.0.0.1")) {
                hostInput.value = "";
            }
        } else if (workerType === "serverless") {
            hostGroup.group.style.display = "none";
            portGroup.group.style.display = "none";
            cudaGroup.group.style.display = "none";
            argsGroup.group.style.display = "none";
            runpodText.style.display = "none";
            serverlessGroups.forEach((group) => { group.style.display = "flex"; });
        }
    };

    typeGroup.appendChild(typeLabel);
    typeGroup.appendChild(typeSelect);
    typeGroup.appendChild(runpodText);
    form.appendChild(typeGroup);

    const hostGroup = ui.createFormGroup("Host:", worker.host || "", `host-${worker.id}`, "text", "e.g., 192.168.1.100");
    hostGroup.group.id = `host-group-${worker.id}`;
    hostGroup.group.style.display = (isRemoteWorker(extension, worker) || worker.type === "cloud") ? "flex" : "none";
    form.appendChild(hostGroup.group);

    const portGroup = ui.createFormGroup("Port:", worker.port, `port-${worker.id}`, "number");
    portGroup.group.id = `port-group-${worker.id}`;
    form.appendChild(portGroup.group);

    const cudaGroup = ui.createFormGroup("CUDA Device:", worker.cuda_device || 0, `cuda-${worker.id}`, "number");
    cudaGroup.group.id = `cuda-group-${worker.id}`;
    cudaGroup.group.style.display = (isRemoteWorker(extension, worker) || worker.type === "cloud") ? "none" : "flex";
    form.appendChild(cudaGroup.group);

    const argsGroup = ui.createFormGroup("Extra Args:", worker.extra_args || "", `args-${worker.id}`);
    argsGroup.group.id = `args-group-${worker.id}`;
    argsGroup.group.style.display = (isRemoteWorker(extension, worker) || worker.type === "cloud") ? "none" : "flex";
    form.appendChild(argsGroup.group);

    const endpointGroup = ui.createFormGroup("Endpoint ID:", worker.endpoint_id || "", `endpoint-id-${worker.id}`, "text", "Runpod endpoint ID");
    endpointGroup.group.id = `endpoint-id-group-${worker.id}`;
    endpointGroup.group.style.display = "none";
    form.appendChild(endpointGroup.group);

    const apiKeyGroup = ui.createFormGroup("Runpod API Key (write-only):", "", `api-key-${worker.id}`, "password", worker.api_key_configured ? "Leave blank to keep saved API key" : "Runpod API key");
    apiKeyGroup.group.id = `api-key-group-${worker.id}`;
    apiKeyGroup.input.autocomplete = "new-password";
    apiKeyGroup.input.spellcheck = false;
    apiKeyGroup.group.style.display = "none";
    form.appendChild(apiKeyGroup.group);

    const clearApiKeyGroup = document.createElement("div");
    clearApiKeyGroup.id = `clear-api-key-group-${worker.id}`;
    clearApiKeyGroup.style.cssText = "display: none; align-items: center; gap: 8px;";
    const clearApiKeyInput = document.createElement("input");
    clearApiKeyInput.type = "checkbox";
    clearApiKeyInput.id = `clear-api-key-${worker.id}`;
    const clearApiKeyLabel = document.createElement("label");
    clearApiKeyLabel.htmlFor = clearApiKeyInput.id;
    clearApiKeyLabel.textContent = "Clear saved API key (disable pool first)";
    clearApiKeyGroup.appendChild(clearApiKeyInput);
    clearApiKeyGroup.appendChild(clearApiKeyLabel);
    clearApiKeyInput.onchange = () => {
        if (clearApiKeyInput.checked) apiKeyGroup.input.value = "";
        apiKeyGroup.input.disabled = clearApiKeyInput.checked;
    };
    form.appendChild(clearApiKeyGroup);

    const jobsPerRunGroup = ui.createFormGroup("Jobs per run:", worker.jobs_per_run ?? 1, `jobs-per-run-${worker.id}`, "number");
    jobsPerRunGroup.group.id = `jobs-per-run-group-${worker.id}`;
    jobsPerRunGroup.input.min = "1";
    jobsPerRunGroup.input.max = "16";
    jobsPerRunGroup.input.step = "1";
    jobsPerRunGroup.input.title = "Maximum jobs submitted in one run";
    jobsPerRunGroup.group.style.display = "none";
    form.appendChild(jobsPerRunGroup.group);

    const concurrencyGroup = ui.createFormGroup("Concurrent jobs (slots):", worker.concurrency ?? 1, `concurrency-${worker.id}`, "number");
    concurrencyGroup.group.id = `concurrency-group-${worker.id}`;
    concurrencyGroup.input.min = "1";
    concurrencyGroup.input.max = "16";
    concurrencyGroup.input.step = "1";
    concurrencyGroup.input.title = "Maximum concurrent jobs per serverless worker";
    concurrencyGroup.group.style.display = "none";
    form.appendChild(concurrencyGroup.group);

    const timeoutGroup = ui.createFormGroup("Timeout (seconds):", worker.timeout_seconds ?? 1800, `timeout-seconds-${worker.id}`, "number");
    timeoutGroup.group.id = `timeout-seconds-group-${worker.id}`;
    timeoutGroup.input.min = "30";
    timeoutGroup.input.max = "86400";
    timeoutGroup.input.step = "1";
    timeoutGroup.group.style.display = "none";
    form.appendChild(timeoutGroup.group);

    const testServerlessButton = ui.createButton("Test Saved Endpoint", () => testServerlessWorker(extension, worker.id));
    testServerlessButton.id = `test-serverless-${worker.id}`;
    testServerlessButton.title = "Test the saved Runpod endpoint without submitting inference";
    testServerlessButton.style.cssText = BUTTON_STYLES.base + BUTTON_STYLES.workerControl;
    const testServerlessGroup = document.createElement("div");
    testServerlessGroup.id = `serverless-test-group-${worker.id}`;
    testServerlessGroup.style.cssText = "display: none;";
    testServerlessGroup.appendChild(testServerlessButton);
    form.appendChild(testServerlessGroup);

    const saveBtn = ui.createButton("Save", () => saveWorkerSettings(extension, worker.id), "background-color: #4a7c4a;");
    saveBtn.style.cssText = BUTTON_STYLES.base + BUTTON_STYLES.success;

    const cancelBtn = ui.createButton("Cancel", () => cancelWorkerSettings(extension, worker.id), "background-color: #555;");
    cancelBtn.style.cssText = BUTTON_STYLES.base + BUTTON_STYLES.cancel;

    const deleteBtn = ui.createButton("Delete", () => deleteWorker(extension, worker.id), "background-color: #7c4a4a;");
    deleteBtn.style.cssText = BUTTON_STYLES.base + BUTTON_STYLES.error + BUTTON_STYLES.marginLeftAuto;

    const buttonGroup = ui.createButtonGroup([saveBtn, cancelBtn, deleteBtn], " margin-top: 8px;");
    form.appendChild(buttonGroup);

    typeSelect.onchange = (event) => updateWorkerType(event.target.value, true);

    if (worker.type === "serverless") {
        typeSelect.value = "serverless";
    } else if (worker.type === "cloud") {
        typeSelect.value = "cloud";
        runpodText.style.display = "block";
    } else if (isRemoteWorker(extension, worker)) {
        typeSelect.value = "remote";
    } else {
        typeSelect.value = "local";
    }

    updateWorkerType(typeSelect.value);

    return form;
}
