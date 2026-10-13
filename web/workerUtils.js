import { TIMEOUTS, ENDPOINTS } from './constants.js';
import { checkAllWorkerStatuses, getWorkerUrl } from './workerLifecycle.js';

export async function handleWorkerOperation(extension, button, operation, successText, errorText) {
    const originalText = button.textContent;
    const originalStyle = button.style.cssText;
    const originalClasses = Array.from(button.classList);
    const stateClasses = ["btn--working", "btn--success", "btn--error"];

    const setButtonStateClass = (className) => {
        button.classList.remove(...stateClasses);
        if (className) {
            button.classList.add(className);
        }
    };
    
    button.textContent = operation.loadingText;
    button.disabled = true;
    setButtonStateClass("btn--working");
    
    try {
        const workersToProcess = extension.enabledWorkers.filter((worker) =>
            operation.endpoint !== ENDPOINTS.CLEAR_MEMORY || worker.type !== "serverless"
        );
        const targets = workersToProcess.map((worker) => ({
            name: worker.name,
            worker,
            serverless: worker.type === "serverless",
            url: worker.type === "serverless" ? null : getWorkerUrl(extension, worker),
        }));

        if (targets.length === 0) {
            button.textContent = "No Workers";
            setButtonStateClass("btn--error");
            setTimeout(() => {
                button.textContent = originalText;
                button.style.cssText = originalStyle;
                button.classList.remove(...stateClasses);
                button.classList.add(...originalClasses);
                button.disabled = false;
            }, TIMEOUTS.BUTTON_RESET);
            return;
        }
        
        const promises = targets.map((target) => {
            if (target.serverless && operation.name === "Interrupt") {
                return extension.api.cancelServerlessWorker(target.worker.id)
                    .then((result) => ({ ok: result?.ok !== false, name: target.name }))
                    .catch(() => ({ ok: false, name: target.name }));
            }
            return fetch(`${target.url}${operation.endpoint}`, {
                method: 'POST',
                mode: 'cors'
            })
                .then(response => ({ ok: response.ok, name: target.name }))
                .catch(() => ({ ok: false, name: target.name }));
        });
        
        const results = await Promise.all(promises);
        const failures = results.filter(r => !r.ok);
        
        if (failures.length === 0) {
            button.textContent = successText;
            setButtonStateClass("btn--success");
            if (operation.onSuccess) operation.onSuccess();
        } else {
            button.textContent = errorText;
            setButtonStateClass("btn--error");
            extension.log(`${operation.name} failed on: ${failures.map(f => f.name).join(", ")}`, "error");
        }
        
        setTimeout(() => {
            button.textContent = originalText;
            button.style.cssText = originalStyle;
            button.classList.remove(...stateClasses);
            button.classList.add(...originalClasses);
        }, TIMEOUTS.BUTTON_RESET);
    } finally {
        button.disabled = false;
    }
}

export async function handleInterruptWorkers(extension, button) {
    return handleWorkerOperation(extension, button, {
        name: "Interrupt",
        endpoint: ENDPOINTS.INTERRUPT,
        loadingText: "Interrupting...",
        onSuccess: () => setTimeout(() => checkAllWorkerStatuses(extension), TIMEOUTS.POST_ACTION_DELAY)
    }, "Interrupted!", "Error! See Console");
}

export async function handleClearMemory(extension, button) {
    return handleWorkerOperation(extension, button, {
        name: "Clear memory",
        endpoint: ENDPOINTS.CLEAR_MEMORY,
        loadingText: "Clearing..."
    }, "Success!", "Error! See Console");
}

export function findNodesByClass(apiPrompt, className) {
    return Object.entries(apiPrompt)
        .filter(([, nodeData]) => nodeData.class_type === className)
        .map(([nodeId, nodeData]) => ({ id: nodeId, data: nodeData }));
}


export function applyProbeResultToWorkerDot(workerId, probeResult) {
    const dot = document.getElementById(`status-${workerId}`);
    if (!dot) {
        return;
    }

    dot.classList.remove(
        'worker-status--online',
        'worker-status--offline',
        'worker-status--processing',
        'worker-status--unknown',
        'status-pulsing',
    );

    if (!probeResult || !probeResult.ok) {
        dot.classList.add('worker-status--offline');
        dot.title = 'Offline - Cannot connect';
        return;
    }

    if ((probeResult.queueRemaining || 0) > 0) {
        dot.classList.add('worker-status--processing');
        dot.title = `Processing (${probeResult.queueRemaining} queued)`;
        return;
    }

    dot.classList.add('worker-status--online');
    dot.title = 'Online - Idle';
}
