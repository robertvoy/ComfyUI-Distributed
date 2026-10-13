import { STATUS_COLORS } from './constants.js';
import { checkAllWorkerStatuses, loadManagedWorkers, updateWorkerControls } from './workerLifecycle.js';
import { renderActionsSection } from './sidebar/actionsSection.js';
import { renderSettingsSection } from './sidebar/settingsSection.js';
import { renderWorkersSection } from './sidebar/workersSection.js';

export function updateWorkerCard(extension, workerId, newStatus = {}) {
    const card = document.querySelector(`[data-worker-id="${workerId}"]`);
    if (!card) {
        return false;
    }

    const worker = extension.config?.workers?.find((w) => w.id === workerId);
    if (!worker) {
        return false;
    }

    const workerState = extension.state.getWorker(workerId);
    const isLaunching = Boolean(workerState?.launching);

    if (worker.type === "serverless") {
        const providerStatus = workerState?.serverlessStatus || {};
        if (newStatus.online && newStatus.processing) {
            extension.ui.updateStatusDot(workerId, STATUS_COLORS.PROCESSING_YELLOW, `Runpod active (${newStatus.queueCount || 0} queued)`, false);
        } else if (newStatus.online) {
            extension.ui.updateStatusDot(workerId, STATUS_COLORS.ONLINE_GREEN, providerStatus.message || "Runpod ready to scale", false);
        } else if (worker.enabled) {
            extension.ui.updateStatusDot(workerId, STATUS_COLORS.OFFLINE_RED, providerStatus.message || "Runpod endpoint unavailable", false);
        }
        updateWorkerControls(extension, workerId);
        return true;
    }

    if (isLaunching && !newStatus.online) {
        extension.ui.updateStatusDot(workerId, STATUS_COLORS.PROCESSING_YELLOW, "Launching...", true);
    } else if (newStatus.online && newStatus.processing) {
        const queue = newStatus.queueCount || 0;
        extension.ui.updateStatusDot(workerId, STATUS_COLORS.PROCESSING_YELLOW, `Online - Processing (${queue} in queue)`, false);
    } else if (newStatus.online) {
        extension.ui.updateStatusDot(workerId, STATUS_COLORS.ONLINE_GREEN, "Online - Idle", false);
    } else if (worker.enabled) {
        extension.ui.updateStatusDot(workerId, STATUS_COLORS.OFFLINE_RED, "Offline - Cannot connect", false);
    }

    updateWorkerControls(extension, workerId);
    return true;
}

export async function renderSidebarContent(extension, el) {
    // Panel is being opened/rendered
    extension.log("Panel opened", "debug");
    if (!el) {
        extension.log("No element provided to renderSidebarContent", "debug");
        return;
    }
    // Prevent infinite recursion
    if (extension._isRendering) {
        extension.log("Already rendering, skipping", "debug");
        return;
    }
    extension._isRendering = true;
    try {
        // Store reference to the panel element
        extension.panelElement = el;
        // Show loading indicator
        el.innerHTML = '';
        const loadingDiv = document.createElement("div");
        loadingDiv.style.cssText =
            "display: flex; align-items: center; justify-content: center; height: calc(100vh - 100px); color: var(--dist-muted-text, #888);";
        loadingDiv.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" style="color: var(--dist-muted-text, #888);">
            <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-dasharray="40 40"/>
        </svg>`;
        el.appendChild(loadingDiv);
        // Add rotation animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes rotate {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
        loadingDiv.querySelector('svg').style.animation = 'rotate 1s linear infinite';
        // Preload data outside render
        await Promise.all([extension.loadConfig(), loadManagedWorkers(extension), extension.refreshTunnelStatus()]);
        extension.tunnelElements = {};
        el.innerHTML = '';
        // Create toolbar header to match ComfyUI style
        const toolbar = document.createElement("div");
        toolbar.className = "p-toolbar p-component border-x-0 border-t-0 rounded-none px-2 py-1 min-h-8";
        toolbar.style.cssText =
            "border-bottom: 1px solid var(--dist-divider, #444); background: transparent; display: flex; align-items: center;";
        const toolbarStart = document.createElement("div");
        toolbarStart.className = "p-toolbar-start";
        toolbarStart.style.cssText = "display: flex; align-items: center;";
        const titleSpan = document.createElement("span");
        titleSpan.className = "text-xs 2xl:text-sm truncate";
        titleSpan.textContent = "COMFYUI DISTRIBUTED";
        titleSpan.title = "ComfyUI Distributed";
        toolbarStart.appendChild(titleSpan);
        toolbar.appendChild(toolbarStart);
        const toolbarCenter = document.createElement("div");
        toolbarCenter.className = "p-toolbar-center";
        toolbar.appendChild(toolbarCenter);
        const toolbarEnd = document.createElement("div");
        toolbarEnd.className = "p-toolbar-end";
        toolbar.appendChild(toolbarEnd);
        el.appendChild(toolbar);
        // Main container with adjusted padding
        const container = document.createElement("div");
        container.style.cssText = "padding: 15px; display: flex; flex-direction: column; height: calc(100% - 32px);";
        // Detect master info on panel open (in case CUDA info wasn't available at startup)
        extension.log(`Panel opened. CUDA device count: ${extension.cudaDeviceCount}, Workers: ${extension.config?.workers?.length || 0}`, "debug");
        if (!extension.cudaDeviceCount) {
            await extension.detectMasterIP();
        }
        // Now render with guaranteed up-to-date config
        // Master Node Section
        const masterDiv = extension.ui.renderEntityCard('master', extension.config?.master, extension);
        container.appendChild(masterDiv);
        container.appendChild(renderWorkersSection(extension));
        container.appendChild(renderActionsSection(extension));
        container.appendChild(renderSettingsSection(extension));
        el.appendChild(container);
        extension._applyThemeToneClass?.();
        // Start checking worker statuses immediately in parallel
        setTimeout(() => checkAllWorkerStatuses(extension), 0);
    } finally {
        // Always reset the rendering flag
        extension._isRendering = false;
    }
}
