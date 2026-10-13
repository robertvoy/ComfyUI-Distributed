import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { DistributedUI } from './ui.js';

import { createStateManager } from './stateManager.js';
import { createApiClient } from './apiClient.js';
import { renderSidebarContent, updateWorkerCard } from './sidebarRenderer.js';
import { handleInterruptWorkers, handleClearMemory } from './workerUtils.js';
import { setupInterceptor } from './executionUtils.js';
import { PULSE_ANIMATION_CSS, TIMEOUTS, STATUS_COLORS } from './constants.js';
import { updateTunnelUIElements, refreshTunnelStatus, handleTunnelToggle } from './tunnelManager.js';
import { checkAllWorkerStatuses, checkWorkerStatus, loadManagedWorkers } from './workerLifecycle.js';
import { detectMasterIP } from './masterDetection.js';
import { parseHostInput, getMasterUrl as buildMasterUrl } from './urlUtils.js';

const WORKERS_CHANGED_EVENT = "distributed:workers-changed";

// Exported for tests; ComfyUI only uses the registerExtension side effect below.
export class DistributedExtension {
    constructor() {
        this.config = null;
        this.originalQueuePrompt = api.queuePrompt.bind(api);
        this.logAutoRefreshInterval = null;
        this.masterSettingsExpanded = false;
        this.app = app; // Store app reference for toast notifications
        this.tunnelStatus = { status: "unknown" };
        this.tunnelElements = {};
        
        // Initialize centralized state
        this.state = createStateManager();
        
        // Initialize UI component factory
        this.ui = new DistributedUI();
        
        // Initialize API client
        this.api = createApiClient(window.location.origin);
        
        // Initialize status check timeout reference
        this.statusCheckTimeout = null;
        
        // Initialize abort controller for status checks
        this.statusCheckAbortController = null;
        this.themeMutationObserver = null;

        // Inject CSS for pulsing animation
        this.injectStyles();

        this.loadConfig().then(async () => {
            this.registerSidebarTab();
            this.setupInterceptor();
            // Don't start polling until panel opens
            // this.startStatusChecking();
            loadManagedWorkers(this);
            // Detect master IP after everything is set up
            this.detectMasterIP();
            // Listen for Nodes 2.0 setting changes (once, for the lifetime of the extension)
            this._setupNodes2Listener();
        });
    }

    // Debug logging helpers
    log(message, level = "info") {
        if (level === "debug" && !this.config?.settings?.debug) return;
        if (level === "error") {
            console.error(`[Distributed] ${message}`);
        } else {
            console.log(`[Distributed] ${message}`);
        }
    }

    injectStyles() {
        const styleId = 'distributed-styles';
        if (!document.getElementById(styleId)) {
            const style = document.createElement('style');
            style.id = styleId;
            style.textContent = PULSE_ANIMATION_CSS;
            document.head.appendChild(style);
        }

        const fileStyleId = 'distributed-file-styles';
        if (!document.getElementById(fileStyleId)) {
            const style = document.createElement('style');
            style.id = fileStyleId;
            fetch(new URL('./distributed.css', import.meta.url))
                .then((response) => response.text())
                .then((cssText) => {
                    style.textContent = cssText;
                })
                .catch((error) => {
                    this.log(`Failed to load distributed.css: ${error.message}`, "error");
                });
            document.head.appendChild(style);
        }
    }

    // --- State & Config Management (Single Source of Truth) ---

    get enabledWorkers() {
        return this.config?.workers?.filter(w => w.enabled) || [];
    }

    get isEnabled() {
        return this.enabledWorkers.length > 0;
    }

    isMasterParticipationEnabled() {
        return !Boolean(this.config?.settings?.master_delegate_only);
    }

    isMasterFallbackActive() {
        return Boolean(this.config?.settings?.master_delegate_only) && this.enabledWorkers.length === 0;
    }

    isMasterParticipating() {
        return this.isMasterParticipationEnabled() || this.isMasterFallbackActive();
    }

    async updateMasterParticipation(enabled) {
        if (!this.config?.settings) {
            this.config.settings = {};
        }
        const delegateOnly = !enabled;
        if (this.config.settings.master_delegate_only === delegateOnly) {
            return;
        }

        await this._updateSetting('master_delegate_only', delegateOnly);

        if (this.panelElement) {
            renderSidebarContent(this, this.panelElement);
        }
    }

    async loadConfig() {
        try {
            this.config = await this.api.getConfig();
            this.log("Loaded config: " + JSON.stringify(this.config), "debug");
            
            // Ensure default flag values
            if (!this.config.settings) {
                this.config.settings = {};
            }
            if (this.config.settings.has_auto_populated_workers === undefined) {
                this.config.settings.has_auto_populated_workers = false;
            }
            
            // Load stored master CUDA device
            this.masterCudaDevice = this.config?.master?.cuda_device ?? undefined;
            
            // Sync to state
            if (this.config.workers) {
                this.config.workers.forEach(w => {
                    this.state.updateWorker(w.id, { enabled: w.enabled });
                });
            }
            this._emitWorkersChanged();
        } catch (error) {
            this.log("Failed to load config: " + error.message, "error");
            this.config = { workers: [], settings: { has_auto_populated_workers: false } };
        }
    }

    _emitWorkersChanged() {
        if (typeof window === "undefined" || typeof window.dispatchEvent !== "function") {
            return;
        }
        window.dispatchEvent(new CustomEvent(WORKERS_CHANGED_EVENT, {
            detail: { workers: this.config?.workers || [] },
        }));
    }

    _applyMasterHost(host) {
        if (!host || !this.config) return;
        if (!this.config.master) this.config.master = {};
        this.config.master.host = host;
        const hostInput = document.getElementById('master-host');
        if (hostInput) {
            hostInput.value = host;
        }
    }

    _parseHostInput(value) {
        return parseHostInput(value);
    }

    updateTunnelUIElements(isRunning, isStarting) {
        return updateTunnelUIElements(this, isRunning, isStarting);
    }

    async refreshTunnelStatus() {
        return refreshTunnelStatus(this);
    }

    async handleTunnelToggle(button) {
        return handleTunnelToggle(this, button);
    }

    async updateWorkerEnabled(workerId, enabled) {
        const worker = this.config.workers.find(w => w.id === workerId);
        const previousEnabled = worker ? Boolean(worker.enabled) : !enabled;
        if (worker) {
            worker.enabled = enabled;
            this.state.updateWorker(workerId, { enabled });
            this._emitWorkersChanged();

            // Immediately update status dot based on enabled state
            const statusDot = document.getElementById(`status-${workerId}`);
            if (statusDot) {
                if (enabled) {
                    // Enabled: Start with checking state and trigger check
                    this.ui.updateStatusDot(workerId, STATUS_COLORS.OFFLINE_RED, "Checking status...", false);
                    setTimeout(() => checkWorkerStatus(this, worker), TIMEOUTS.STATUS_CHECK_DELAY);
                } else {
                    // Disabled: Set to gray
                    this.ui.updateStatusDot(workerId, STATUS_COLORS.DISABLED_GRAY, "Disabled", false);
                }
            }
        }
        
        try {
            await this.api.updateWorker(workerId, { enabled });
        } catch (error) {
            this.log("Error updating worker: " + error.message, "error");
            // The backend kept the old state (e.g. a serverless worker without a
            // saved API key); show that instead of a toggle that did not stick.
            if (worker) {
                worker.enabled = previousEnabled;
                this.state.updateWorker(workerId, { enabled: previousEnabled });
                this._emitWorkersChanged();
            }
            this.app?.extensionManager?.toast?.add({
                severity: "error",
                summary: enabled ? "Could not enable worker" : "Could not disable worker",
                detail: error.message,
                life: 5000,
            });
        }

        if (this.panelElement) {
            await renderSidebarContent(this, this.panelElement);
        }
    }

    async _updateSetting(key, value) {
        // Update local config
        if (!this.config.settings) {
            this.config.settings = {};
        }
        this.config.settings[key] = value;
        
        try {
            await this.api.updateSetting(key, value);

            const prettyKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            let detail;
            if (key === 'worker_timeout_seconds') {
                const secs = parseInt(value, 10);
                detail = `Worker Timeout set to ${Number.isFinite(secs) ? secs : value}s`;
            } else if (typeof value === 'boolean') {
                detail = `${prettyKey} ${value ? 'enabled' : 'disabled'}`;
            } else {
                detail = `${prettyKey} set to ${value}`;
            }

            app.extensionManager.toast.add({
                severity: "success",
                summary: "Setting Updated",
                detail,
                life: 2000
            });
        } catch (error) {
            this.log(`Error updating setting '${key}': ${error.message}`, "error");
            app.extensionManager.toast.add({
                severity: "error",
                summary: "Setting Update Failed",
                detail: error.message,
                life: 3000
            });
        }
    }

    // --- UI Rendering ---

    registerSidebarTab() {
        app.extensionManager.registerSidebarTab({
            id: "distributed",
            icon: "pi pi-server",
            title: "Distributed",
            tooltip: "Distributed Control Panel",
            type: "custom",
            render: (el) => {
                this.panelElement = el;
                this.onPanelOpen();
                renderSidebarContent(this, el);
                this._applyNodes2Style();
                this._applyThemeToneClass();
            },
            destroy: () => {
                this.onPanelClose();
            }
        });
    }
    
    onPanelOpen() {
        this.log("Panel opened - starting status polling", "debug");
        if (!this.statusCheckTimeout) {
            checkAllWorkerStatuses(this);
        }
        this._startThemeObserver();
        this._applyThemeToneClass();
    }

    onPanelClose() {
        this.log("Panel closed - stopping status polling", "debug");

        // Cancel any pending status checks
        if (this.statusCheckAbortController) {
            this.statusCheckAbortController.abort();
            this.statusCheckAbortController = null;
        }

        // Clear the timeout
        if (this.statusCheckTimeout) {
            clearTimeout(this.statusCheckTimeout);
            this.statusCheckTimeout = null;
        }
        this._stopThemeObserver();

        this.panelElement = null;
    }

    _applyNodes2Style() {
        if (!this.panelElement) return;
        const enabled = app.ui.settings.getSettingValue("Comfy.VueNodes.Enabled") ?? false;
        this.panelElement.classList.toggle('distributed-panel--nodes2', Boolean(enabled));
    }

    _parseColorToRgba(colorValue) {
        if (!colorValue || typeof colorValue !== "string") {
            return null;
        }

        const color = colorValue.trim().toLowerCase();
        if (!color || color === "transparent") {
            return null;
        }

        const rgbMatch = color.match(/^rgba?\(([^)]+)\)$/);
        if (rgbMatch) {
            const parts = rgbMatch[1].split(",").map((part) => Number(part.trim()));
            if (parts.length < 3 || parts.slice(0, 3).some((part) => Number.isNaN(part))) {
                return null;
            }
            const alpha = parts.length >= 4 && Number.isFinite(parts[3]) ? parts[3] : 1;
            return {
                r: Math.max(0, Math.min(255, parts[0])),
                g: Math.max(0, Math.min(255, parts[1])),
                b: Math.max(0, Math.min(255, parts[2])),
                a: Math.max(0, Math.min(1, alpha)),
            };
        }

        const hexMatch = color.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
        if (hexMatch) {
            const value = hexMatch[1];
            const expanded = value.length === 3
                ? value.split("").map((c) => `${c}${c}`).join("")
                : value;
            const r = parseInt(expanded.slice(0, 2), 16);
            const g = parseInt(expanded.slice(2, 4), 16);
            const b = parseInt(expanded.slice(4, 6), 16);
            return { r, g, b, a: 1 };
        }

        return null;
    }

    _isPanelLightTheme() {
        const fallbackLight = window.matchMedia?.("(prefers-color-scheme: light)")?.matches || false;
        if (!this.panelElement) {
            return fallbackLight;
        }

        let current = this.panelElement;
        while (current) {
            const bg = getComputedStyle(current).backgroundColor;
            const rgba = this._parseColorToRgba(bg);
            if (rgba && rgba.a > 0.02) {
                // Relative luminance approximation (0..1)
                const luminance = (0.2126 * rgba.r + 0.7152 * rgba.g + 0.0722 * rgba.b) / 255;
                return luminance > 0.58;
            }
            current = current.parentElement;
        }

        return fallbackLight;
    }

    _applyThemeToneClass() {
        if (!this.panelElement) {
            return;
        }
        this.panelElement.classList.toggle("distributed-panel--light", this._isPanelLightTheme());
    }

    _startThemeObserver() {
        if (this.themeMutationObserver) {
            return;
        }

        this.themeMutationObserver = new MutationObserver(() => {
            this._applyThemeToneClass();
        });
        this.themeMutationObserver.observe(document.documentElement, {
            attributes: true,
            attributeFilter: ["class", "style"],
        });
        if (document.body) {
            this.themeMutationObserver.observe(document.body, {
                attributes: true,
                attributeFilter: ["class", "style"],
            });
        }
    }

    _stopThemeObserver() {
        if (!this.themeMutationObserver) {
            return;
        }
        this.themeMutationObserver.disconnect();
        this.themeMutationObserver = null;
    }

    _setupNodes2Listener() {
        app.ui.settings.addEventListener("Comfy.VueNodes.Enabled.change", (e) => {
            const enabled = e.detail?.value ?? false;
            if (this.panelElement) {
                this.panelElement.classList.toggle('distributed-panel--nodes2', Boolean(enabled));
                this._applyThemeToneClass();
            }
        });
    }

    // --- Core Logic & Execution ---

    setupInterceptor() {
        setupInterceptor(this);
    }

    updateWorkerCard(workerId, newStatus) {
        return updateWorkerCard(this, workerId, newStatus);
    }

    /**
     * Cleanup method to stop intervals and listeners
     */
    cleanup() {
        if (this.logAutoRefreshInterval) {
            clearInterval(this.logAutoRefreshInterval);
            this.logAutoRefreshInterval = null;
        }

        if (this.statusCheckTimeout) {
            clearTimeout(this.statusCheckTimeout);
            this.statusCheckTimeout = null;
        }

        this.log("Cleaned up intervals", "debug");
    }

    getMasterUrl() {
        return buildMasterUrl(this.config, window.location, (message, level) => this.log(message, level));
    }

    async detectMasterIP() {
        return detectMasterIP(this);
    }

    _handleInterruptWorkers(button) {
        return handleInterruptWorkers(this, button);
    }

    _handleClearMemory(button) {
        return handleClearMemory(this, button);
    }
}

app.registerExtension({
    name: "Distributed.Panel",
    async setup() {
        new DistributedExtension();
    }
});
