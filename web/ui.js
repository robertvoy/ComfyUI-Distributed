import { BUTTON_STYLES, UI_STYLES, STATUS_COLORS, UI_COLORS, TIMEOUTS } from './constants.js';
import { createButtonHelper as createButtonHelperFn } from './ui/buttonHelpers.js';
import { showCloudflareWarning as showCloudflareWarningFn } from './ui/cloudflareWarning.js';
import { createWorkerSettingsForm as createWorkerSettingsFormFn } from './ui/settingsForm.js';
import { renderEntityCard as renderEntityCardFn } from './ui/entityCard.js';
import { createLogModal } from './ui/logModal.js';
import { launchWorker, stopWorker, updateWorkerControls, viewWorkerLog } from './workerLifecycle.js';
import { isRemoteWorker } from './workerSettings.js';

const cardConfigs = {
    master: {
        checkbox: { 
            enabled: true,
            masterToggle: true,
            title: "Toggle master participation in workloads"
        },
        statusDot: { 
            id: 'master-status',
            initialColor: (_, extension) => extension.isMasterParticipating() ? STATUS_COLORS.ONLINE_GREEN : STATUS_COLORS.DISABLED_GRAY,
            initialTitle: (_, extension) => extension.isMasterParticipating() ? 'Master participating' : 'Master orchestrator only',
            dynamic: true
        },
        infoText: (data, extension) => {
            const cudaDevice = extension.config?.master?.cuda_device ?? extension.masterCudaDevice;
            const cudaInfo = cudaDevice !== undefined ? `CUDA ${cudaDevice} • ` : '';
            const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
            const participationEnabled = extension.isMasterParticipationEnabled();
            const fallbackActive = extension.isMasterFallbackActive();
            let delegateBadge = '';
            if (!participationEnabled && fallbackActive) {
                delegateBadge = `<br><span class="dist-worker-info__fallback">Fallback active • Master executing</span>`;
            }
            return `<span class="dist-worker-info__title" id="master-name-display">${data?.name || extension.config?.master?.name || "Master"}</span><br><span class="dist-worker-info__meta"><span id="master-cuda-info">${cudaInfo}Port ${port}</span></span>${delegateBadge}`;
        },
        controls: { 
            type: 'master'
        },
        settings: { 
            formType: 'master', 
            id: 'master-settings',
            expandedTracker: 'masterSettingsExpanded'
        },
        hover: true,
        expand: true,
        border: 'solid'
    },
    worker: {
        checkbox: { 
            enabled: true, 
            title: "Enable/disable this worker" 
        },
        statusDot: { 
            dynamic: true,
            initialColor: (data) => data.enabled ? STATUS_COLORS.OFFLINE_RED : STATUS_COLORS.DISABLED_GRAY,
            initialTitle: (data) => data.enabled ? "Checking status..." : "Disabled",
            id: (data) => `status-${data.id}`
        },
        infoText: (data, extension) => {
            const isRemote = isRemoteWorker(extension, data);
            const isCloud = data.type === 'cloud';
            
            if (isCloud) {
                // For cloud workers, don't show port (it's always 443)
                return `<span class="dist-worker-info__title">${data.name}</span><br><span class="dist-worker-info__meta">${data.host}</span>`;
            } else if (isRemote) {
                const hostLabel = data.host
                    ? `${data.host}:${data.port}`
                    : `Unconfigured remote worker • Port ${data.port}`;
                return `<span class="dist-worker-info__title">${data.name}</span><br><span class="dist-worker-info__meta">${hostLabel}</span>`;
            } else {
                const cudaInfo = data.cuda_device !== undefined ? `CUDA ${data.cuda_device} • ` : '';
                return `<span class="dist-worker-info__title">${data.name}</span><br><span class="dist-worker-info__meta">${cudaInfo}Port ${data.port}</span>`;
            }
        },
        controls: { 
            dynamic: true 
        },
        settings: { 
            formType: 'worker',
            id: (data) => `settings-${data.id}`,
            expandedId: (data) => data?.id
        },
        hover: true,
        expand: true,
        border: 'solid'
    },
    blueprint: {
        checkbox: { 
            type: 'icon', 
            content: '+', 
            width: 42,
            style: `border-right: 2px dashed ${UI_COLORS.BORDER_LIGHT}; color: ${UI_COLORS.ACCENT_COLOR}; font-size: 24px; font-weight: 500;` 
        },
        statusDot: { 
            color: 'transparent', 
            border: `1px solid ${UI_COLORS.BORDER_LIGHT}` 
        },
        infoText: () => `<strong style="color: #aaa; font-size: 16px;">Add New Worker</strong><br><small style="color: ${UI_COLORS.BORDER_LIGHT};">[CUDA] • [Port]</small>`,
        controls: { 
            type: 'ghost', 
            text: 'Configure', 
            style: `border: 1px solid ${UI_COLORS.BORDER_DARK}; background: transparent; color: ${UI_COLORS.BORDER_LIGHT};` 
        },
        hover: 'placeholder',
        expand: false,
        border: 'dashed'
    },
    add: {
        checkbox: { 
            type: 'icon', 
            content: '+',
            width: 43,
            style: `border-right: 1px dashed ${UI_COLORS.BORDER_DARK}; color: ${UI_COLORS.BORDER_LIGHT}; font-size: 18px;` 
        },
        statusDot: { 
            color: 'transparent', 
            border: `1px solid ${UI_COLORS.BORDER_LIGHT}` 
        },
        infoText: () => `<span style="color: ${UI_COLORS.ICON_COLOR}; font-weight: bold; font-size: 13px;">Add New Worker</span>`,
        controls: null,
        hover: 'placeholder',
        expand: false,
        border: 'dashed',
        minHeight: '48px'
    }
};

export class DistributedUI {
    constructor() {
        // UI element styles
        this.styles = UI_STYLES;
    }

    createStatusDot(id, color = "#666", title = "Status") {
        const dot = document.createElement("span");
        if (id) dot.id = id;
        dot.style.cssText = this.styles.statusDot + ` background-color: ${color};`;
        dot.title = title;
        return dot;
    }

    createButton(text, onClick, customStyle = "") {
        const button = document.createElement("button");
        button.textContent = text;
        button.className = "distributed-button";
        button.style.cssText = BUTTON_STYLES.base + customStyle;
        if (onClick) button.onclick = onClick;
        return button;
    }

    createButtonGroup(buttons, style = "") {
        const group = document.createElement("div");
        group.style.cssText = this.styles.buttonGroup + style;
        buttons.forEach(button => group.appendChild(button));
        return group;
    }

    createWorkerControls(workerId, handlers = {}) {
        const controlsDiv = document.createElement("div");
        controlsDiv.id = `controls-${workerId}`;
        controlsDiv.style.cssText = this.styles.controlsDiv;
        
        const buttons = [];
        
        if (handlers.launch) {
            const launchBtn = this.createButton('Launch', handlers.launch);
            launchBtn.id = `launch-${workerId}`;
            launchBtn.title = "Launch this worker instance";
            buttons.push(launchBtn);
        }
        
        if (handlers.stop) {
            const stopBtn = this.createButton('Stop', handlers.stop);
            stopBtn.id = `stop-${workerId}`;
            stopBtn.title = "Stop this worker instance";
            buttons.push(stopBtn);
        }
        
        if (handlers.viewLog) {
            const logBtn = this.createButton('View Log', handlers.viewLog);
            logBtn.id = `log-${workerId}`;
            logBtn.title = "View worker log file";
            buttons.push(logBtn);
        }
        
        buttons.forEach(btn => controlsDiv.appendChild(btn));
        return controlsDiv;
    }

    createFormGroup(label, value, id, type = "text", placeholder = "") {
        const group = document.createElement("div");
        group.style.cssText = this.styles.formGroup;
        
        const labelEl = document.createElement("label");
        labelEl.textContent = label;
        labelEl.htmlFor = id;
        labelEl.style.cssText = this.styles.formLabel;
        
        const input = document.createElement("input");
        input.type = type;
        input.id = id;
        input.value = value;
        input.placeholder = placeholder;
        input.classList.add('dist-form-input');
        input.style.cssText = this.styles.formInput;
        
        group.appendChild(labelEl);
        group.appendChild(input);
        return { group, input };
    }


    createInfoBox(text) {
        const box = document.createElement("div");
        box.classList.add('dist-info-box');
        box.style.cssText = this.styles.infoBox;
        box.textContent = text;
        return box;
    }

    addHoverEffect(element, onHover, onLeave) {
        element.onmouseover = onHover;
        element.onmouseout = onLeave;
    }

    createCard(type = 'worker', options = {}) {
        const card = document.createElement("div");
        card.classList.add('dist-card');

        switch(type) {
            case 'master':
            case 'worker':
                card.style.cssText = this.styles.workerCard;
                break;
            case 'blueprint':
                card.classList.add('dist-card--blueprint');
                card.style.cssText = this.styles.cardBase + this.styles.cardBlueprint;
                if (options.onClick) card.onclick = options.onClick;
                if (options.title) card.title = options.title;
                break;
            case 'add':
                card.classList.add('dist-card--add');
                card.style.cssText = this.styles.cardBase + this.styles.cardAdd;
                if (options.onClick) card.onclick = options.onClick;
                if (options.title) card.title = options.title;
                break;
        }
        
        if (options.onMouseEnter) {
            card.addEventListener('mouseenter', options.onMouseEnter);
        }
        if (options.onMouseLeave) {
            card.addEventListener('mouseleave', options.onMouseLeave);
        }
        
        return card;
    }

    createCardColumn(type = 'checkbox', options = {}) {
        const column = document.createElement("div");

        switch(type) {
            case 'checkbox':
                column.classList.add('dist-card__left-col');
                column.style.cssText = this.styles.checkboxColumn;
                if (options.title) column.title = options.title;
                break;
            case 'icon':
                column.style.cssText = this.styles.columnBase + this.styles.iconColumn;
                break;
            case 'content':
                column.style.cssText = this.styles.contentColumn;
                break;
        }
        
        return column;
    }

    createInfoRow(options = {}) {
        const row = document.createElement("div");
        row.style.cssText = this.styles.infoRow;
        if (options.onClick) row.onclick = options.onClick;
        return row;
    }

    createWorkerContent() {
        const content = document.createElement("div");
        content.style.cssText = this.styles.workerContent;
        return content;
    }

    createSettingsForm(fields = [], options = {}) {
        const form = document.createElement("div");
        form.style.cssText = this.styles.settingsForm;
        
        fields.forEach(field => {
            if (field.type === 'checkbox') {
                const group = document.createElement("div");
                group.style.cssText = this.styles.checkboxGroup;
                
                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.id = field.id;
                checkbox.checked = field.checked || false;
                if (field.onChange) checkbox.onchange = field.onChange;
                
                const label = document.createElement("label");
                label.htmlFor = field.id;
                label.textContent = field.label;
                label.style.cssText = this.styles.formLabelClickable;
                
                group.appendChild(checkbox);
                group.appendChild(label);
                form.appendChild(group);
            } else {
                const result = this.createFormGroup(field.label, field.value, field.id, field.type, field.placeholder);
                if (field.groupId) result.group.id = field.groupId;
                if (field.display) result.group.style.display = field.display;
                form.appendChild(result.group);
            }
        });
        
        if (options.buttons) {
            const buttonGroup = this.createButtonGroup(options.buttons, options.buttonStyle || " margin-top: 8px;");
            form.appendChild(buttonGroup);
        }
        
        return form;
    }


    createButtonHelper(text, onClick, style) {
        return createButtonHelperFn(this, text, onClick, style);
    }

    updateMasterDisplay(extension) {
        // Use persistent config value as fallback
        const cudaDevice = extension?.config?.master?.cuda_device ?? extension?.masterCudaDevice;
        
        // Update CUDA info if element exists
        const cudaInfo = document.getElementById('master-cuda-info');
        if (cudaInfo) {
            const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
            if (cudaDevice !== undefined && cudaDevice !== null) {
                cudaInfo.textContent = `CUDA ${cudaDevice} • Port ${port}`;
            } else {
                cudaInfo.textContent = `Port ${port}`;
            }
        }
        
        // Update name if changed
        const nameDisplay = document.getElementById('master-name-display');
        if (nameDisplay && extension?.config?.master?.name) {
            nameDisplay.textContent = extension.config.master.name;
        }
    }

    showToast(app, severity, summary, detail, life = 3000) {
        if (app.extensionManager?.toast?.add) {
            app.extensionManager.toast.add({ severity, summary, detail, life });
        }
    }

    showCloudflareWarning(extension, masterHost) {
        return showCloudflareWarningFn(extension, masterHost);
    }

    updateStatusDot(workerId, color, title, pulsing = false) {
        const statusDot = document.getElementById(`status-${workerId}`);
        if (!statusDot) return;

        const statusClasses = [
            "worker-status--online",
            "worker-status--offline",
            "worker-status--unknown",
            "worker-status--processing",
        ];
        statusDot.classList.remove(...statusClasses);

        const colorClassMap = {
            [STATUS_COLORS.ONLINE_GREEN]: "worker-status--online",
            [STATUS_COLORS.OFFLINE_RED]: "worker-status--offline",
            [STATUS_COLORS.DISABLED_GRAY]: "worker-status--unknown",
            [STATUS_COLORS.PROCESSING_YELLOW]: "worker-status--processing",
        };

        const statusClass = colorClassMap[color] || "worker-status--unknown";
        statusDot.classList.add(statusClass);
        statusDot.style.backgroundColor = "";
        statusDot.title = title;
        statusDot.classList.toggle('status-pulsing', pulsing);
    }

    showLogModal(extension, workerId, logData, fetchLog = null) {
        if (this._logModal) {
            this._logModal.unmount();
            this._logModal = null;
        }

        const worker = extension.config.workers.find(w => w.id === workerId);
        const workerName = worker?.name || `Worker ${workerId}`;

        const modal = createLogModal();
        this._logModal = modal;
        const themeClass =
            extension.panelElement?.classList.contains("distributed-panel--light")
                ? "distributed-panel--light"
                : "";
        modal.mount(document.body, {
            workerName,
            logData,
            fetchLog: fetchLog || (async () => extension.api.getWorkerLog(workerId, 1000)),
            themeClass,
            onClose: () => {
                if (this._logModal === modal) {
                    this._logModal = null;
                }
            },
        });
    }

    createWorkerSettingsForm(extension, worker) {
        return createWorkerSettingsFormFn(this, extension, worker);
    }

    createSettingsToggle() {
        const settingsRow = document.createElement("div");
        settingsRow.style.cssText = this.styles.settingsToggle;
        
        const settingsTitle = document.createElement("h4");
        settingsTitle.textContent = "Settings";
        settingsTitle.style.cssText = "margin: 0; font-size: 14px;";
        
        const settingsToggle = document.createElement("span");
        settingsToggle.textContent = "▶"; // Right arrow when collapsed
        settingsToggle.style.cssText =
            "font-size: 12px; color: var(--dist-settings-arrow, #888); transition: all 0.2s ease;";
        
        settingsRow.appendChild(settingsToggle);
        settingsRow.appendChild(settingsTitle);
        
        return { settingsRow, settingsToggle };
    }


    createCheckboxOrIconColumn(config, data, extension) {
        const column = this.createCardColumn('checkbox');
        
        if (config?.type === 'icon') {
            column.style.flex = `0 0 ${config.width || 44}px`;
            column.innerHTML = config.content || '+';
            if (config.style) {
                const styles = config.style.split(';').filter(s => s.trim());
                styles.forEach(style => {
                    const [prop, value] = style.split(':').map(s => s.trim());
                    if (prop && value) {
                        column.style[prop.replace(/-([a-z])/g, (g) => g[1].toUpperCase())] = value;
                    }
                });
            }
        } else {
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.id = `gpu-${data?.id || 'master'}`;
            checkbox.checked = config?.checked !== undefined ? config.checked : data?.enabled;
            checkbox.disabled = config?.disabled || false;
            checkbox.style.cssText = `cursor: ${config?.disabled ? 'default' : 'pointer'}; width: 16px; height: 16px;`;
            
            if (config?.opacity) checkbox.style.opacity = config.opacity;
            if (config?.title) column.title = config.title;
            
            const isMasterToggle = config?.masterToggle && typeof extension.isMasterParticipating === 'function';
            if (isMasterToggle) {
                const participationEnabled = extension.isMasterParticipationEnabled();
                const fallbackActive = extension.isMasterFallbackActive();
                const buildTitle = (enabled, fallback) => {
                    if (enabled) {
                        return "Master participating • Click to switch to orchestrator-only";
                    }
                    if (fallback) {
                        return "No workers selected • Master fallback execution active";
                    }
                    return "Master orchestrator-only • Click to re-enable participation";
                };

                checkbox.checked = participationEnabled;
                checkbox.style.pointerEvents = "none";
                column.style.cursor = "pointer";
                column.title = buildTitle(participationEnabled, fallbackActive);
                column.onclick = async (event) => {
                    if (event) {
                        event.stopPropagation();
                        event.preventDefault();
                    }
                    const nextState = !extension.isMasterParticipationEnabled();
                    const nextFallback = !nextState && extension.enabledWorkers.length === 0;
                    checkbox.checked = nextState;
                    column.title = buildTitle(nextState, nextFallback);
                    await extension.updateMasterParticipation(nextState);
                };
            } else if (config?.enabled && !config?.disabled && data?.id) {
                checkbox.style.pointerEvents = "none";
                column.style.cursor = "pointer";
                column.onclick = async () => {
                    checkbox.checked = !checkbox.checked;
                    await extension.updateWorkerEnabled(data.id, checkbox.checked);
                };
            }
            
            column.appendChild(checkbox);
        }
        
        return column;
    }

    createStatusDotHelper(config, data, extension) {
        let color = config.color || "#666";
        let title = config.title || "Status";
        let id = config.id;
        
        if (typeof config.initialColor === 'function') {
            color = config.initialColor(data, extension);
        }
        if (typeof config.initialTitle === 'function') {
            title = config.initialTitle(data, extension);
        }
        if (typeof config.id === 'function') {
            id = config.id(data);
        }
        
        const dot = this.createStatusDot(id, color, title);
        
        if (config.border) {
            dot.style.border = config.border;
        }
        
        if (config.pulsing && (typeof config.pulsing !== 'function' || config.pulsing(data))) {
            dot.classList.add('status-pulsing');
        }
        
        return dot;
    }

    createSettingsToggleHelper(expandedId, extension) {
        const arrow = document.createElement("span");
        arrow.className = "settings-arrow";
        arrow.innerHTML = "▶";
        arrow.style.cssText = this.styles.settingsArrow;
        
        const isExpanded = typeof expandedId === 'function' ? 
            extension.state.isWorkerExpanded(expandedId(extension)) : 
            (expandedId === 'master' ? false : extension.state.isWorkerExpanded(expandedId));
            
        if (isExpanded) {
            arrow.style.transform = "rotate(90deg)";
        }
        
        return arrow;
    }

    createControlsSection(config, data, extension, isRemote) {
        if (!config) return null;
        
        const controlsDiv = document.createElement("div");
        controlsDiv.id = `controls-${data?.id || 'master'}`;
        controlsDiv.style.cssText = this.styles.controlsDiv;
        
        // Always create a wrapper div for consistent layout
        const controlsWrapper = document.createElement("div");
        controlsWrapper.style.cssText = this.styles.controlsWrapper;
        
        if (config.type === 'master') {
            const participationEnabled = extension.isMasterParticipationEnabled();
            const fallbackActive = extension.isMasterFallbackActive();
            let message;
            const badge = document.createElement("div");
            badge.classList.add("dist-info-box", "master-info-badge");
            badge.style.cssText = this.styles.infoBox;
            if (fallbackActive) {
                message = "No workers selected. Master fallback execution active.";
                badge.textContent = message;
                badge.classList.add("master-info-badge--fallback");
            } else if (!participationEnabled) {
                message = "Master disabled: running as orchestrator only";
                badge.textContent = message;
                badge.classList.add("master-info-badge--delegate");
            } else {
                message = "Master participating in workflows";
                badge.textContent = message;
            }
            controlsWrapper.appendChild(badge);
        } else if (config.dynamic && data) {
            if (isRemote) {
                const isCloud = data.type === 'cloud';
                const workerTypeText = isCloud ? "Cloud worker" : "Remote worker";
                const workerTypeBadge = this.createInfoBox(workerTypeText);
                workerTypeBadge.title = "Worker is externally hosted";
                controlsWrapper.appendChild(workerTypeBadge);

                const logBtn = this.createButton('View Log', () => viewWorkerLog(extension, data.id, true));
                logBtn.id = `log-${data.id}`;
                logBtn.style.cssText = BUTTON_STYLES.base + BUTTON_STYLES.workerControl;
                logBtn.classList.add("btn--log");
                logBtn.title = "View remote worker log";
                // Keep disabled workers from showing a log button before state sync.
                if (!data.enabled) {
                    logBtn.classList.add("is-hidden");
                    logBtn.style.display = "none";
                    logBtn.disabled = true;
                }
                controlsWrapper.appendChild(logBtn);
            } else {
                const controls = this.createWorkerControls(data.id, {
                    launch: () => launchWorker(extension, data.id),
                    stop: () => stopWorker(extension, data.id),
                    viewLog: () => viewWorkerLog(extension, data.id)
                });
                
                const launchBtn = controls.querySelector(`#launch-${data.id}`);
                const stopBtn = controls.querySelector(`#stop-${data.id}`);
                const logBtn = controls.querySelector(`#log-${data.id}`);
                
                launchBtn.style.cssText = BUTTON_STYLES.base + BUTTON_STYLES.workerControl;
                launchBtn.classList.add("btn--launch");
                launchBtn.title = "Launch worker (runs in background with logging)";
                
                stopBtn.style.cssText = BUTTON_STYLES.base + BUTTON_STYLES.workerControl + BUTTON_STYLES.hidden;
                stopBtn.classList.add("btn--stop");
                stopBtn.title = "Stop worker";
                
                logBtn.style.cssText = BUTTON_STYLES.base + BUTTON_STYLES.workerControl + BUTTON_STYLES.hidden;
                logBtn.classList.add("btn--log");
                
                while (controls.firstChild) {
                    controlsWrapper.appendChild(controls.firstChild);
                }
            }
        } else if (config.type === 'info') {
            const infoBtn = this.createButton(config.text, null, config.style || "");
            infoBtn.style.cssText = BUTTON_STYLES.base + BUTTON_STYLES.workerControl + (config.style || BUTTON_STYLES.info) + " cursor: default;";
            infoBtn.disabled = true;
            controlsWrapper.appendChild(infoBtn);
        } else if (config.type === 'ghost') {
            const ghostBtn = document.createElement("button");
            ghostBtn.style.cssText = `flex: 1; padding: 5px 14px; font-size: 11px; font-weight: 500; border-radius: 4px; cursor: default; ${config.style || ""}`;
            ghostBtn.textContent = config.text;
            ghostBtn.disabled = true;
            controlsWrapper.appendChild(ghostBtn);
        }
        
        controlsDiv.appendChild(controlsWrapper);
        return controlsDiv;
    }

    createSettingsSection(config, data, extension) {
        const settingsDiv = document.createElement("div");
        const settingsId = typeof config.id === 'function' ? config.id(data) : config.id;
        settingsDiv.id = settingsId;
        settingsDiv.className = "worker-settings";
        
        const expandedId = typeof config.expandedId === 'function' ? config.expandedId(data) : config.expandedId;
        const isExpanded = expandedId === 'master-settings' ? false : extension.state.isWorkerExpanded(expandedId);
        
        settingsDiv.style.cssText = this.styles.workerSettings;
        
        if (isExpanded) {
            settingsDiv.classList.add("expanded");
            settingsDiv.style.padding = "12px";
            settingsDiv.style.marginTop = "8px";
            settingsDiv.style.marginBottom = "8px";
        }
        
        let settingsForm;
        if (config.formType === 'master') {
            settingsForm = this.createMasterSettingsForm(extension, data);
        } else if (config.formType === 'worker') {
            settingsForm = this.createWorkerSettingsForm(extension, data);
        }
        
        if (settingsForm) {
            settingsDiv.appendChild(settingsForm);
        }
        
        return settingsDiv;
    }

    createMasterSettingsForm(extension, data) {
        const settingsForm = document.createElement("div");
        settingsForm.style.cssText = "display: flex; flex-direction: column; gap: 8px;";
        
        const nameResult = this.createFormGroup("Name:", extension.config?.master?.name || "Master", "master-name");
        settingsForm.appendChild(nameResult.group);
        
        const hostResult = this.createFormGroup("Host:", extension.config?.master?.host || "", "master-host", "text", "Auto-detect if empty");
        settingsForm.appendChild(hostResult.group);

        // Cloudflare tunnel toggle (simple button inside master settings)
        const tunnelBtn = this.createButton("Enable Cloudflare Tunnel", (e) => extension.handleTunnelToggle(e.target));
        tunnelBtn.id = "cloudflare-tunnel-button";
        tunnelBtn.style.cssText = BUTTON_STYLES.base + " margin: 4px 0 -5px 0;";
        tunnelBtn.classList.add("tunnel-button", "tunnel-button--enable");
        settingsForm.appendChild(tunnelBtn);
        extension.tunnelElements = { button: tunnelBtn };
        extension.updateTunnelUIElements();
        
        const saveBtn = this.createButton("Save", async () => {
            const nameInput = document.getElementById('master-name');
            const hostInput = document.getElementById('master-host');
            
            if (!extension.config.master) extension.config.master = {};
            extension.config.master.name = nameInput.value.trim() || "Master";
            
            const hostValue = hostInput.value.trim();
            
            await extension.api.updateMaster({
                host: hostValue,
                name: extension.config.master.name
            });
            
            // Reload config to refresh any updated values
            await extension.loadConfig();
            
            // If host was emptied, trigger auto-detection
            if (!hostValue) {
                extension.log("Host field cleared, triggering IP auto-detection", "debug");
                await extension.detectMasterIP();
                // Reload config again to get the auto-detected IP
                await extension.loadConfig();
                // Update the input field with the detected IP
                document.getElementById('master-host').value = extension.config?.master?.host || "";
            }
            
            document.getElementById('master-name-display').textContent = extension.config.master.name;
            this.updateMasterDisplay(extension);
            
            // Show toast notification
            if (extension.app?.extensionManager?.toast) {
                const message = !hostValue ? 
                    "Master settings saved and IP auto-detected" : 
                    "Master settings saved successfully";
                extension.app.extensionManager.toast.add({
                    severity: "success",
                    summary: "Master Updated",
                    detail: message,
                    life: 3000
                });
            }
            
            saveBtn.textContent = "Saved!";
            setTimeout(() => { saveBtn.textContent = "Save"; }, TIMEOUTS.FLASH_LONG);
        }, "background-color: #4a7c4a;");
        saveBtn.style.cssText = BUTTON_STYLES.base + BUTTON_STYLES.success;
        
        const cancelBtn = this.createButton("Cancel", () => {
            document.getElementById('master-name').value = extension.config?.master?.name || "Master";
            document.getElementById('master-host').value = extension.config?.master?.host || "";
        }, "background-color: #555;");
        cancelBtn.style.cssText = BUTTON_STYLES.base + BUTTON_STYLES.cancel;
        
        const buttonGroup = this.createButtonGroup([saveBtn, cancelBtn], " margin-top: 8px;");
        settingsForm.appendChild(buttonGroup);
        
        return settingsForm;
    }

    addPlaceholderHover(card, leftColumn, entityType) {
        const cardTypeClass = entityType === 'blueprint' ? 'placeholder-card--blueprint' : 'placeholder-card--add';
        const columnTypeClass = entityType === 'blueprint' ? 'placeholder-column--blueprint' : 'placeholder-column--add';
        card.classList.add('placeholder-card', cardTypeClass);
        leftColumn.classList.add('placeholder-column', columnTypeClass);

        card.onmouseover = () => {
            card.classList.add('is-hovered');
            leftColumn.classList.add('is-hovered');
        };
        
        card.onmouseout = () => {
            card.classList.remove('is-hovered');
            leftColumn.classList.remove('is-hovered');
        };
    }

    renderEntityCard(entityType, data, extension) {
        return renderEntityCardFn(this, cardConfigs, entityType, data, extension);
    }
}
