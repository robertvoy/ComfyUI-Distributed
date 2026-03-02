import { updateWorkerControls, toggleWorkerExpanded } from "../workerLifecycle.js";
import { isRemoteWorker } from "../workerSettings.js";

export function renderEntityCard(ui, cardConfigs, entityType, data, extension) {
    const config = cardConfigs[entityType] || {};
    const isPlaceholder = entityType === 'blueprint' || entityType === 'add';
    const isWorker = entityType === 'worker';
    const isMaster = entityType === 'master';
    const isRemote = isWorker && isRemoteWorker(extension, data);

    const cardOptions = {
        onClick: isPlaceholder ? data?.onClick : null,
    };
    if (isPlaceholder) {
        cardOptions.title = entityType === 'blueprint' ? "Click to add your first worker" : "Click to add a new worker";
    }
    const card = ui.createCard(entityType, cardOptions);
    if (isWorker && data?.id) {
        card.dataset.workerId = String(data.id);
    }

    const leftColumn = ui.createCheckboxOrIconColumn(config.checkbox, data, extension);
    card.appendChild(leftColumn);

    const rightColumn = ui.createCardColumn('content');
    rightColumn.classList.add("entity-card-content");

    const infoRow = ui.createInfoRow();
    if (config.infoRowPadding) {
        infoRow.style.padding = config.infoRowPadding;
    }
    if (config.minHeight === 'auto') {
        infoRow.style.minHeight = 'auto';
    } else if (config.minHeight) {
        infoRow.style.minHeight = config.minHeight;
    }
    if (config.expand) {
        infoRow.title = "Click to expand settings";
        infoRow.onclick = () => {
            if (isMaster) {
                const masterSettingsExpanded = !extension.masterSettingsExpanded;
                extension.masterSettingsExpanded = masterSettingsExpanded;
                const masterSettingsDiv = document.getElementById("master-settings");
                const arrow = infoRow.querySelector('.settings-arrow');
                if (masterSettingsExpanded) {
                    masterSettingsDiv.classList.add("expanded");
                    masterSettingsDiv.style.padding = "12px";
                    masterSettingsDiv.style.marginTop = "8px";
                    masterSettingsDiv.style.marginBottom = "8px";
                    arrow.style.transform = "rotate(90deg)";
                } else {
                    masterSettingsDiv.classList.remove("expanded");
                    masterSettingsDiv.style.padding = "0 12px";
                    masterSettingsDiv.style.marginTop = "0";
                    masterSettingsDiv.style.marginBottom = "0";
                    arrow.style.transform = "rotate(0deg)";
                }
            } else {
                toggleWorkerExpanded(extension, data.id);
            }
        };
    }

    const workerContent = ui.createWorkerContent();
    if (entityType === 'add') {
        workerContent.style.alignItems = "center";
    }

    const statusDot = ui.createStatusDotHelper(config.statusDot, data, extension);
    workerContent.appendChild(statusDot);

    const infoSpan = document.createElement("span");
    infoSpan.classList.add("dist-worker-info");
    infoSpan.innerHTML = config.infoText(data, extension);
    workerContent.appendChild(infoSpan);

    infoRow.appendChild(workerContent);

    let settingsArrow;
    if (config.expand) {
        const expandedId = config.settings?.expandedId || (isMaster ? 'master' : data?.id);
        settingsArrow = ui.createSettingsToggleHelper(expandedId, extension);
        if (isMaster && !extension.masterSettingsExpanded) {
            settingsArrow.style.transform = "rotate(0deg)";
        }
        infoRow.appendChild(settingsArrow);
    }

    rightColumn.appendChild(infoRow);

    if (config.hover === true) {
        rightColumn.classList.add("entity-card-content--hoverable");
        rightColumn.onmouseover = () => {
            rightColumn.classList.add("entity-card-content--hovered");
            if (settingsArrow) {
                settingsArrow.style.color = "var(--dist-settings-arrow-hover, #fff)";
            }
        };
        rightColumn.onmouseout = () => {
            rightColumn.classList.remove("entity-card-content--hovered");
            if (settingsArrow) {
                settingsArrow.style.color = "var(--dist-settings-arrow, #888)";
            }
        };
    }

    const controlsDiv = ui.createControlsSection(config.controls, data, extension, isRemote);
    if (controlsDiv) {
        rightColumn.appendChild(controlsDiv);
    }

    if (config.settings) {
        const settingsDiv = ui.createSettingsSection(config.settings, data, extension);
        rightColumn.appendChild(settingsDiv);
    }

    card.appendChild(rightColumn);

    if (config.hover === 'placeholder') {
        ui.addPlaceholderHover(card, leftColumn, entityType);
    }

    if (isWorker) {
        updateWorkerControls(extension, data.id);
    }

    return card;
}
