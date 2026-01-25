import { app } from "/scripts/app.js";

// Configuration for each batch divider node type
const BATCH_DIVIDER_NODES = {
    "ImageBatchDivider": { outputPrefix: "batch_", outputType: "IMAGE" },
    "AudioBatchDivider": { outputPrefix: "audio_", outputType: "AUDIO" }
};

app.registerExtension({
    name: "Distributed.BatchDividers",
    async nodeCreated(node) {
        const config = BATCH_DIVIDER_NODES[node.comfyClass];
        if (!config) return;

        try {
            const updateOutputs = () => {
                if (!node.widgets) return;

                const divideByWidget = node.widgets.find(w => w.name === "divide_by");
                if (!divideByWidget) return;

                const divideBy = parseInt(divideByWidget.value, 10) || 1;
                const totalOutputs = divideBy;

                // Ensure outputs array exists
                if (!node.outputs) node.outputs = [];

                // Remove excess outputs
                while (node.outputs.length > totalOutputs) {
                    node.removeOutput(node.outputs.length - 1);
                }

                // Add missing outputs
                while (node.outputs.length < totalOutputs) {
                    const outputIndex = node.outputs.length + 1;
                    node.addOutput(`${config.outputPrefix}${outputIndex}`, config.outputType);
                }

                if (node.setDirty) node.setDirty(true);
            };

            // Initial update with delay to allow workflow loading
            setTimeout(updateOutputs, 200);

            // Find the widget and set up responsive handlers
            const divideByWidget = node.widgets.find(w => w.name === "divide_by");
            if (divideByWidget) {
                const originalCallback = divideByWidget.callback;
                divideByWidget.callback = (value) => {
                    updateOutputs();
                    if (originalCallback) originalCallback.call(divideByWidget, value);
                };

                if (divideByWidget.inputEl) {
                    divideByWidget.inputEl.addEventListener('input', updateOutputs);
                }

                const observer = new MutationObserver(updateOutputs);
                if (divideByWidget.element) {
                    observer.observe(divideByWidget.element, { attributes: true, childList: true, subtree: true });
                }

                node._batchDividerCleanup = () => {
                    observer.disconnect();
                    if (divideByWidget.inputEl) {
                        divideByWidget.inputEl.removeEventListener('input', updateOutputs);
                    }
                    divideByWidget.callback = originalCallback;
                };
            }

            const originalConfigure = node.configure;
            node.configure = function(data) {
                const result = originalConfigure ? originalConfigure.call(this, data) : undefined;
                updateOutputs();
                return result;
            };
        } catch (error) {
            console.error(`Error in ${node.comfyClass} extension:`, error);
        }
    },

    nodeBeforeRemove(node) {
        if (BATCH_DIVIDER_NODES[node.comfyClass] && node._batchDividerCleanup) {
            node._batchDividerCleanup();
        }
    }
});
