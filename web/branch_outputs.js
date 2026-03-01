import { app } from "/scripts/app.js";

const NODE_CLASSES = new Set(["DistributedBranch", "DistributedJoin"]);
const MIN_BRANCHES = 2;
const MAX_BRANCHES = 10;

function clampBranchCount(value) {
    const parsed = parseInt(value, 10);
    if (!Number.isFinite(parsed)) return MIN_BRANCHES;
    return Math.max(MIN_BRANCHES, Math.min(MAX_BRANCHES, parsed));
}

app.registerExtension({
    name: "Distributed.BranchOutputs",
    async nodeCreated(node) {
        if (!NODE_CLASSES.has(node.comfyClass)) return;

        const updateOutputs = () => {
            if (!node.widgets) return;
            const branchWidget = node.widgets.find((widget) => widget.name === "num_branches");
            if (!branchWidget) return;

            const totalOutputs = clampBranchCount(branchWidget.value);

            if (!node.outputs) node.outputs = [];

            while (node.outputs.length > totalOutputs) {
                node.removeOutput(node.outputs.length - 1);
            }

            while (node.outputs.length < totalOutputs) {
                const outputIndex = node.outputs.length + 1;
                node.addOutput(`branch_${outputIndex}`, "*");
            }

            for (let idx = 0; idx < node.outputs.length; idx++) {
                node.outputs[idx].name = `branch_${idx + 1}`;
                node.outputs[idx].type = "*";
            }

            if (typeof node.setDirtyCanvas === "function") {
                node.setDirtyCanvas(true, true);
            } else if (typeof node.setDirty === "function") {
                node.setDirty(true);
            }
        };

        setTimeout(updateOutputs, 200);

        const branchWidget = node.widgets?.find((widget) => widget.name === "num_branches");
        if (!branchWidget) return;

        const originalCallback = branchWidget.callback;
        branchWidget.callback = (value) => {
            updateOutputs();
            if (originalCallback) originalCallback.call(branchWidget, value);
        };

        const onInput = () => updateOutputs();
        if (branchWidget.inputEl) {
            branchWidget.inputEl.addEventListener("input", onInput);
        }

        const originalConfigure = node.configure;
        node.configure = function (data) {
            const result = originalConfigure ? originalConfigure.call(this, data) : undefined;
            updateOutputs();
            return result;
        };

        node._branchOutputsCleanup = () => {
            branchWidget.callback = originalCallback;
            if (branchWidget.inputEl) {
                branchWidget.inputEl.removeEventListener("input", onInput);
            }
        };
    },

    nodeBeforeRemove(node) {
        if (NODE_CLASSES.has(node.comfyClass) && node._branchOutputsCleanup) {
            node._branchOutputsCleanup();
        }
    },
});
