import { app } from "/scripts/app.js";

const BRANCH_NODE_CLASS = "DistributedBranch";
const BRANCH_COLLECTOR_NODE_CLASS = "DistributedBranchCollector";
const NODE_CLASSES = new Set([BRANCH_NODE_CLASS, BRANCH_COLLECTOR_NODE_CLASS]);
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

        const updateNodeIO = () => {
            if (!node.widgets) return;
            const branchWidget = node.widgets.find((widget) => widget.name === "num_branches");
            if (!branchWidget) return;

            const totalBranches = clampBranchCount(branchWidget.value);

            if (node.comfyClass === BRANCH_COLLECTOR_NODE_CLASS) {
                if (!node.inputs) node.inputs = [];

                while (node.inputs.length > totalBranches) {
                    node.removeInput(node.inputs.length - 1);
                }

                while (node.inputs.length < totalBranches) {
                    const inputIndex = node.inputs.length + 1;
                    node.addInput(`branch_${inputIndex}`, "*");
                }

                for (let idx = 0; idx < node.inputs.length; idx++) {
                    node.inputs[idx].name = `branch_${idx + 1}`;
                    node.inputs[idx].type = "*";
                }
            }

            if (!node.outputs) node.outputs = [];

            while (node.outputs.length > totalBranches) {
                node.removeOutput(node.outputs.length - 1);
            }

            while (node.outputs.length < totalBranches) {
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

        setTimeout(updateNodeIO, 200);

        const branchWidget = node.widgets?.find((widget) => widget.name === "num_branches");
        if (!branchWidget) return;

        const originalCallback = branchWidget.callback;
        branchWidget.callback = (value) => {
            updateNodeIO();
            if (originalCallback) originalCallback.call(branchWidget, value);
        };

        const onInput = () => updateNodeIO();
        if (branchWidget.inputEl) {
            branchWidget.inputEl.addEventListener("input", onInput);
        }

        const originalConfigure = node.configure;
        node.configure = function (data) {
            const result = originalConfigure ? originalConfigure.call(this, data) : undefined;
            updateNodeIO();
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
