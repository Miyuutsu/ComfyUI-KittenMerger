import { app } from "../../scripts/app.js";

function cleanModelName(raw) {
    if (!raw || typeof raw !== "string") return "";
    let s = raw.trim().replace(/^\[[^\]]+\]\s*/, "");
    s = s.replace(/\\/g, "/").split("/").pop() || "";
    s = s.replace(/\.(safetensors|sft|ckpt|pt|pth|bin|gguf)$/i, "");
    return s.trim();
}

function getGraphLinkById(graph, linkId) {
    if (!graph || !graph.links || linkId == null) return null;
    if (typeof graph.links.get === "function") {
        return graph.links.get(linkId);
    }
    return graph.links[linkId] || null;
}

function findInputLinkObj(node, inp, slotIndex) {
    const graph = node?.graph || app?.graph;
    if (!graph || !inp) return null;
    if (inp.link != null) {
        const direct = getGraphLinkById(graph, inp.link);
        if (direct) return direct;
    }
    const links = graph.links;
    if (!links) return null;
    const iter = typeof links.values === "function" ? links.values() : Object.values(links);
    for (const l of iter) {
        if (l && l.target_id === node.id && l.target_slot === slotIndex) {
            return l;
        }
    }
    return null;
}

function isInputSlotConnected(node, inp, slotIndex) {
    if (!inp) return false;
    if (inp.link != null) return true;
    if (node?._isConfiguring || app?.configuringGraph) {
        return Boolean(findInputLinkObj(node, inp, slotIndex));
    }
    return false;
}

function isCanvasConnecting() {
    const canvas = app?.canvas;
    if (!canvas) return false;
    return Boolean(
        canvas.connecting_node ||
            canvas.connecting_input ||
            canvas.connecting_output ||
            canvas._connecting
    );
}

function resolveUpstreamModelName(node, inp, slotIndex, visited = new Set()) {
    const graph = node?.graph || app?.graph;
    if (!graph || !inp) return null;

    const linkObj = findInputLinkObj(node, inp, slotIndex);
    if (!linkObj) return null;

    const visitKey = `${linkObj.origin_id}:${linkObj.origin_slot}`;
    if (visited.has(visitKey)) return null;
    visited.add(visitKey);

    const originNode = graph.getNodeById(linkObj.origin_id);
    if (!originNode) return null;

    if (
        originNode.type === "Reroute" ||
        originNode.type === "AnimaDtypeConverter" ||
        (originNode.inputs?.length === 1 &&
            originNode.outputs?.length === 1 &&
            (!originNode.widgets || originNode.widgets.length === 0))
    ) {
        const targetSlot = originNode.type === "AnimaDtypeConverter" ? linkObj.origin_slot : 0;
        const targetInp = originNode.inputs?.[targetSlot];
        if (targetInp) {
            return resolveUpstreamModelName(originNode, targetInp, targetSlot, visited);
        }
    }

    if (originNode.type === "AnimaModelMerger" || originNode.type === "ModelMergeInspector") {
        const outSlot = originNode.outputs?.[linkObj.origin_slot];
        if (outSlot?.name) {
            const mPass = /^model_(\d+)$/.exec(outSlot.name);
            if (mPass) {
                const targetName = `model_${mPass[1]}`;
                const upIdx = (originNode.inputs || []).findIndex((x) => x.name === targetName);
                if (upIdx !== -1) {
                    return (
                        resolveUpstreamModelName(originNode, originNode.inputs[upIdx], upIdx, visited) ||
                        targetName
                    );
                }
            }
            const mStage = /^model_(s\d+)$/.exec(outSlot.name);
            if (mStage) {
                return outSlot.label && outSlot.label !== outSlot.name ? outSlot.label : mStage[1];
            }
        }
    }

    const loraWidget = originNode.widgets?.find((w) => w.name === "lora_name");
    if (loraWidget && loraWidget.value) {
        const loraClean = cleanModelName(String(loraWidget.value));
        if (loraClean) return loraClean;
    }

    const preferredWidgetNames = ["model_name", "ckpt_name", "unet_name", "diffusion_name", "vae_name", "clip_name"];
    if (Array.isArray(originNode.widgets)) {
        for (const wName of preferredWidgetNames) {
            const w = originNode.widgets.find((x) => x.name === wName);
            if (w && typeof w.value === "string" && w.value.trim()) {
                const cleaned = cleanModelName(w.value);
                if (cleaned && !cleaned.startsWith("No ")) return cleaned;
            }
        }
    }

    return null;
}

function ensureConfiguredSlotsExist(node, info) {
    if (!info) return;
    if (!node.inputs) node.inputs = [];
    if (!node.outputs) node.outputs = [];

    if (Array.isArray(info.inputs)) {
        for (const savedInp of info.inputs) {
            if (!savedInp || !savedInp.name) continue;
            const existingIdx = node.inputs.findIndex((x) => x.name === savedInp.name);
            if (existingIdx === -1) {
                node.addInput(savedInp.name, savedInp.type || "*");
            }
        }
    }

    if (Array.isArray(info.outputs)) {
        while (node.outputs.length < info.outputs.length) {
            const spec = info.outputs[node.outputs.length];
            node.addOutput(spec?.name || `output_${node.outputs.length + 1}`, spec?.type || "*");
        }
        if (node.outputs.length > info.outputs.length) {
            node.outputs.length = info.outputs.length;
        }
        for (let i = 0; i < info.outputs.length; i++) {
            if (info.outputs[i]?.name) {
                node.outputs[i].name = info.outputs[i].name;
                node.outputs[i].label = info.outputs[i].label || info.outputs[i].name;
                node.outputs[i].type = info.outputs[i].type || "*";
            }
        }
    }
}

app.registerExtension({
    name: "KittenMerger.SaverAndConverterUI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "AnimaDtypeConverter" && nodeData.name !== "AnimaModelSaver") {
            return;
        }

        const isConverter = nodeData.name === "AnimaDtypeConverter";

        const syncLoraWidgetsState = (node) => {
            if (isConverter) return;
            const fmtWidget = node.widgets?.find((w) => w.name === "lora_format");
            const rankWidget = node.widgets?.find((w) => w.name === "lora_rank");
            const lokrWidget = node.widgets?.find((w) => w.name === "lokr_factor");
            if (!fmtWidget) return;

            const isFullDiff = fmtWidget.value === "full_diff (lossless step)";
            const doraWidget = node.widgets?.find((w) => w.name === "dora_scale"); if (doraWidget) doraWidget.disabled = isFullDiff;
            const isLokr = fmtWidget.value === "lokr";
            if (rankWidget) rankWidget.disabled = isFullDiff;
            if (lokrWidget) lokrWidget.disabled = !isLokr;
            app.graph?.setDirtyCanvas(true, false);
        };

        const resolveUniversalSlotLabel = (node, inp, slotIndex) => {
            const linkObj = findInputLinkObj(node, inp, slotIndex);
            const graph = node?.graph || app?.graph;
            if (!linkObj || !graph) return inp.name;
            const originNode = graph.getNodeById(linkObj.origin_id);
            if (!originNode) return inp.name;
            const originOut = originNode.outputs?.[linkObj.origin_slot];
            const outType = originOut?.type && originOut.type !== "*" ? String(originOut.type) : "";
            const upName = resolveUpstreamModelName(node, inp, slotIndex);
            if (upName && outType) return `${outType}: ${upName}`;
            if (upName) return upName;
            if (outType && originOut?.label && originOut.label !== outType) {
                return `${outType}: ${originOut.label}`;
            }
            if (outType) return `${inp.name} (${outType})`;
            return originNode.title || inp.name;
        };

        const syncUniversalSlots = (node, { allowShrink = true } = {}) => {
            if (node._isSyncingUniversal) return false;
            node._isSyncingUniversal = true;
            let didMutate = false;

            const canShrink =
                allowShrink &&
                !node._isConfiguring &&
                !app.configuringGraph &&
                !isCanvasConnecting();

            try {
                if (!node.inputs) node.inputs = [];
                if (!node.outputs) node.outputs = [];

                let highestConnected = 0;
                let highestExisting = 0;

                node.inputs.forEach((inp, idx) => {
                    const m = /^input_(\d+)$/.exec(inp.name || "");
                    if (m) {
                        const k = parseInt(m[1], 10);
                        highestExisting = Math.max(highestExisting, k);
                        if (isInputSlotConnected(node, inp, idx)) {
                            highestConnected = Math.max(highestConnected, k);
                        }
                    }
                });

                const targetCount = Math.min(32, Math.max(1, highestConnected + 1));
                const desiredCount = canShrink ? targetCount : Math.max(targetCount, highestExisting);

                const existingNames = new Set(
                    node.inputs
                        .filter((inp) => /^input_\d+$/.test(inp.name || ""))
                        .map((inp) => inp.name)
                );

                for (let i = 1; i <= desiredCount; i++) {
                    const name = `input_${i}`;
                    if (!existingNames.has(name)) {
                        node.addInput(name, "*");
                        existingNames.add(name);
                        didMutate = true;
                    }
                }

                if (canShrink) {
                    for (let i = node.inputs.length - 1; i >= 0; i--) {
                        const inp = node.inputs[i];
                        const m = /^input_(\d+)$/.exec(inp?.name || "");
                        if (!m) continue;
                        const k = parseInt(m[1], 10);
                        if (k > desiredCount && !isInputSlotConnected(node, inp, i)) {
                            node.removeInput(i);
                            didMutate = true;
                        }
                    }
                }

                const slotLabels = {};
                node.inputs.forEach((inp, idx) => {
                    const m = /^input_(\d+)$/.exec(inp.name || "");
                    if (!m) return;
                    const k = parseInt(m[1], 10);
                    const connected = isInputSlotConnected(node, inp, idx);
                    const targetLabel = connected ? resolveUniversalSlotLabel(node, inp, idx) : inp.name;
                    slotLabels[k] = connected ? targetLabel : `output_${k}`;
                    if (inp.label !== targetLabel) {
                        inp.label = targetLabel;
                        didMutate = true;
                    }
                });

                if (isConverter && !node._isConfiguring && !app.configuringGraph) {
                    let highestOutNeeded = Math.max(1, highestConnected);
                    node.outputs.forEach((out, idx) => {
                        if (out?.links && out.links.length > 0) {
                            highestOutNeeded = Math.max(highestOutNeeded, idx + 1);
                        }
                    });

                    while (node.outputs.length < highestOutNeeded) {
                        const nextIdx = node.outputs.length + 1;
                        node.addOutput(`output_${nextIdx}`, "*");
                        didMutate = true;
                    }

                    if (canShrink && node.outputs.length > highestOutNeeded) {
                        node.outputs.length = highestOutNeeded;
                        didMutate = true;
                    }

                    for (let i = 0; i < node.outputs.length; i++) {
                        const out = node.outputs[i];
                        const k = i + 1;
                        const desiredName = `output_${k}`;
                        const desiredLabel = slotLabels[k] || desiredName;
                        if (out.name !== desiredName || out.label !== desiredLabel) {
                            out.name = desiredName;
                            out.label = desiredLabel;
                            didMutate = true;
                        }
                    }
                }
            } finally {
                node._isSyncingUniversal = false;
            }
            return didMutate;
        };

        const checkSlots = (node, options = {}) => {
            if (syncUniversalSlots(node, options)) {
                node.setDirtyCanvas?.(true, true);
                app.graph?.setDirtyCanvas(true, true);
            }
        };

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            this._isConfiguring = true;
            ensureConfiguredSlotsExist(this, info);
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            syncUniversalSlots(this, { allowShrink: false });
            syncLoraWidgetsState(this);
            requestAnimationFrame(() => {
                this._isConfiguring = false;
                checkSlots(this, { allowShrink: true });
            });
            setTimeout(() => {
                this._isConfiguring = false;
                checkSlots(this, { allowShrink: true });
            }, 60);
            return r;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (isConverter && Array.isArray(this.outputs) && this.outputs.length > 1) {
                this.outputs.length = 1;
            }
            const fmtWidget = this.widgets?.find((w) => w.name === "lora_format");
            if (fmtWidget) {
                const origCb = fmtWidget.callback;
                fmtWidget.callback = () => {
                    if (origCb) origCb.apply(fmtWidget, arguments);
                    syncLoraWidgetsState(this);
                };
            }
            syncUniversalSlots(this, { allowShrink: false });
            syncLoraWidgetsState(this);
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            checkSlots(this);
            requestAnimationFrame(() => checkSlots(this));
            setTimeout(() => checkSlots(this), 25);
            return r;
        };

        const onDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function () {
            const now = performance.now();
            if (!this._lastUnivSlotCheck || now - this._lastUnivSlotCheck > 300) {
                this._lastUnivSlotCheck = now;
                checkSlots(this);
            }
            if (onDrawBackground) {
                onDrawBackground.apply(this, arguments);
            }
        };
    },
});
