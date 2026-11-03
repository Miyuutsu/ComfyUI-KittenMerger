import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const SWITCH_NAMES = [
    "diagnostics",
    "summary",
    "changed_blocks",
    "changed_keys",
    "zeroed_keys",
    "all_keys",
];

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
        (originNode.inputs?.length === 1 &&
            originNode.outputs?.length === 1 &&
            (!originNode.widgets || originNode.widgets.length === 0))
    ) {
        return resolveUpstreamModelName(originNode, originNode.inputs[0], 0, visited);
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
        const baseInpIdx = (originNode.inputs || []).findIndex(
            (x) => x.name === "base_model" || x.name === "model"
        );
        if (baseInpIdx !== -1) {
            const baseClean = resolveUpstreamModelName(
                originNode,
                originNode.inputs[baseInpIdx],
                baseInpIdx,
                visited
            );
            if (baseClean && loraClean) {
                return `${baseClean}+${loraClean}`;
            }
        }
        if (loraClean) return loraClean;
    }

    const preferredWidgetNames = ["model_name", "ckpt_name", "unet_name", "diffusion_name"];
    if (Array.isArray(originNode.widgets)) {
        for (const wName of preferredWidgetNames) {
            const w = originNode.widgets.find((x) => x.name === wName);
            if (w && typeof w.value === "string" && w.value.trim()) {
                const cleaned = cleanModelName(w.value);
                if (cleaned && !cleaned.startsWith("No ")) return cleaned;
            }
        }
        for (const w of originNode.widgets) {
            if (
                typeof w.value === "string" &&
                /\.(safetensors|sft|ckpt|pt|pth|bin|gguf)$/i.test(w.value.trim())
            ) {
                const cleaned = cleanModelName(w.value);
                if (cleaned) return cleaned;
            }
        }
    }

    const modelInpIdx = (originNode.inputs || []).findIndex((x) => x.name === "model");
    if (modelInpIdx !== -1) {
        const upName = resolveUpstreamModelName(
            originNode,
            originNode.inputs[modelInpIdx],
            modelInpIdx,
            visited
        );
        if (upName) return upName;
    }

    return null;
}

function getUpstreamMergerOutputs(node) {
    const graph = node?.graph || app?.graph;
    if (!graph || !node.inputs) return null;
    const schemaIdx = node.inputs.findIndex((x) => x.name === "merge_schema");
    if (schemaIdx === -1) return null;
    const linkObj = findInputLinkObj(node, node.inputs[schemaIdx], schemaIdx);
    if (!linkObj) return null;
    const originNode = graph.getNodeById(linkObj.origin_id);
    if (!originNode || originNode.type !== "AnimaModelMerger") return null;

    const modelOuts = (originNode.outputs || []).filter((o) => o && o.type === "MODEL");
    if (modelOuts.some((o) => /^model_s\d+$/.test(o.name || ""))) {
        return modelOuts.map((o) => ({ name: o.name, label: o.label || o.name, type: "MODEL" }));
    }

    const st = originNode._mergerState;
    if (st && Array.isArray(st.stages) && st.stages.length > 0) {
        const res = [];
        const mCount = st.modelOutputCount || 1;
        const resolvedNames = st.resolvedModelNames || {};
        for (let i = 1; i <= mCount; i++) {
            const slotName = `model_${i}`;
            res.push({
                name: slotName,
                label: resolvedNames[`m${i}`] || slotName,
                type: "MODEL",
            });
        }
        for (let i = 0; i < st.stages.length; i++) {
            const sId = st.stages[i].id || `s${i + 1}`;
            const slotName = `model_${sId}`;
            const customStageName = (st.stages[i].name || "").trim();
            res.push({
                name: slotName,
                label: customStageName || slotName,
                type: "MODEL",
            });
        }
        return res;
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
                node.addInput(savedInp.name, savedInp.type || "MODEL");
            }
        }
    }

    if (Array.isArray(info.outputs)) {
        while (node.outputs.length < info.outputs.length) {
            const spec = info.outputs[node.outputs.length];
            node.addOutput(spec?.name || `model_${node.outputs.length}`, spec?.type || "MODEL");
        }
        if (node.outputs.length > info.outputs.length) {
            node.outputs.length = info.outputs.length;
        }
        for (let i = 0; i < info.outputs.length; i++) {
            if (info.outputs[i]?.name) {
                node.outputs[i].name = info.outputs[i].name;
                node.outputs[i].label = info.outputs[i].label || info.outputs[i].name;
                node.outputs[i].type = info.outputs[i].type || (i === 0 ? "STRING" : "MODEL");
            }
        }
    }
}

app.registerExtension({
    name: "KittenMerger.InspectorUI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ModelMergeInspector") {
            return;
        }

        const updateNodeDisplay = (node) => {
            if (!node._cachedSections || !node._displayWidget) {
                return;
            }
            const parts = [node._cachedSections["HEADER"]];
            for (const swName of SWITCH_NAMES) {
                const w = node.widgets?.find((x) => x.name === swName);
                if (w && w.value && node._cachedSections[swName]) {
                    parts.push(node._cachedSections[swName]);
                }
            }
            if (parts.length === 1) {
                parts.push("No switches turned on. Toggle at least one switch above.");
            }
            node._displayWidget.value = parts.join("\n\n");
            app.graph?.setDirtyCanvas(true, false);
        };

        const syncZeroReplaceWidgets = (node) => {
            const rzWidget = node.widgets?.find((x) => x.name === "replace_zeros");
            const modeWidget = node.widgets?.find((x) => x.name === "replace_zeros_mode");
            const fromWidget = node.widgets?.find((x) => x.name === "replace_zeros_from");
            const active = rzWidget && rzWidget.value !== "none";

            for (const w of [modeWidget, fromWidget]) {
                if (w) {
                    w.disabled = !active;
                }
            }
            app.graph?.setDirtyCanvas(true, false);
        };

        const syncDynamicModelSlots = (node, { allowShrink = true } = {}) => {
            if (node._isSyncingSlots) return false;
            node._isSyncingSlots = true;
            let didMutate = false;

            const canShrink =
                allowShrink &&
                !node._isConfiguring &&
                !app.configuringGraph &&
                !isCanvasConnecting();

            try {
                if (!node.inputs) node.inputs = [];
                if (!node.outputs) node.outputs = [];

                let highestConnectedInput = 0;
                let highestExistingInput = 0;

                node.inputs.forEach((inp, idx) => {
                    const m = /^model_(\d+)$/.exec(inp.name || "");
                    if (m) {
                        const k = parseInt(m[1], 10);
                        highestExistingInput = Math.max(highestExistingInput, k);
                        if (isInputSlotConnected(node, inp, idx)) {
                            highestConnectedInput = Math.max(highestConnectedInput, k);
                        }
                    }
                });

                const targetInputCount = Math.min(32, Math.max(1, highestConnectedInput + 1));
                const desiredInputCount = canShrink
                    ? targetInputCount
                    : Math.max(targetInputCount, highestExistingInput);

                const existingModelNames = new Set(
                    node.inputs
                        .filter((inp) => /^model_\d+$/.test(inp.name || ""))
                        .map((inp) => inp.name)
                );

                for (let i = 1; i <= desiredInputCount; i++) {
                    const name = `model_${i}`;
                    if (!existingModelNames.has(name)) {
                        node.addInput(name, "MODEL");
                        existingModelNames.add(name);
                        didMutate = true;
                    }
                }

                if (canShrink) {
                    for (let i = node.inputs.length - 1; i >= 0; i--) {
                        const inp = node.inputs[i];
                        const m = /^model_(\d+)$/.exec(inp?.name || "");
                        if (!m) continue;
                        const k = parseInt(m[1], 10);
                        if (k > desiredInputCount && !isInputSlotConnected(node, inp, i)) {
                            node.removeInput(i);
                            didMutate = true;
                        }
                    }
                }

                node.inputs.forEach((inp, idx) => {
                    if (!/^model_\d+$/.test(inp.name || "")) return;
                    const connected = isInputSlotConnected(node, inp, idx);
                    const resolved = connected ? resolveUpstreamModelName(node, inp, idx) : null;
                    const targetLabel = connected && resolved ? resolved : inp.name;
                    if (inp.label !== targetLabel) {
                        inp.label = targetLabel;
                        didMutate = true;
                    }
                });

                if (node._isConfiguring || app.configuringGraph) {
                    return didMutate;
                }

                const desiredOutputs = [{ name: "report", label: "report", type: "STRING" }];
                const bundledOuts = getUpstreamMergerOutputs(node);

                const directInputLabels = {};
                node.inputs.forEach((inp, idx) => {
                    const m = /^model_(\d+)$/.exec(inp.name || "");
                    if (m && isInputSlotConnected(node, inp, idx) && inp.label && inp.label !== inp.name) {
                        directInputLabels[parseInt(m[1], 10)] = inp.label;
                    }
                });

                if (bundledOuts && bundledOuts.length > 0) {
                    for (let i = 0; i < bundledOuts.length; i++) {
                        const bo = { ...bundledOuts[i] };
                        if (directInputLabels[i + 1]) {
                            bo.label = directInputLabels[i + 1];
                        }
                        desiredOutputs.push(bo);
                    }
                } else {
                    let highestOutputNeeded = Math.max(1, highestConnectedInput);
                    node.outputs.forEach((out, idx) => {
                        if (idx >= 1 && out.links && out.links.length > 0) {
                            highestOutputNeeded = Math.max(highestOutputNeeded, idx);
                        }
                    });
                    for (let i = 1; i <= highestOutputNeeded; i++) {
                        const slotName = `model_${i}`;
                        desiredOutputs.push({
                            name: slotName,
                            label: directInputLabels[i] || slotName,
                            type: "MODEL",
                        });
                    }
                }

                for (let idx = desiredOutputs.length; idx < node.outputs.length; idx++) {
                    const out = node.outputs[idx];
                    if (out && Array.isArray(out.links) && out.links.length > 0) {
                        desiredOutputs.push({
                            name: out.name || `model_${idx}`,
                            label: out.label || out.name || `model_${idx}`,
                            type: out.type || "MODEL",
                        });
                    }
                }

                while (node.outputs.length < desiredOutputs.length) {
                    const spec = desiredOutputs[node.outputs.length];
                    node.addOutput(spec.name, spec.type);
                    didMutate = true;
                }

                if (canShrink && node.outputs.length > desiredOutputs.length) {
                    node.outputs.length = desiredOutputs.length;
                    didMutate = true;
                }

                for (let i = 0; i < desiredOutputs.length; i++) {
                    const out = node.outputs[i];
                    const spec = desiredOutputs[i];
                    if (out.name !== spec.name || out.label !== spec.label) {
                        out.name = spec.name;
                        out.label = spec.label;
                        out.type = spec.type;
                        didMutate = true;
                    }
                }
            } finally {
                node._isSyncingSlots = false;
            }
            return didMutate;
        };

        const checkDynamicSlots = (node, options = {}) => {
            if (!node.inputs) return;
            if (syncDynamicModelSlots(node, options)) {
                node.setDirtyCanvas?.(true, true);
                app.graph?.setDirtyCanvas(true, true);
            }
        };

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            this._isConfiguring = true;
            ensureConfiguredSlotsExist(this, info);
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            syncDynamicModelSlots(this, { allowShrink: false });
            syncZeroReplaceWidgets(this);

            requestAnimationFrame(() => {
                this._isConfiguring = false;
                checkDynamicSlots(this, { allowShrink: true });
            });
            setTimeout(() => {
                this._isConfiguring = false;
                checkDynamicSlots(this, { allowShrink: true });
            }, 60);
            return r;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            if (!this._displayWidget) {
                const displayWidget = ComfyWidgets["STRING"](
                    this,
                    "report_display",
                    ["STRING", { multiline: true }],
                    app
                ).widget;

                if (displayWidget && displayWidget.inputEl) {
                    displayWidget.inputEl.readOnly = true;
                    displayWidget.inputEl.style.fontFamily = "monospace";
                    displayWidget.inputEl.style.fontSize = "11.5px";
                    displayWidget.inputEl.style.lineHeight = "1.35";
                    displayWidget.inputEl.style.whiteSpace = "pre";
                    displayWidget.inputEl.style.overflowX = "auto";
                }
                displayWidget.value = "Run workflow to populate inspector...";
                displayWidget.serializeValue = () => undefined;

                this._displayWidget = displayWidget;
            }

            this._cachedSections = null;

            for (const swName of SWITCH_NAMES) {
                const w = this.widgets?.find((x) => x.name === swName);
                if (w) {
                    const origCallback = w.callback;
                    w.callback = () => {
                        if (origCallback) {
                            origCallback.apply(w, arguments);
                        }
                        updateNodeDisplay(this);
                    };
                }
            }

            const rzWidget = this.widgets?.find((x) => x.name === "replace_zeros");
            if (rzWidget) {
                const origCallback = rzWidget.callback;
                rzWidget.callback = () => {
                    if (origCallback) {
                        origCallback.apply(rzWidget, arguments);
                    }
                    syncZeroReplaceWidgets(this);
                };
            }

            if (Array.isArray(this.outputs) && this.outputs.length > 2) {
                this.outputs.length = 2;
            }
            syncDynamicModelSlots(this, { allowShrink: false });
            syncZeroReplaceWidgets(this);
            this.setSize([620, 600]);
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            checkDynamicSlots(this);
            requestAnimationFrame(() => checkDynamicSlots(this));
            setTimeout(() => checkDynamicSlots(this), 25);
            return r;
        };

        const onDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function (ctx) {
            const now = performance.now();
            if (!this._lastSlotCheckTime || now - this._lastSlotCheckTime > 300) {
                this._lastSlotCheckTime = now;
                checkDynamicSlots(this);
            }
            if (onDrawBackground) {
                onDrawBackground.apply(this, arguments);
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }
            if (message && message.sections && message.sections[0]) {
                this._cachedSections = message.sections[0];
                updateNodeDisplay(this);
            } else if (message && message.report_text && message.report_text[0] && this._displayWidget) {
                this._displayWidget.value = message.report_text[0];
            }
        };
    },
});
