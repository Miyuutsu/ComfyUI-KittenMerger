import { app } from "../../scripts/app.js";

const SUB_LAYERS = [
    "adaln_modulation_self_attn",
    "adaln_modulation_cross_attn",
    "adaln_modulation_mlp",
    "self_attn",
    "cross_attn",
    "mlp",
];

const DEFAULT_SUB_WEIGHTS = {
    adaln_modulation_self_attn: ["1.weight", "1.bias", "2.weight", "2.bias"],
    adaln_modulation_cross_attn: ["1.weight", "1.bias", "2.weight", "2.bias"],
    adaln_modulation_mlp: ["1.weight", "1.bias", "2.weight", "2.bias"],
    self_attn: [
        "q_proj.weight",
        "q_norm.weight",
        "k_proj.weight",
        "k_norm.weight",
        "v_proj.weight",
        "output_proj.weight",
    ],
    cross_attn: [
        "q_proj.weight",
        "q_norm.weight",
        "k_proj.weight",
        "k_norm.weight",
        "v_proj.weight",
        "output_proj.weight",
    ],
    mlp: ["layer1.weight", "layer2.weight"],
};

const OP_MODES = [
    {
        code: "ws",
        short: "ws",
        tip: "Weighted Sum: w * A + (1 - w) * B  [1.0 = Model A, 0.0 = Model B]",
    },
    {
        code: "add",
        short: "+",
        tip: "Add: A + w * B  [adds scaled Model B to Model A]",
    },
    {
        code: "sub",
        short: "-",
        tip: "Subtract: A - w * B  [subtracts scaled Model B from Model A]",
    },
    {
        code: "add_diff",
        short: "ad",
        tip: "Add Difference (3-Model): A + w * (B - C)",
    },
    {
        code: "cosine_a",
        short: "ca",
        tip: "Cosine A: favors Model A's cosine-dissimilar features from B (uses B - C if Model C is set), scaled by (1 - w)",
    },
    {
        code: "cosine_b",
        short: "cb",
        tip: "Cosine B: favors Model A's cosine-similar features from B (uses B - C if Model C is set), scaled by (1 - w)",
    },
];

const THREE_MODEL_OPS = new Set(["add_diff", "cosine_a", "cosine_b"]);

let stylesInjected = false;
function injectMergerStyles() {
    if (stylesInjected) return;
    stylesInjected = true;
    const style = document.createElement("style");
    style.textContent = `
        .kitten-merger-num::-webkit-outer-spin-button,
        .kitten-merger-num::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        .kitten-merger-num {
            -moz-appearance: textfield;
            appearance: textfield;
        }
        .kitten-merger-cb {
            appearance: auto;
            -webkit-appearance: checkbox;
            width: 12px;
            height: 12px;
            margin: 0;
            cursor: pointer;
            accent-color: #3498db;
            flex-shrink: 0;
        }
    `;
    document.head.appendChild(style);
}

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

function buildLegacyBasicKeyOrder(blockCount = 40) {
    return [
        "pos_embedder.",
        "x_embedder.",
        "t_embedder.",
        "t_embedding_norm.",
        ...Array.from({ length: blockCount }, (_, i) => `blocks.${i}.`),
        "final_layer.",
    ];
}

function buildLegacyGranularKeyOrder(blockCount = 40) {
    const keys = [
        "llm_adapter.embed.",
        "llm_adapter.norm.",
        "llm_adapter.out_proj.",
        ...Array.from({ length: 6 }, (_, i) => `llm_adapter.blocks.${i}.`),
    ];
    for (let i = 0; i < blockCount; i++) {
        for (const sub of SUB_LAYERS) {
            keys.push(`blocks.${i}.${sub}.`);
        }
    }
    return keys;
}

function mapWidgetArrayToOverrides(values, explicitType = null) {
    const nums = values.map((v) => parseFloat(v)).filter((v) => !Number.isNaN(v));
    let keyOrder = null;

    if (explicitType === "ModelMergeAnima2_9B" || nums.length === 45) {
        keyOrder = buildLegacyBasicKeyOrder(40);
    } else if (nums.length === 33) {
        keyOrder = buildLegacyBasicKeyOrder(28);
    } else if (explicitType === "ModelMergeAnima2_9B_Granular" || nums.length === 249) {
        keyOrder = buildLegacyGranularKeyOrder(40);
    } else if (nums.length === 177) {
        keyOrder = buildLegacyGranularKeyOrder(28);
    } else if (nums.length === 40 || nums.length === 28) {
        keyOrder = Array.from({ length: nums.length }, (_, i) => `blocks.${i}.`);
    }

    if (!keyOrder) return null;
    const out = {};
    const count = Math.min(keyOrder.length, nums.length);
    for (let i = 0; i < count; i++) {
        out[keyOrder[i]] = Number(nums[i].toFixed(4));
    }
    return out;
}

function parseImportPayload(rawText) {
    const trimmed = (rawText || "").trim();
    if (!trimmed) {
        return { kind: "empty", message: "Paste JSON/values or load a file first." };
    }

    try {
        const data = JSON.parse(trimmed);

        if (data && Array.isArray(data.nodes)) {
            const legacyBasicNodes = [];
            const legacyGranularNodes = [];
            const animaMergerNodes = [];

            for (const n of data.nodes) {
                if (!n || typeof n !== "object") continue;
                if (n.type === "ModelMergeAnima2_9B" && Array.isArray(n.widgets_values)) {
                    const ov = mapWidgetArrayToOverrides(n.widgets_values, "ModelMergeAnima2_9B");
                    if (ov) {
                        legacyBasicNodes.push({
                            label: `Node #${n.id} (${n.title || n.type}) [45 block/embedder weights]`,
                            overrides: ov,
                        });
                    }
                } else if (n.type === "ModelMergeAnima2_9B_Granular" && Array.isArray(n.widgets_values)) {
                    const ov = mapWidgetArrayToOverrides(n.widgets_values, "ModelMergeAnima2_9B_Granular");
                    if (ov) {
                        legacyGranularNodes.push({
                            label: `Node #${n.id} (${n.title || n.type}) [249 granular sub-layer weights]`,
                            overrides: ov,
                        });
                    }
                } else if (n.type === "AnimaModelMerger" && Array.isArray(n.widgets_values)) {
                    for (const wv of n.widgets_values) {
                        if (typeof wv === "string" && wv.trim().startsWith("{")) {
                            try {
                                const parsedRecipe = JSON.parse(wv);
                                if (parsedRecipe && Array.isArray(parsedRecipe.stages)) {
                                    animaMergerNodes.push({
                                        label: `Node #${n.id} (${n.title || n.type}) [${parsedRecipe.stages.length} stage(s)]`,
                                        recipe: parsedRecipe,
                                    });
                                }
                            } catch (e) {}
                        }
                    }
                }
            }

            const candidates = [];
            if (legacyBasicNodes.length > 0 && legacyGranularNodes.length > 0) {
                const combined = {
                    ...legacyBasicNodes[0].overrides,
                    ...legacyGranularNodes[0].overrides,
                };
                candidates.push({
                    label: `Combined: ${legacyBasicNodes[0].label} + ${legacyGranularNodes[0].label}`,
                    kind: "overrides",
                    overrides: combined,
                });
            }
            for (const b of legacyBasicNodes) {
                candidates.push({ label: b.label, kind: "overrides", overrides: b.overrides });
            }
            for (const g of legacyGranularNodes) {
                candidates.push({ label: g.label, kind: "overrides", overrides: g.overrides });
            }
            for (const m of animaMergerNodes) {
                candidates.push({ label: m.label, kind: "recipe", recipe: m.recipe });
            }

            if (candidates.length > 0) {
                return { kind: "workflow_candidates", candidates };
            }
            return {
                kind: "error",
                message: "Workflow JSON loaded, but no ModelMergeAnima2_9B, ModelMergeAnima2_9B_Granular, or AnimaModelMerger nodes were found.",
            };
        }

        if (data && typeof data === "object" && Array.isArray(data.widgets_values)) {
            const ov = mapWidgetArrayToOverrides(data.widgets_values, data.type);
            if (ov) {
                return {
                    kind: "overrides",
                    overrides: ov,
                    description: `Detected ${data.type || "legacy node"} (${Object.keys(ov).length} values).`,
                };
            }
        }

        if (Array.isArray(data)) {
            const ov = mapWidgetArrayToOverrides(data);
            if (ov) {
                return {
                    kind: "overrides",
                    overrides: ov,
                    description: `Detected widget array (${Object.keys(ov).length} values).`,
                };
            }
            return {
                kind: "error",
                message: `Unsupported array length (${data.length}). Expected 45 (2.9B Blocks), 249 (2.9B Granular), 40, 33, or 177.`,
            };
        }

        if (data && typeof data === "object" && Array.isArray(data.stages)) {
            return {
                kind: "recipe",
                recipe: data,
                description: `Detected AnimaModelMerger Recipe (${data.stages.length} stage(s)).`,
            };
        }

        if (data && typeof data === "object" && data.overrides && typeof data.overrides === "object") {
            return {
                kind: "stage",
                stage: data,
                description: `Detected Stage '${data.id || "stage"}' (${Object.keys(data.overrides).length} overrides).`,
            };
        }

        if (data && typeof data === "object") {
            const ov = {};
            for (const [k, v] of Object.entries(data)) {
                const num = parseFloat(v);
                if (!Number.isNaN(num)) {
                    const normKey = k.endsWith(".") || k.endsWith(".weight") || k.endsWith(".bias") ? k : `${k}.`;
                    ov[normKey] = Number(num.toFixed(4));
                }
            }
            if (Object.keys(ov).length > 0) {
                return {
                    kind: "overrides",
                    overrides: ov,
                    description: `Detected flat key-value map (${Object.keys(ov).length} keys).`,
                };
            }
        }
    } catch (e) {}

    const kvOverrides = {};
    const lines = trimmed.split(/\r?\n/);
    for (const line of lines) {
        const m = /^\s*["']?([a-zA-Z0-9_.\-]+)["']?\s*[:=]\s*([-+]?\d*\.?\d+)\s*,?\s*$/.exec(line);
        if (m) {
            const rawK = m[1];
            const val = parseFloat(m[2]);
            if (!Number.isNaN(val)) {
                const normKey =
                    rawK.endsWith(".") || rawK.endsWith(".weight") || rawK.endsWith(".bias")
                        ? rawK
                        : `${rawK}.`;
                kvOverrides[normKey] = Number(val.toFixed(4));
            }
        }
    }
    if (Object.keys(kvOverrides).length > 0) {
        return {
            kind: "overrides",
            overrides: kvOverrides,
            description: `Parsed ${Object.keys(kvOverrides).length} key-value line(s).`,
        };
    }

    const rawNums = trimmed
        .split(/[\s,]+/)
        .map((x) => parseFloat(x))
        .filter((x) => !Number.isNaN(x));
    const numOv = mapWidgetArrayToOverrides(rawNums);
    if (numOv) {
        return {
            kind: "overrides",
            overrides: numOv,
            description: `Parsed ${Object.keys(numOv).length} sequential numbers.`,
        };
    }

    return {
        kind: "error",
        message: "Could not parse input. Paste a workflow JSON, stage/recipe JSON, key:value pairs, or 45 / 249 float values.",
    };
}

function normalizeViewMode(mode) {
    if (mode === "both") return "sub_layers";
    if (mode === "advanced") return "layers";
    if (["basic", "blocks", "layers", "sub_layers"].includes(mode)) return mode;
    return "sub_layers";
}

function normalizeStageObj(st, idx, fallbackViewMode = "sub_layers") {
    const model_a = st.model_a || st.base || "m1";
    const model_b = st.model_b || st.donor || "m2";
    const model_c = st.model_c || "none";
    const global_op = OP_MODES.some((m) => m.code === st.global_op) ? st.global_op : "ws";
    const global_ratio = typeof st.global_ratio === "number" ? st.global_ratio : 0.5;
    const fix_zeros = Boolean(st.fix_zeros);
    const enabled = st.enabled !== undefined ? Boolean(st.enabled) : true;
    const name = typeof st.name === "string" ? st.name : "";
    const view_mode = normalizeViewMode(st.view_mode || st.viewMode || fallbackViewMode);
    return {
        id: st.id || `s${idx + 1}`,
        name,
        enabled,
        view_mode,
        model_a,
        model_b,
        model_c,
        global_op,
        global_ratio,
        fix_zeros,
        overrides: st.overrides && typeof st.overrides === "object" ? { ...st.overrides } : {},
        op_overrides: st.op_overrides && typeof st.op_overrides === "object" ? { ...st.op_overrides } : {},
        auto_zero_keys: Array.isArray(st.auto_zero_keys) ? [...st.auto_zero_keys] : [],
    };
}

function stageUsesThreeModels(st) {
    if (THREE_MODEL_OPS.has(st.global_op)) return true;
    const opOv = st.op_overrides || {};
    return Object.values(opOv).some((v) => THREE_MODEL_OPS.has(v));
}

function formatStageTabLabel(st) {
    const a = st.model_a || "m1";
    const b = st.model_b || "m2";
    const c = st.model_c && st.model_c !== "none" ? st.model_c : null;
    const op = st.global_op || "ws";
    const zTag = st.fix_zeros ? "[FZ]" : "";
    const offTag = st.enabled === false ? "[OFF] " : "";
    const customName = (st.name || "").trim();
    const head = customName ? `${customName} (${st.id})` : st.id;
    if (op === "add") return `${offTag}${head}: ${a}+${b}(+)${zTag}`;
    if (op === "sub") return `${offTag}${head}: ${a}-${b}${zTag}`;
    if (op === "add_diff") return `${offTag}${head}: ${a}+(${b}-${c || "?"})${zTag}`;
    if (op === "cosine_a") return `${offTag}${head}: ${a}~${b}(ca)${zTag}`;
    if (op === "cosine_b") return `${offTag}${head}: ${a}~${b}(cb)${zTag}`;
    return `${offTag}${head}: ${a}+${b}${zTag}`;
}

function buildTreeStructure(blockCount = 40, customLeafKeys = null) {
    const groups = [
        {
            id: "embedders.",
            label: "Embedders (pos / x / t / norm)",
            children: [
                { id: "pos_embedder.", label: "pos_embedder.", children: [] },
                { id: "x_embedder.", label: "x_embedder.", children: [] },
                { id: "t_embedder.", label: "t_embedder.", children: [] },
                { id: "t_embedding_norm.", label: "t_embedding_norm.", children: [] },
            ],
        },
        {
            id: "llm_adapter.",
            label: "LLM Adapter (embed / norm / 0..5)",
            children: [
                { id: "llm_adapter.embed.", label: "llm_adapter.embed.", children: [] },
                { id: "llm_adapter.norm.", label: "llm_adapter.norm.", children: [] },
                { id: "llm_adapter.out_proj.", label: "llm_adapter.out_proj.", children: [] },
                ...Array.from({ length: 6 }, (_, i) => ({
                    id: `llm_adapter.blocks.${i}.`,
                    label: `llm_adapter.blocks.${i}.`,
                    children: [],
                })),
            ],
        },
    ];

    for (let b = 0; b < blockCount; b++) {
        const bPrefix = `blocks.${b}.`;
        const subItems = SUB_LAYERS.map((sub) => {
            const subPrefix = `${bPrefix}${sub}.`;
            const leaves = (DEFAULT_SUB_WEIGHTS[sub] || []).map((w) => ({
                id: `${subPrefix}${w}`,
                label: w,
            }));
            return {
                id: subPrefix,
                label: `${sub}.`,
                children: leaves,
            };
        });
        groups.push({
            id: bPrefix,
            label: `blocks.${b}.`,
            children: subItems,
        });
    }

    groups.push({
        id: "final_layer.",
        label: "Final Layer (adaln / linear)",
        children: [
            { id: "final_layer.adaln_modulation.", label: "final_layer.adaln_modulation.", children: [] },
            { id: "final_layer.linear.", label: "final_layer.linear.", children: [] },
        ],
    });

    if (Array.isArray(customLeafKeys) && customLeafKeys.length > 0) {
        const subMap = new Map();
        for (const g of groups) {
            for (const sub of g.children) {
                sub.children = [];
                subMap.set(sub.id, sub);
            }
        }
        for (const k of customLeafKeys) {
            for (const [subPrefix, subObj] of subMap.entries()) {
                if (k.startsWith(subPrefix)) {
                    const shortLabel = k.slice(subPrefix.length);
                    if (shortLabel) {
                        subObj.children.push({ id: k, label: shortLabel });
                    }
                    break;
                }
            }
        }
    }

    return groups;
}

function triggerJsonDownload(filename, text) {
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
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
    name: "KittenMerger.AnimaMergerUI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "AnimaModelMerger") {
            return;
        }

        injectMergerStyles();

        const ensureAdequateNodeHeight = (node) => {
            if (!node || !Array.isArray(node.size)) return;
            const slotCount = Math.max(node.inputs?.length || 1, node.outputs?.length || 1);
            const activeStage = node._mergerState?.stages?.[node._mergerState?.activeStageIdx || 0];
            const isBasic = (activeStage?.view_mode || "sub_layers") === "basic";
            const minHeight = isBasic
                ? Math.max(360, 260 + slotCount * 22)
                : Math.max(660, 480 + slotCount * 22);
            if (node.size[1] < minHeight) {
                node.setSize([Math.max(node.size[0], 680), minHeight]);
            }
        };

        const updateInputSlotLabelsAndNames = (node) => {
            if (!node.inputs) return false;
            let changed = false;
            const resolvedMap = {};
            const backendNames = node._mergerState?.backendModelNames || {};

            node.inputs.forEach((inp, idx) => {
                const m = /^model_(\d+)$/.exec(inp.name || "");
                if (!m) return;
                const mId = `m${m[1]}`;
                const connected = isInputSlotConnected(node, inp, idx);
                let resolved = null;
                if (connected) {
                    resolved = resolveUpstreamModelName(node, inp, idx) || backendNames[mId] || null;
                }
                if (resolved) {
                    resolvedMap[mId] = resolved;
                }
                const targetLabel = connected && resolved ? resolved : inp.name;
                if (inp.label !== targetLabel) {
                    inp.label = targetLabel;
                    changed = true;
                }
            });

            if (node._mergerState) {
                const prevJson = JSON.stringify(node._mergerState.resolvedModelNames || {});
                const nextJson = JSON.stringify(resolvedMap);
                if (prevJson !== nextJson) {
                    node._mergerState.resolvedModelNames = resolvedMap;
                    changed = true;
                }
            }
            return changed;
        };

        const computeDesiredModelOutputCount = (node, canShrink = true) => {
            let highestConnected = 1;
            (node.inputs || []).forEach((inp, idx) => {
                const m = /^model_(\d+)$/.exec(inp.name || "");
                if (m && isInputSlotConnected(node, inp, idx)) {
                    highestConnected = Math.max(highestConnected, parseInt(m[1], 10));
                }
            });

            (node.outputs || []).forEach((out) => {
                const m = /^model_(\d+)$/.exec(out.name || "");
                if (m) {
                    const k = parseInt(m[1], 10);
                    if (!canShrink || (out.links && out.links.length > 0)) {
                        highestConnected = Math.max(highestConnected, k);
                    }
                }
            });
            return highestConnected;
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

                if (updateInputSlotLabelsAndNames(node)) {
                    didMutate = true;
                }

                if (node._isConfiguring || app.configuringGraph) {
                    return didMutate;
                }

                const highestModelCount = computeDesiredModelOutputCount(node, canShrink);
                if (node._mergerState) {
                    node._mergerState.modelOutputCount = highestModelCount;
                }

                const stageList = node._mergerState?.stages || [{ id: "s1" }];
                const resolvedNames = node._mergerState?.resolvedModelNames || {};
                const desiredOutputs = [{ name: "merge_schema", label: "merge_schema", type: "STRING" }];
                for (let i = 1; i <= highestModelCount; i++) {
                    const slotName = `model_${i}`;
                    const customModelLabel = resolvedNames[`m${i}`];
                    desiredOutputs.push({
                        name: slotName,
                        label: customModelLabel || slotName,
                        type: "MODEL",
                    });
                }
                for (let i = 0; i < stageList.length; i++) {
                    const sId = stageList[i].id || `s${i + 1}`;
                    const slotName = `model_${sId}`;
                    const customStageName = (stageList[i].name || "").trim();
                    desiredOutputs.push({
                        name: slotName,
                        label: customStageName || slotName,
                        type: "MODEL",
                    });
                }

                const structureMatches =
                    node.outputs.length === desiredOutputs.length &&
                    node.outputs.every(
                        (out, idx) =>
                            out.name === desiredOutputs[idx].name && out.type === desiredOutputs[idx].type
                    );

                if (!structureMatches) {
                    didMutate = true;
                    const desiredNameSet = new Set(desiredOutputs.map((d) => d.name));
                    const savedLinksByName = new Map();

                    for (let i = node.outputs.length - 1; i >= 0; i--) {
                        const out = node.outputs[i];
                        if (!out) continue;
                        if (desiredNameSet.has(out.name)) {
                            if (Array.isArray(out.links) && out.links.length > 0) {
                                savedLinksByName.set(out.name, [...out.links]);
                            }
                        } else if (canShrink && Array.isArray(out.links) && out.links.length > 0) {
                            node.disconnectOutput(i);
                        }
                    }

                    while (node.outputs.length < desiredOutputs.length) {
                        const spec = desiredOutputs[node.outputs.length];
                        node.addOutput(spec.name, spec.type);
                    }
                    if (node.outputs.length > desiredOutputs.length) {
                        node.outputs.length = desiredOutputs.length;
                    }

                    const graph = node.graph || app.graph;
                    for (let idx = 0; idx < desiredOutputs.length; idx++) {
                        const spec = desiredOutputs[idx];
                        const out = node.outputs[idx];
                        out.name = spec.name;
                        out.label = spec.label;
                        out.type = spec.type;
                        const restoredLinks = savedLinksByName.get(spec.name) || [];
                        out.links = restoredLinks;
                        for (const linkId of restoredLinks) {
                            const linkObj = getGraphLinkById(graph, linkId);
                            if (linkObj) {
                                linkObj.origin_slot = idx;
                            }
                        }
                    }
                } else {
                    for (let idx = 0; idx < desiredOutputs.length; idx++) {
                        const spec = desiredOutputs[idx];
                        const out = node.outputs[idx];
                        if (out.label !== spec.label) {
                            out.label = spec.label;
                            didMutate = true;
                        }
                    }
                }

                if (didMutate) {
                    ensureAdequateNodeHeight(node);
                }
            } finally {
                node._isSyncingSlots = false;
            }
            return didMutate;
        };

        const checkDynamicSlots = (node, options = {}) => {
            if (!node.inputs) return;
            const mutated = syncDynamicModelSlots(node, options);
            const connSig = (node.inputs || [])
                .filter((inp) => /^model_\d+$/.test(inp.name || ""))
                .map((inp, idx) => `${inp.name}:${isInputSlotConnected(node, inp, idx) ? "1" : "0"}:${inp.label || ""}`)
                .join("|");

            if (mutated || node._lastConnSig !== connSig) {
                node._lastConnSig = connSig;
                if (node._saveStateToWidget) {
                    node._saveStateToWidget();
                }
                if (node._renderMergerUI) {
                    node._renderMergerUI();
                }
                node.setDirtyCanvas?.(true, true);
                app.graph?.setDirtyCanvas(true, true);
            }
        };

        const getConnectedModelIds = (node) => {
            const ids = [];
            (node.inputs || []).forEach((inp, idx) => {
                const m = /^model_(\d+)$/.exec(inp.name || "");
                if (m && isInputSlotConnected(node, inp, idx)) {
                    ids.push(`m${m[1]}`);
                }
            });
            if (ids.length === 0) ids.push("m1", "m2");
            else if (ids.length === 1) ids.push("m2");
            return ids;
        };

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            this._isConfiguring = true;
            ensureConfiguredSlotsExist(this, info);
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            if (this._loadStateFromWidget) {
                this._loadStateFromWidget();
            }
            syncDynamicModelSlots(this, { allowShrink: false });
            if (this._renderMergerUI) {
                this._renderMergerUI();
            }

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

            const recipeWidget = this.widgets?.find((w) => w.name === "recipe_json");
            if (recipeWidget) {
                recipeWidget.type = "hidden";
                recipeWidget.computeSize = () => [0, -4];
            }

            this._mergerState = {
                viewMode: "sub_layers",
                activeStageIdx: 0,
                modelOutputCount: 1,
                blockCount: 40,
                customLeafKeys: null,
                resolvedModelNames: {},
                backendModelNames: {},
                stageZeroFixes: {},
                expandedLvl1: new Set(),
                expandedLvl2: new Set(),
                selectedRows: new Set(),
                lastClickedRowId: null,
                showImpExpModal: false,
                stages: [
                    normalizeStageObj(
                        {
                            id: "s1",
                            model_a: "m1",
                            model_b: "m2",
                            model_c: "none",
                            global_op: "ws",
                            global_ratio: 0.5,
                            fix_zeros: false,
                            overrides: {},
                            op_overrides: {},
                        },
                        0
                    ),
                ],
            };

            const loadStateFromWidget = () => {
                if (!recipeWidget || !recipeWidget.value) return;
                try {
                    const parsed = JSON.parse(recipeWidget.value);
                    if (!parsed || typeof parsed !== "object") return;
                    const fallbackMode = parsed.viewMode ? normalizeViewMode(parsed.viewMode) : "sub_layers";
                    this._mergerState.viewMode = fallbackMode;
                    if (Array.isArray(parsed.stages) && parsed.stages.length > 0) {
                        this._mergerState.stages = parsed.stages.map((s, i) =>
                            normalizeStageObj(s, i, fallbackMode)
                        );
                    }
                    if (typeof parsed.activeStageIdx === "number") {
                        this._mergerState.activeStageIdx = Math.max(
                            0,
                            Math.min(parsed.activeStageIdx, this._mergerState.stages.length - 1)
                        );
                    }
                    if (typeof parsed.modelOutputCount === "number" && parsed.modelOutputCount >= 1) {
                        this._mergerState.modelOutputCount = parsed.modelOutputCount;
                    }
                    if (Array.isArray(parsed.expandedLvl1)) {
                        this._mergerState.expandedLvl1 = new Set(parsed.expandedLvl1);
                    }
                    if (Array.isArray(parsed.expandedLvl2)) {
                        this._mergerState.expandedLvl2 = new Set(parsed.expandedLvl2);
                    }
                } catch (e) {}
            };

            const getConnectedStageOutputIds = () => {
                const connectedStages = new Set();
                const graph = this.graph || app?.graph;
                (this.outputs || []).forEach((out) => {
                    const m = /^model_(s\d+)$/.exec(out.name || "");
                    if (m && Array.isArray(out.links) && out.links.length > 0) {
                        connectedStages.add(m[1]);
                    }
                });
                const schemaOut = (this.outputs || [])[0];
                if (graph && schemaOut && Array.isArray(schemaOut.links)) {
                    for (const linkId of schemaOut.links) {
                        const lObj = getGraphLinkById(graph, linkId);
                        const targetNode = lObj ? graph.getNodeById(lObj.target_id) : null;
                        if (targetNode && targetNode.type === "ModelMergeInspector") {
                            (targetNode.outputs || []).forEach((tOut) => {
                                const m = /^model_(s\d+)$/.exec(tOut.name || "");
                                if (m && Array.isArray(tOut.links) && tOut.links.length > 0) {
                                    connectedStages.add(m[1]);
                                }
                            });
                        }
                    }
                }
                return Array.from(connectedStages);
            };

            const saveStateToWidget = () => {
                if (!recipeWidget) return;
                recipeWidget.value = JSON.stringify({
                    viewMode: this._mergerState.viewMode,
                    activeStageIdx: this._mergerState.activeStageIdx,
                    modelOutputCount: this._mergerState.modelOutputCount || 1,
                    connectedStageIds: getConnectedStageOutputIds(),
                    expandedLvl1: Array.from(this._mergerState.expandedLvl1),
                    expandedLvl2: Array.from(this._mergerState.expandedLvl2),
                    stages: this._mergerState.stages,
                });
            };

            this._loadStateFromWidget = loadStateFromWidget;
            this._saveStateToWidget = saveStateToWidget;

            // Trim initial 65 Python return outputs to 3 without calling removeOutput
            if (Array.isArray(this.outputs) && this.outputs.length > 3) {
                this.outputs.length = 3;
                this.outputs[0].name = "merge_schema";
                this.outputs[0].label = "merge_schema";
                this.outputs[1].name = "model_1";
                this.outputs[1].label = "model_1";
                this.outputs[2].name = "model_s1";
                this.outputs[2].label = "model_s1";
            }

            loadStateFromWidget();
            syncDynamicModelSlots(this, { allowShrink: false });
            saveStateToWidget();

            const container = document.createElement("div");
            container.style.cssText = `
                position: relative;
                display: flex;
                flex-direction: column;
                width: 100%;
                height: 100%;
                background: #1a1a1a;
                color: #e0e0e0;
                font-family: monospace;
                font-size: 11.5px;
                border: 1px solid #333;
                border-radius: 4px;
                overflow: hidden;
                box-sizing: border-box;
            `;

            const topBar = document.createElement("div");
            topBar.style.cssText = `
                padding: 6px 8px;
                background: #242424;
                border-bottom: 1px solid #383838;
                display: flex;
                flex-direction: column;
                gap: 4px;
                flex-shrink: 0;
            `;

            const treeScroll = document.createElement("div");
            treeScroll.style.cssText = `
                flex: 1;
                overflow-y: auto;
                padding: 4px 6px;
                display: flex;
                flex-direction: column;
                gap: 2px;
            `;

            const modalOverlay = document.createElement("div");
            modalOverlay.style.cssText = `
                position: absolute;
                inset: 0;
                background: rgba(15, 15, 15, 0.96);
                z-index: 20;
                display: none;
                flex-direction: column;
                padding: 10px;
                gap: 8px;
                box-sizing: border-box;
                overflow-y: auto;
            `;

            container.appendChild(topBar);
            container.appendChild(treeScroll);
            container.appendChild(modalOverlay);

            if (typeof this.addDOMWidget === "function") {
                this._treeDomWidget = this.addDOMWidget("merger_tree_ui", "div", container, {
                    serialize: false,
                    hideOnZoom: false,
                });
            }

            const getActiveStage = () => {
                const stList = this._mergerState.stages;
                if (this._mergerState.activeStageIdx >= stList.length) {
                    this._mergerState.activeStageIdx = Math.max(0, stList.length - 1);
                }
                return stList[this._mergerState.activeStageIdx];
            };

            const resolveCustomNameForSource = (srcId) => {
                if (!srcId || srcId === "none") return null;
                const nameMap = this._mergerState.resolvedModelNames || {};
                if (nameMap[srcId]) return nameMap[srcId];
                const stObj = (this._mergerState.stages || []).find((s) => s.id === srcId);
                if (stObj && stObj.name && stObj.name.trim()) {
                    return stObj.name.trim();
                }
                return null;
            };

            const getSourceDisplayLabel = (srcId, maxLen = 24) => {
                if (!srcId || srcId === "none") return "none";
                const custom = resolveCustomNameForSource(srcId);
                if (!custom) return srcId;
                const short = custom.length > maxLen ? `${custom.slice(0, maxLen - 2)}..` : custom;
                return `${srcId} (${short})`;
            };

            const getSourceShortName = (srcId, maxLen = 18) => {
                if (!srcId || srcId === "none") return "none";
                const custom = resolveCustomNameForSource(srcId);
                if (!custom) return srcId;
                return custom.length > maxLen ? `${custom.slice(0, maxLen - 2)}..` : custom;
            };

            const applyCachedZeroFixesToStage = (stage) => {
                const fixes = this._mergerState.stageZeroFixes?.[stage.id];
                if (!fixes || typeof fixes !== "object") return 0;
                if (!stage.overrides) stage.overrides = {};
                if (!Array.isArray(stage.auto_zero_keys)) stage.auto_zero_keys = [];

                let count = 0;
                for (const [k, val] of Object.entries(fixes)) {
                    stage.overrides[k] = val;
                    if (!stage.auto_zero_keys.includes(k)) {
                        stage.auto_zero_keys.push(k);
                    }
                    count++;
                }
                return count;
            };

            const removeAutoZeroFixesFromStage = (stage) => {
                if (Array.isArray(stage.auto_zero_keys) && stage.overrides) {
                    for (const k of stage.auto_zero_keys) {
                        delete stage.overrides[k];
                    }
                }
                const fixes = this._mergerState.stageZeroFixes?.[stage.id];
                if (fixes && stage.overrides) {
                    for (const [k, val] of Object.entries(fixes)) {
                        if (stage.overrides[k] === val) {
                            delete stage.overrides[k];
                        }
                    }
                }
                stage.auto_zero_keys = [];
            };

            this._applyCachedZeroFixesToStage = applyCachedZeroFixesToStage;

            const getEffectiveVal = (stage, lvl1Id, lvl2Id = null, lvl3Id = null) => {
                const ov = stage.overrides || {};
                const zf = (stage.fix_zeros && this._mergerState.stageZeroFixes?.[stage.id]) || {};
                if (lvl3Id && ov[lvl3Id] !== undefined) return ov[lvl3Id];
                if (lvl3Id && zf[lvl3Id] !== undefined) return zf[lvl3Id];
                if (lvl2Id && ov[lvl2Id] !== undefined) return ov[lvl2Id];
                if (lvl2Id && zf[lvl2Id] !== undefined) return zf[lvl2Id];
                if (lvl1Id && ov[lvl1Id] !== undefined) return ov[lvl1Id];
                if (lvl1Id && zf[lvl1Id] !== undefined) return zf[lvl1Id];
                return stage.global_ratio ?? 0.5;
            };

            const getEffectiveOp = (stage, lvl1Id, lvl2Id = null, lvl3Id = null) => {
                const opOv = stage.op_overrides || {};
                if (lvl3Id && opOv[lvl3Id] !== undefined) return opOv[lvl3Id];
                if (lvl2Id && opOv[lvl2Id] !== undefined) return opOv[lvl2Id];
                if (lvl1Id && opOv[lvl1Id] !== undefined) return opOv[lvl1Id];
                return stage.global_op || "ws";
            };

            let activeRowControlsMap = new Map();
            let visibleRowIdsInOrder = [];

            const handleRowSelectionClick = (rowId, event) => {
                const sel = this._mergerState.selectedRows;
                if (event.shiftKey && this._mergerState.lastClickedRowId) {
                    const idxA = visibleRowIdsInOrder.indexOf(this._mergerState.lastClickedRowId);
                    const idxB = visibleRowIdsInOrder.indexOf(rowId);
                    if (idxA !== -1 && idxB !== -1) {
                        const start = Math.min(idxA, idxB);
                        const end = Math.max(idxA, idxB);
                        const shouldSelect = !sel.has(rowId);
                        for (let i = start; i <= end; i++) {
                            if (shouldSelect) sel.add(visibleRowIdsInOrder[i]);
                            else sel.delete(visibleRowIdsInOrder[i]);
                        }
                        this._mergerState.lastClickedRowId = rowId;
                        renderMergerUI();
                        return;
                    }
                }
                if (sel.has(rowId)) {
                    sel.delete(rowId);
                } else {
                    sel.add(rowId);
                }
                this._mergerState.lastClickedRowId = rowId;
                renderMergerUI();
            };

            const createSliderRow = ({
                rowId = null,
                label,
                indentPx,
                hasChildren,
                isExpanded,
                onToggleExpand,
                effectiveValue,
                effectiveOp,
                isDetached,
                isOpDetached,
                isSelected = false,
                onChangeValue,
                onChangeOp,
                onReattach,
            }) => {
                const row = document.createElement("div");
                const bg = isSelected
                    ? "rgba(52, 152, 219, 0.22)"
                    : isDetached
                      ? "rgba(230, 126, 34, 0.14)"
                      : "transparent";
                const borderLeft = isSelected
                    ? "#3498db"
                    : isDetached
                      ? "#e67e22"
                      : "transparent";

                row.style.cssText = `
                    display: flex;
                    align-items: center;
                    gap: 4px;
                    padding: 2px 4px;
                    padding-left: ${indentPx}px;
                    background: ${bg};
                    border-left: 3px solid ${borderLeft};
                    border-radius: 2px;
                `;

                if (rowId) {
                    const cb = document.createElement("input");
                    cb.type = "checkbox";
                    cb.className = "kitten-merger-cb";
                    cb.checked = isSelected;
                    cb.title = "Select row for multi-slider delta adjustment (Shift+Click for range)";
                    cb.onclick = (e) => {
                        e.stopPropagation();
                        handleRowSelectionClick(rowId, e);
                    };
                    row.appendChild(cb);
                }

                const expBtn = document.createElement("span");
                expBtn.style.cssText = `
                    width: 12px;
                    cursor: ${hasChildren ? "pointer" : "default"};
                    user-select: none;
                    color: #999;
                    font-weight: bold;
                    text-align: center;
                    flex-shrink: 0;
                `;
                expBtn.textContent = hasChildren ? (isExpanded ? "v" : ">") : "·";
                if (hasChildren && onToggleExpand) {
                    expBtn.onclick = onToggleExpand;
                }

                const lbl = document.createElement("span");
                lbl.style.cssText = `
                    flex: 1;
                    min-width: 65px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    user-select: none;
                    cursor: ${hasChildren || rowId ? "pointer" : "default"};
                    color: ${isSelected ? "#5dade2" : isDetached ? "#f39c12" : "#ddd"};
                `;
                lbl.textContent = label;
                lbl.title = rowId
                    ? `${label} (Ctrl+Click or Shift+Click to multi-select)`
                    : label;
                lbl.onclick = (e) => {
                    if (rowId && (e.ctrlKey || e.metaKey || e.shiftKey)) {
                        e.preventDefault();
                        handleRowSelectionClick(rowId, e);
                        return;
                    }
                    if (hasChildren && onToggleExpand) {
                        onToggleExpand();
                    } else if (rowId) {
                        handleRowSelectionClick(rowId, e);
                    }
                };

                const slider = document.createElement("input");
                slider.type = "range";
                slider.min = "0";
                slider.max = "1";
                slider.step = "0.01";
                slider.value = String(effectiveValue);
                slider.style.cssText = "width: 88px; cursor: pointer; margin: 0; flex-shrink: 0;";

                const numInput = document.createElement("input");
                numInput.type = "number";
                numInput.className = "kitten-merger-num";
                numInput.min = "-2";
                numInput.max = "3";
                numInput.step = "0.01";
                numInput.value = Number(effectiveValue).toFixed(2);
                numInput.style.cssText = `
                    width: 44px;
                    background: #141414;
                    color: ${isDetached ? "#f39c12" : "#ccc"};
                    border: 1px solid #444;
                    border-radius: 3px;
                    font-family: monospace;
                    font-size: 11px;
                    padding: 1px 4px;
                    text-align: right;
                    flex-shrink: 0;
                    box-sizing: border-box;
                `;

                if (rowId) {
                    activeRowControlsMap.set(rowId, {
                        slider,
                        numInput,
                        getInitialVal: () => effectiveValue,
                    });
                }

                let dragSnapshot = null;

                const takeMultiSnapshot = () => {
                    const sel = this._mergerState.selectedRows;
                    if (!rowId || !sel.has(rowId) || sel.size <= 1) {
                        dragSnapshot = null;
                        return;
                    }
                    const snap = new Map();
                    for (const selId of sel) {
                        const ctrl = activeRowControlsMap.get(selId);
                        if (ctrl) {
                            snap.set(selId, ctrl.getInitialVal());
                        }
                    }
                    dragSnapshot = {
                        baseVal: effectiveValue,
                        rows: snap,
                    };
                };

                slider.onpointerdown = () => {
                    takeMultiSnapshot();
                };

                slider.oninput = (e) => {
                    const v = parseFloat(e.target.value);
                    numInput.value = v.toFixed(2);
                    if (dragSnapshot && dragSnapshot.rows.size > 1) {
                        const delta = v - dragSnapshot.baseVal;
                        const st = getActiveStage();
                        for (const [selId, startVal] of dragSnapshot.rows.entries()) {
                            const nextVal = Math.max(
                                0,
                                Math.min(1, Number((startVal + delta).toFixed(4)))
                            );
                            st.overrides[selId] = nextVal;
                            const ctrl = activeRowControlsMap.get(selId);
                            if (ctrl && selId !== rowId) {
                                ctrl.slider.value = String(nextVal);
                                ctrl.numInput.value = nextVal.toFixed(2);
                            }
                        }
                        saveStateToWidget();
                    } else {
                        onChangeValue(v, false);
                    }
                };

                slider.onchange = (e) => {
                    const v = parseFloat(e.target.value);
                    if (dragSnapshot && dragSnapshot.rows.size > 1) {
                        const delta = v - dragSnapshot.baseVal;
                        const st = getActiveStage();
                        for (const [selId, startVal] of dragSnapshot.rows.entries()) {
                            const nextVal = Math.max(
                                0,
                                Math.min(1, Number((startVal + delta).toFixed(4)))
                            );
                            st.overrides[selId] = nextVal;
                        }
                        dragSnapshot = null;
                        saveStateToWidget();
                        renderMergerUI();
                    } else {
                        dragSnapshot = null;
                        onChangeValue(v, true);
                    }
                };

                numInput.onchange = (e) => {
                    let v = parseFloat(e.target.value);
                    if (Number.isNaN(v)) v = 0.5;
                    const sel = this._mergerState.selectedRows;
                    if (rowId && sel.has(rowId) && sel.size > 1) {
                        const delta = v - effectiveValue;
                        const st = getActiveStage();
                        for (const selId of sel) {
                            const ctrl = activeRowControlsMap.get(selId);
                            const startVal = ctrl ? ctrl.getInitialVal() : effectiveValue;
                            const nextVal = Math.max(
                                -2,
                                Math.min(3, Number((startVal + delta).toFixed(4)))
                            );
                            st.overrides[selId] = nextVal;
                        }
                        saveStateToWidget();
                        renderMergerUI();
                    } else {
                        slider.value = String(Math.max(0, Math.min(1, v)));
                        onChangeValue(v, true);
                    }
                };

                const applySteppedValue = (stepDelta) => {
                    const sel = this._mergerState.selectedRows;
                    if (rowId && sel.has(rowId) && sel.size > 1) {
                        const st = getActiveStage();
                        for (const selId of sel) {
                            const ctrl = activeRowControlsMap.get(selId);
                            const curVal = ctrl ? ctrl.getInitialVal() : effectiveValue;
                            const nextVal = Math.max(
                                -2,
                                Math.min(3, Number((curVal + stepDelta).toFixed(4)))
                            );
                            st.overrides[selId] = nextVal;
                        }
                        saveStateToWidget();
                        renderMergerUI();
                    } else {
                        const nextVal = Math.max(
                            -2,
                            Math.min(3, Number((effectiveValue + stepDelta).toFixed(4)))
                        );
                        onChangeValue(nextVal, true);
                    }
                };

                const stepDownBtn = document.createElement("button");
                stepDownBtn.textContent = "▼";
                stepDownBtn.title = "Decrease by 0.01 (Shift+Click for 0.05)";
                stepDownBtn.style.cssText = `
                    width: 16px;
                    height: 17px;
                    font-size: 8.5px;
                    padding: 0;
                    cursor: pointer;
                    background: #262626;
                    color: #bbb;
                    border: 1px solid #444;
                    border-radius: 2px;
                    flex-shrink: 0;
                `;
                stepDownBtn.onclick = (e) => {
                    e.stopPropagation();
                    applySteppedValue(e.shiftKey ? -0.05 : -0.01);
                };

                const stepUpBtn = document.createElement("button");
                stepUpBtn.textContent = "▲";
                stepUpBtn.title = "Increase by 0.01 (Shift+Click for 0.05)";
                stepUpBtn.style.cssText = `
                    width: 16px;
                    height: 17px;
                    font-size: 8.5px;
                    padding: 0;
                    cursor: pointer;
                    background: #262626;
                    color: #bbb;
                    border: 1px solid #444;
                    border-radius: 2px;
                    flex-shrink: 0;
                `;
                stepUpBtn.onclick = (e) => {
                    e.stopPropagation();
                    applySteppedValue(e.shiftKey ? 0.05 : 0.01);
                };

                const opGroup = document.createElement("div");
                opGroup.style.cssText = "display: flex; gap: 1px; flex-shrink: 0;";
                OP_MODES.forEach((m) => {
                    const btn = document.createElement("button");
                    const isSelectedOp = effectiveOp === m.code;
                    btn.textContent = m.short;
                    btn.title = m.tip;
                    const activeBg = isOpDetached ? "#d35400" : "#2980b9";
                    btn.style.cssText = `
                        min-width: 19px;
                        height: 17px;
                        font-family: monospace;
                        font-size: 9.5px;
                        padding: 0 2px;
                        cursor: pointer;
                        background: ${isSelectedOp ? activeBg : "#222"};
                        color: ${isSelectedOp ? "#fff" : "#888"};
                        border: 1px solid ${isSelectedOp ? (isOpDetached ? "#e67e22" : "#3498db") : "#3a3a3a"};
                        border-radius: 2px;
                    `;
                    btn.onclick = () => {
                        const sel = this._mergerState.selectedRows;
                        if (rowId && sel.has(rowId) && sel.size > 1) {
                            const st = getActiveStage();
                            for (const selId of sel) {
                                st.op_overrides[selId] = m.code;
                            }
                            saveStateToWidget();
                            renderMergerUI();
                        } else if (onChangeOp) {
                            onChangeOp(m.code);
                        }
                    };
                    opGroup.appendChild(btn);
                });

                const reattachBtn = document.createElement("button");
                reattachBtn.textContent = "R";
                reattachBtn.title = "Reattach weight and operation to parent";
                reattachBtn.style.cssText = `
                    width: 18px;
                    height: 17px;
                    font-size: 10px;
                    padding: 0;
                    cursor: pointer;
                    background: #c0392b;
                    color: #fff;
                    border: none;
                    border-radius: 3px;
                    flex-shrink: 0;
                    visibility: ${isDetached ? "visible" : "hidden"};
                `;
                if (isDetached && onReattach) {
                    reattachBtn.onclick = () => {
                        const sel = this._mergerState.selectedRows;
                        if (rowId && sel.has(rowId) && sel.size > 1) {
                            const st = getActiveStage();
                            for (const selId of sel) {
                                delete st.overrides[selId];
                                delete st.op_overrides[selId];
                            }
                            saveStateToWidget();
                            renderMergerUI();
                        } else {
                            onReattach();
                        }
                    };
                }

                row.appendChild(expBtn);
                row.appendChild(lbl);
                row.appendChild(slider);
                row.appendChild(stepDownBtn);
                row.appendChild(numInput);
                row.appendChild(stepUpBtn);
                row.appendChild(opGroup);
                row.appendChild(reattachBtn);
                return row;
            };

            const applyImportedOverridesToStage = (stage, importedOverrides, mergeIntoExisting, invertWeights) => {
                const baseOv = mergeIntoExisting ? { ...(stage.overrides || {}) } : {};
                for (const [k, rawVal] of Object.entries(importedOverrides)) {
                    const num = parseFloat(rawVal);
                    if (Number.isNaN(num)) continue;
                    const finalVal = invertWeights ? Number((1.0 - num).toFixed(4)) : Number(num.toFixed(4));
                    baseOv[k] = finalVal;
                }
                stage.overrides = baseOv;
            };

            const renderImportExportModal = () => {
                if (!this._mergerState.showImpExpModal) {
                    modalOverlay.style.display = "none";
                    return;
                }
                modalOverlay.style.display = "flex";
                modalOverlay.innerHTML = "";

                const st = getActiveStage();

                const headerRow = document.createElement("div");
                headerRow.style.cssText = "display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #333; padding-bottom: 6px;";
                const title = document.createElement("span");
                title.style.cssText = "font-weight: bold; color: #f39c12; font-size: 12px;";
                title.textContent = `Import / Export Block & Layer Values (Active Stage: ${st.id})`;

                const closeBtn = document.createElement("button");
                closeBtn.textContent = "Close [X]";
                closeBtn.style.cssText = "padding: 2px 8px; background: #444; color: #fff; border: none; border-radius: 3px; cursor: pointer; font-size: 11px;";
                closeBtn.onclick = () => {
                    this._mergerState.showImpExpModal = false;
                    modalOverlay.style.display = "none";
                };
                headerRow.appendChild(title);
                headerRow.appendChild(closeBtn);
                modalOverlay.appendChild(headerRow);

                const statusMsg = document.createElement("div");
                statusMsg.style.cssText = "color: #2ecc71; font-size: 11px; min-height: 15px;";

                const textArea = document.createElement("textarea");
                textArea.placeholder =
                    "Paste here or load a .json file:\n" +
                    "• Old ComfyUI Workflow .json (auto-detects ModelMergeAnima2_9B & ModelMergeAnima2_9B_Granular)\n" +
                    "• Raw widgets_values array (45 values for 2.9B Basic or 249 values for 2.9B Granular)\n" +
                    "• Key-Value JSON {'pos_embedder.': 1.0, 'blocks.0.': 0.9, ...}\n" +
                    "• Single Stage JSON or Full Multi-Stage Merger Recipe JSON";
                textArea.style.cssText = `
                    width: 100%;
                    height: 190px;
                    background: #121212;
                    color: #d4d4d4;
                    border: 1px solid #444;
                    border-radius: 3px;
                    font-family: monospace;
                    font-size: 11px;
                    padding: 6px;
                    box-sizing: border-box;
                    resize: vertical;
                `;

                const expRow = document.createElement("div");
                expRow.style.cssText = "display: flex; align-items: center; gap: 5px; flex-wrap: wrap;";
                const expLbl = document.createElement("span");
                expLbl.style.cssText = "color: #aaa; font-weight: bold;";
                expLbl.textContent = "Export:";
                expRow.appendChild(expLbl);

                const makeBtn = (label, bg, onClick) => {
                    const b = document.createElement("button");
                    b.textContent = label;
                    b.style.cssText = `padding: 3px 7px; font-size: 10.5px; cursor: pointer; background: ${bg}; color: #fff; border: none; border-radius: 3px;`;
                    b.onclick = onClick;
                    return b;
                };

                const copyOrPopulate = async (obj, labelText) => {
                    const str = JSON.stringify(obj, null, 2);
                    textArea.value = str;
                    try {
                        await navigator.clipboard.writeText(str);
                        statusMsg.style.color = "#2ecc71";
                        statusMsg.textContent = `${labelText} copied to clipboard and shown below.`;
                    } catch (e) {
                        statusMsg.style.color = "#f39c12";
                        statusMsg.textContent = `${labelText} populated below (Ctrl+C to copy).`;
                    }
                };

                expRow.appendChild(
                    makeBtn("Stage JSON", "#2980b9", () => copyOrPopulate(st, `Stage ${st.id} JSON`))
                );
                expRow.appendChild(
                    makeBtn("Download Stage .json", "#1f618d", () => {
                        const str = JSON.stringify(st, null, 2);
                        textArea.value = str;
                        triggerJsonDownload(`anima_stage_${st.id}.json`, str);
                    })
                );
                expRow.appendChild(
                    makeBtn("All Stages Recipe", "#27ae60", () =>
                        copyOrPopulate({ stages: this._mergerState.stages }, "Full Multi-Stage Recipe")
                    )
                );
                expRow.appendChild(
                    makeBtn("Download Recipe .json", "#1e8449", () => {
                        const str = JSON.stringify({ stages: this._mergerState.stages }, null, 2);
                        textArea.value = str;
                        triggerJsonDownload("anima_merger_recipe.json", str);
                    })
                );
                expRow.appendChild(
                    makeBtn("Flat Key-Value Map", "#8e44ad", () => {
                        const flat = {};
                        const groups = buildTreeStructure(
                            this._mergerState.blockCount,
                            this._mergerState.customLeafKeys
                        );
                        for (const g of groups) {
                            flat[g.id] = getEffectiveVal(st, g.id);
                            for (const sub of g.children) {
                                flat[sub.id] = getEffectiveVal(st, g.id, sub.id);
                            }
                        }
                        copyOrPopulate(flat, "Flat Key-Value Map");
                    })
                );
                modalOverlay.appendChild(expRow);

                const impOptRow = document.createElement("div");
                impOptRow.style.cssText = "display: flex; align-items: center; gap: 10px; flex-wrap: wrap; background: #1e1e1e; padding: 5px 8px; border-radius: 3px;";

                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.accept = ".json,.txt";
                fileInput.style.display = "none";

                const loadFileBtn = makeBtn("Load .json / Workflow File...", "#d35400", () => fileInput.click());
                impOptRow.appendChild(loadFileBtn);
                impOptRow.appendChild(fileInput);

                const mergeLbl = document.createElement("label");
                mergeLbl.style.cssText = "display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 11px;";
                const mergeCb = document.createElement("input");
                mergeCb.type = "checkbox";
                mergeCb.checked = true;
                mergeLbl.appendChild(mergeCb);
                mergeLbl.appendChild(document.createTextNode("Merge into current stage (keep non-overlapping keys)"));
                impOptRow.appendChild(mergeLbl);

                const invLbl = document.createElement("label");
                invLbl.style.cssText = "display: flex; align-items: center; gap: 4px; cursor: pointer; font-size: 11px;";
                const invCb = document.createElement("input");
                invCb.type = "checkbox";
                invCb.checked = false;
                invLbl.appendChild(invCb);
                invLbl.appendChild(document.createTextNode("Invert imported weights (1.0 - w)"));
                impOptRow.appendChild(invLbl);

                modalOverlay.appendChild(impOptRow);

                const candidateRow = document.createElement("div");
                candidateRow.style.cssText = "display: none; align-items: center; gap: 6px; background: #252525; padding: 5px 8px; border-radius: 3px;";
                const candLbl = document.createElement("span");
                candLbl.textContent = "Workflow Node:";
                const candSelect = document.createElement("select");
                candSelect.style.cssText = "flex: 1; background: #141414; color: #eee; border: 1px solid #444; font-size: 11px; padding: 2px 4px;";
                candidateRow.appendChild(candLbl);
                candidateRow.appendChild(candSelect);
                modalOverlay.appendChild(candidateRow);

                let currentCandidates = null;

                const inspectTextAndPopulateCandidates = () => {
                    const res = parseImportPayload(textArea.value);
                    if (res.kind === "workflow_candidates") {
                        currentCandidates = res.candidates;
                        candSelect.innerHTML = "";
                        res.candidates.forEach((c, idx) => {
                            const o = document.createElement("option");
                            o.value = String(idx);
                            o.textContent = c.label;
                            candSelect.appendChild(o);
                        });
                        candidateRow.style.display = "flex";
                        statusMsg.style.color = "#3498db";
                        statusMsg.textContent = `Found ${res.candidates.length} importable option(s) in workflow JSON. Select above and click Apply Import.`;
                    } else {
                        currentCandidates = null;
                        candidateRow.style.display = "none";
                        if (res.description) {
                            statusMsg.style.color = "#3498db";
                            statusMsg.textContent = res.description;
                        }
                    }
                };

                fileInput.onchange = (e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    const reader = new FileReader();
                    reader.onload = (ev) => {
                        textArea.value = String(ev.target?.result || "");
                        inspectTextAndPopulateCandidates();
                    };
                    reader.readAsText(file);
                };

                textArea.oninput = () => {
                    inspectTextAndPopulateCandidates();
                };

                modalOverlay.appendChild(textArea);

                const applyRow = document.createElement("div");
                applyRow.style.cssText = "display: flex; align-items: center; gap: 8px;";

                const applyBtn = makeBtn("Apply Import", "#27ae60", () => {
                    const mergeMode = mergeCb.checked;
                    const invertMode = invCb.checked;
                    const res = parseImportPayload(textArea.value);

                    if (res.kind === "empty" || res.kind === "error") {
                        statusMsg.style.color = "#e74c3c";
                        statusMsg.textContent = res.message;
                        return;
                    }

                    let targetPayload = res;
                    if (res.kind === "workflow_candidates" && currentCandidates) {
                        const selIdx = parseInt(candSelect.value, 10) || 0;
                        targetPayload = currentCandidates[selIdx] || currentCandidates[0];
                    }

                    if (targetPayload.kind === "overrides") {
                        applyImportedOverridesToStage(st, targetPayload.overrides, mergeMode, invertMode);
                    } else if (targetPayload.kind === "stage") {
                        const impSt = normalizeStageObj(targetPayload.stage, this._mergerState.activeStageIdx);
                        st.global_ratio = invertMode ? Number((1.0 - impSt.global_ratio).toFixed(4)) : impSt.global_ratio;
                        st.global_op = impSt.global_op;
                        applyImportedOverridesToStage(st, impSt.overrides, mergeMode, invertMode);
                        st.op_overrides = mergeMode
                            ? { ...(st.op_overrides || {}), ...(impSt.op_overrides || {}) }
                            : { ...(impSt.op_overrides || {}) };
                    } else if (targetPayload.kind === "recipe") {
                        const stagesArr = targetPayload.recipe.stages || [];
                        if (stagesArr.length > 0) {
                            this._mergerState.stages = stagesArr.map((s, idx) => {
                                const norm = normalizeStageObj(s, idx);
                                if (invertMode) {
                                    norm.global_ratio = Number((1.0 - norm.global_ratio).toFixed(4));
                                    for (const k of Object.keys(norm.overrides)) {
                                        norm.overrides[k] = Number((1.0 - norm.overrides[k]).toFixed(4));
                                    }
                                }
                                return norm;
                            });
                            this._mergerState.activeStageIdx = 0;
                            syncDynamicModelSlots(this, { allowShrink: true });
                        }
                    }

                    this._mergerState.showImpExpModal = false;
                    saveStateToWidget();
                    renderMergerUI();
                });

                applyRow.appendChild(applyBtn);
                applyRow.appendChild(statusMsg);
                modalOverlay.appendChild(applyRow);
            };

            const renderMergerUI = () => {
                updateInputSlotLabelsAndNames(this);
                const st = getActiveStage();
                const connectedModels = getConnectedModelIds(this);
                const viewMode = st.view_mode || "sub_layers";
                const nameMap = this._mergerState.resolvedModelNames || {};

                activeRowControlsMap = new Map();
                visibleRowIdsInOrder = [];

                topBar.innerHTML = "";

                // Row 1: Stage Tabs + Imp/Exp + View Mode selector
                const stageTabRow = document.createElement("div");
                stageTabRow.style.cssText = "display: flex; align-items: center; gap: 4px; flex-wrap: wrap;";
                const stageTabBtns = [];

                this._mergerState.stages.forEach((sObj, idx) => {
                    const tabBtn = document.createElement("button");
                    const isActive = idx === this._mergerState.activeStageIdx;
                    const isStageEnabled = sObj.enabled !== false;
                    tabBtn.textContent = formatStageTabLabel(sObj);
                    stageTabBtns.push(tabBtn);
                    const fullA = nameMap[sObj.model_a] ? `${sObj.model_a} (${nameMap[sObj.model_a]})` : sObj.model_a;
                    const fullB = nameMap[sObj.model_b] ? `${sObj.model_b} (${nameMap[sObj.model_b]})` : sObj.model_b;
                    tabBtn.title = `Stage ${sObj.id}${isStageEnabled ? "" : " (DISABLED)"}: Model A = ${fullA}, Model B = ${fullB}${sObj.fix_zeros ? " [Fix Zeros ON]" : ""}`;
                    tabBtn.style.cssText = `
                        padding: 2px 7px;
                        font-family: monospace;
                        font-size: 11px;
                        cursor: pointer;
                        background: ${isActive ? (isStageEnabled ? "#2980b9" : "#555") : "#333"};
                        color: ${isStageEnabled ? "#fff" : "#999"};
                        opacity: ${isStageEnabled ? "1" : "0.7"};
                        border: 1px solid ${isActive ? (isStageEnabled ? "#3498db" : "#777") : "#444"};
                        border-radius: 3px;
                    `;
                    tabBtn.onclick = () => {
                        this._mergerState.activeStageIdx = idx;
                        saveStateToWidget();
                        renderMergerUI();
                    };
                    stageTabRow.appendChild(tabBtn);
                });

                const addStageBtn = document.createElement("button");
                addStageBtn.textContent = "+ Stage";
                addStageBtn.style.cssText = `
                    padding: 2px 6px;
                    font-size: 11px;
                    cursor: pointer;
                    background: #27ae60;
                    color: #fff;
                    border: none;
                    border-radius: 3px;
                `;
                addStageBtn.onclick = () => {
                    const nextNum = this._mergerState.stages.length + 1;
                    const prevStageId = this._mergerState.stages[this._mergerState.stages.length - 1].id;
                    const nextModelB = connectedModels[Math.min(nextNum, connectedModels.length - 1)] || "m2";
                    this._mergerState.stages.push(
                        normalizeStageObj(
                            {
                                id: `s${nextNum}`,
                                name: "",
                                view_mode: st.view_mode || "sub_layers",
                                model_a: prevStageId,
                                model_b: nextModelB,
                                model_c: "none",
                                global_op: "ws",
                                global_ratio: 0.5,
                                fix_zeros: false,
                                overrides: {},
                                op_overrides: {},
                            },
                            nextNum - 1
                        )
                    );
                    this._mergerState.activeStageIdx = this._mergerState.stages.length - 1;
                    syncDynamicModelSlots(this, { allowShrink: true });
                    saveStateToWidget();
                    renderMergerUI();
                    this.setDirtyCanvas?.(true, true);
                };
                stageTabRow.appendChild(addStageBtn);

                const rightControls = document.createElement("div");
                rightControls.style.cssText = "margin-left: auto; display: flex; align-items: center; gap: 4px;";

                const impExpBtn = document.createElement("button");
                impExpBtn.textContent = "Imp/Exp";
                impExpBtn.title = "Import or Export block/layer values (supports legacy ModelMergeAnima2_9B & Granular workflows)";
                impExpBtn.style.cssText = `
                    padding: 2px 6px;
                    font-size: 10.5px;
                    cursor: pointer;
                    background: #8e44ad;
                    color: #fff;
                    border: none;
                    border-radius: 3px;
                `;
                impExpBtn.onclick = () => {
                    this._mergerState.showImpExpModal = !this._mergerState.showImpExpModal;
                    renderImportExportModal();
                };
                rightControls.appendChild(impExpBtn);

                const modeSelect = document.createElement("select");
                modeSelect.style.cssText = `
                    background: #141414;
                    color: #ddd;
                    border: 1px solid #444;
                    border-radius: 3px;
                    font-size: 11px;
                    padding: 1px 4px;
                `;
                [
                    { val: "basic", text: "Mode: Basic" },
                    { val: "blocks", text: "Mode: Blocks" },
                    { val: "layers", text: "Mode: Layers" },
                    { val: "sub_layers", text: "Mode: Sub-Layers" },
                ].forEach((opt) => {
                    const o = document.createElement("option");
                    o.value = opt.val;
                    o.textContent = opt.text;
                    if (opt.val === viewMode) o.selected = true;
                    modeSelect.appendChild(o);
                });
                modeSelect.onchange = (e) => {
                    st.view_mode = e.target.value;
                    ensureAdequateNodeHeight(this);
                    saveStateToWidget();
                    renderMergerUI();
                };
                rightControls.appendChild(modeSelect);
                stageTabRow.appendChild(rightControls);
                topBar.appendChild(stageTabRow);

                // Row 2: Dedicated non-wrapping row for Model A(1) / Model B(0) / [Model C] / Fix Zeros
                const priorStageIds = this._mergerState.stages
                    .slice(0, this._mergerState.activeStageIdx)
                    .map((x) => x.id);
                const validSources = [...connectedModels, ...priorStageIds];

                const srcRow = document.createElement("div");
                srcRow.style.cssText = "display: flex; align-items: center; gap: 5px; flex-wrap: nowrap; width: 100%;";

                const makeSourceDropdown = (currentVal, onSelect, allowNone = false) => {
                    const sel = document.createElement("select");
                    sel.style.cssText = `
                        flex: 1;
                        min-width: 70px;
                        max-width: 185px;
                        background: #141414;
                        color: #eee;
                        border: 1px solid #444;
                        border-radius: 3px;
                        font-size: 11px;
                        padding: 1px 4px;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                    `;
                    const baseOpts = allowNone ? ["none", ...validSources] : validSources;
                    const opts = baseOpts.includes(currentVal) ? baseOpts : [currentVal, ...baseOpts];
                    opts.forEach((srcId) => {
                        const o = document.createElement("option");
                        o.value = srcId;
                        o.textContent = getSourceDisplayLabel(srcId, 22);
                        o.title = nameMap[srcId] ? `${srcId}: ${nameMap[srcId]}` : srcId;
                        if (srcId === currentVal) o.selected = true;
                        sel.appendChild(o);
                    });
                    sel.title = nameMap[currentVal] ? `${currentVal}: ${nameMap[currentVal]}` : currentVal;
                    sel.onchange = (e) => onSelect(e.target.value);
                    return sel;
                };

                const lblA = document.createElement("span");
                lblA.textContent = "A(1):";
                lblA.style.flexShrink = "0";
                lblA.title = "Model A (Weight = 1.0)";
                const selA = makeSourceDropdown(st.model_a, (val) => {
                    st.model_a = val;
                    saveStateToWidget();
                    renderMergerUI();
                });

                const lblB = document.createElement("span");
                lblB.textContent = "B(0):";
                lblB.style.flexShrink = "0";
                lblB.title = "Model B (Weight = 0.0)";
                const selB = makeSourceDropdown(st.model_b, (val) => {
                    st.model_b = val;
                    saveStateToWidget();
                    renderMergerUI();
                });

                srcRow.appendChild(lblA);
                srcRow.appendChild(selA);
                srcRow.appendChild(lblB);
                srcRow.appendChild(selB);

                if (stageUsesThreeModels(st)) {
                    const lblC = document.createElement("span");
                    lblC.textContent = "C:";
                    lblC.style.flexShrink = "0";
                    lblC.title = "Model C (for Add Difference / Cosine modes)";
                    const selC = makeSourceDropdown(
                        st.model_c || "none",
                        (val) => {
                            st.model_c = val;
                            saveStateToWidget();
                            renderMergerUI();
                        },
                        true
                    );
                    srcRow.appendChild(lblC);
                    srcRow.appendChild(selC);
                }

                const isEnabled = st.enabled !== false;
                const enableStageBtn = document.createElement("button");
                enableStageBtn.textContent = isEnabled ? "Enabled" : "Disabled";
                enableStageBtn.title =
                    "Enable or disable this stage. Disabled stages are skipped unless another enabled stage or output wire directly depends on them. Disable all stages to use the Merger as a pure passthrough.";
                enableStageBtn.style.cssText = `
                    margin-left: auto;
                    padding: 2px 7px;
                    font-size: 10px;
                    cursor: pointer;
                    background: ${isEnabled ? "#27ae60" : "#555"};
                    color: #fff;
                    border: 1px solid ${isEnabled ? "#2ecc71" : "#666"};
                    border-radius: 3px;
                    flex-shrink: 0;
                `;
                enableStageBtn.onclick = () => {
                    st.enabled = !isEnabled;
                    saveStateToWidget();
                    renderMergerUI();
                };
                srcRow.appendChild(enableStageBtn);

                const cachedStageFixes = this._mergerState.stageZeroFixes?.[st.id];
                const fixCount = cachedStageFixes ? Object.keys(cachedStageFixes).length : 0;
                const fixZeroBtn = document.createElement("button");
                fixZeroBtn.textContent = st.fix_zeros
                    ? `Fix Zeros: ON${fixCount > 0 ? ` (${fixCount})` : ""}`
                    : `Fix Zeros${fixCount > 0 ? ` (${fixCount})` : ""}`;
                fixZeroBtn.title =
                    "1-Click Stage Zero Fix: Automatically sets any zero/unpopulated layers in Model B to 1.00 (purely Model A) and any zero layers in Model A to 0.00 (purely Model B).";
                fixZeroBtn.style.cssText = `
                    padding: 2px 7px;
                    font-size: 10px;
                    cursor: pointer;
                    background: ${st.fix_zeros ? "#16a085" : "#2c3e50"};
                    color: #fff;
                    border: 1px solid ${st.fix_zeros ? "#1abc9c" : "#3d566e"};
                    border-radius: 3px;
                    flex-shrink: 0;
                `;
                fixZeroBtn.onclick = () => {
                    st.fix_zeros = !st.fix_zeros;
                    if (st.fix_zeros) {
                        applyCachedZeroFixesToStage(st);
                    } else {
                        removeAutoZeroFixesFromStage(st);
                    }
                    saveStateToWidget();
                    renderMergerUI();
                };
                srcRow.appendChild(fixZeroBtn);

                topBar.appendChild(srcRow);

                // Row 3: Dedicated Toolbar Row for Stage Name + Multi-Select + Reattach All + Del Stage
                const actionRow = document.createElement("div");
                actionRow.style.cssText = "display: flex; align-items: center; gap: 5px; width: 100%;";

                const nameLbl = document.createElement("span");
                nameLbl.textContent = "Name:";
                nameLbl.style.cssText = "color: #aaa; font-size: 10.5px; flex-shrink: 0;";
                const nameInput = document.createElement("input");
                nameInput.type = "text";
                nameInput.placeholder = `${st.id} (optional)`;
                nameInput.value = st.name || "";
                nameInput.title = "Optional custom name for this stage (updates stage tab and output slot label)";
                nameInput.style.cssText = `
                    width: 115px;
                    background: #141414;
                    color: #eee;
                    border: 1px solid #444;
                    border-radius: 3px;
                    font-family: monospace;
                    font-size: 10.5px;
                    padding: 1px 4px;
                `;
                nameInput.oninput = (e) => {
                    st.name = e.target.value;
                    if (stageTabBtns[this._mergerState.activeStageIdx]) {
                        stageTabBtns[this._mergerState.activeStageIdx].textContent = formatStageTabLabel(st);
                    }
                    syncDynamicModelSlots(this, { allowShrink: false });
                    saveStateToWidget();
                    this.setDirtyCanvas?.(true, true);
                };
                nameInput.onchange = () => {
                    renderMergerUI();
                };
                actionRow.appendChild(nameLbl);
                actionRow.appendChild(nameInput);

                if (viewMode !== "basic") {
                    const selCount = this._mergerState.selectedRows.size;
                    const selAllBtn = document.createElement("button");
                    selAllBtn.textContent = "Sel All";
                    selAllBtn.title = "Select all visible rows in tree";
                    selAllBtn.style.cssText = "padding: 2px 6px; font-size: 10px; cursor: pointer; background: #2c3e50; color: #eee; border: none; border-radius: 3px;";
                    selAllBtn.onclick = () => {
                        for (const id of visibleRowIdsInOrder) {
                            this._mergerState.selectedRows.add(id);
                        }
                        renderMergerUI();
                    };

                    const invSelBtn = document.createElement("button");
                    invSelBtn.textContent = "Inv";
                    invSelBtn.title = "Invert selection among visible rows";
                    invSelBtn.style.cssText = "padding: 2px 6px; font-size: 10px; cursor: pointer; background: #2c3e50; color: #eee; border: none; border-radius: 3px;";
                    invSelBtn.onclick = () => {
                        for (const id of visibleRowIdsInOrder) {
                            if (this._mergerState.selectedRows.has(id)) {
                                this._mergerState.selectedRows.delete(id);
                            } else {
                                this._mergerState.selectedRows.add(id);
                            }
                        }
                        renderMergerUI();
                    };

                    const clrSelBtn = document.createElement("button");
                    clrSelBtn.textContent = selCount > 0 ? `Deselect All (${selCount})` : "Deselect All";
                    clrSelBtn.title = "Deselect all selected sliders";
                    clrSelBtn.style.cssText = `
                        padding: 2px 7px;
                        font-size: 10px;
                        cursor: ${selCount > 0 ? "pointer" : "default"};
                        background: ${selCount > 0 ? "#2980b9" : "#2f2f2f"};
                        color: ${selCount > 0 ? "#fff" : "#777"};
                        border: 1px solid ${selCount > 0 ? "#3498db" : "#3a3a3a"};
                        border-radius: 3px;
                    `;
                    clrSelBtn.onclick = () => {
                        if (this._mergerState.selectedRows.size > 0) {
                            this._mergerState.selectedRows.clear();
                            renderMergerUI();
                        }
                    };

                    actionRow.appendChild(selAllBtn);
                    actionRow.appendChild(invSelBtn);
                    actionRow.appendChild(clrSelBtn);
                }

                const overrideKeys = new Set([
                    ...Object.keys(st.overrides || {}),
                    ...Object.keys(st.op_overrides || {}),
                ]);
                const resetStageBtn = document.createElement("button");
                resetStageBtn.textContent = `Reattach All (${overrideKeys.size})`;
                resetStageBtn.style.cssText = `
                    margin-left: auto;
                    padding: 2px 6px;
                    font-size: 10px;
                    cursor: pointer;
                    background: #444;
                    color: #eee;
                    border: none;
                    border-radius: 3px;
                `;
                resetStageBtn.onclick = () => {
                    st.overrides = {};
                    st.op_overrides = {};
                    st.auto_zero_keys = [];
                    st.fix_zeros = false;
                    saveStateToWidget();
                    renderMergerUI();
                };
                actionRow.appendChild(resetStageBtn);

                if (this._mergerState.stages.length > 1) {
                    const delBtn = document.createElement("button");
                    delBtn.textContent = "Del Stage";
                    delBtn.style.cssText = "padding: 2px 6px; font-size: 10px; cursor: pointer; background: #7f1d1d; color: #fff; border: none; border-radius: 3px;";
                    delBtn.onclick = () => {
                        this._mergerState.stages.splice(this._mergerState.activeStageIdx, 1);
                        this._mergerState.stages.forEach((s, i) => {
                            s.id = `s${i + 1}`;
                        });
                        this._mergerState.activeStageIdx = Math.max(0, this._mergerState.activeStageIdx - 1);
                        syncDynamicModelSlots(this, { allowShrink: true });
                        saveStateToWidget();
                        renderMergerUI();
                        this.setDirtyCanvas?.(true, true);
                    };
                    actionRow.appendChild(delBtn);
                }
                topBar.appendChild(actionRow);

                // Row 4: Stage Master / Global Ratio & Global Op Selector
                const shortA = getSourceShortName(st.model_a, 16);
                const shortB = getSourceShortName(st.model_b, 16);
                const globalRow = createSliderRow({
                    rowId: null,
                    label: `Global (${shortA} [1] <-> ${shortB} [0])`,
                    indentPx: 2,
                    hasChildren: false,
                    isExpanded: false,
                    effectiveValue: st.global_ratio ?? 0.5,
                    effectiveOp: st.global_op || "ws",
                    isDetached: false,
                    isOpDetached: false,
                    isSelected: false,
                    onChangeValue: (val, shouldRender) => {
                        st.global_ratio = val;
                        saveStateToWidget();
                        if (shouldRender) renderMergerUI();
                    },
                    onChangeOp: (opCode) => {
                        st.global_op = opCode;
                        if (opCode === "add_diff" && (!st.model_c || st.model_c === "none")) {
                            st.model_c = validSources[Math.min(2, validSources.length - 1)] || "m2";
                        }
                        saveStateToWidget();
                        renderMergerUI();
                    },
                });
                topBar.appendChild(globalRow);

                renderImportExportModal();

                if (viewMode === "basic") {
                    treeScroll.style.display = "none";
                    const basicInfo = document.createElement("div");
                    basicInfo.style.cssText = "color: #999; font-size: 10.5px; padding: 2px 4px;";
                    const ovCount = overrideKeys.size;
                    const fzNote = st.fix_zeros ? " [Fix Zeros ON]" : "";
                    basicInfo.textContent =
                        ovCount > 0
                            ? `Basic Mode active (${ovCount} block/layer override${ovCount === 1 ? "" : "s"} applied in background)${fzNote}.`
                            : `Basic Mode active (uniform stage operation across all blocks & layers)${fzNote}.`;
                    topBar.appendChild(basicInfo);
                    return;
                }

                treeScroll.style.display = "flex";
                const prevScrollTop = treeScroll.scrollTop;
                treeScroll.innerHTML = "";

                const treeGroups = buildTreeStructure(
                    this._mergerState.blockCount,
                    this._mergerState.customLeafKeys
                );

                for (const grp of treeGroups) {
                    const isGrpExpanded =
                        viewMode === "layers" ||
                        (viewMode === "sub_layers" && this._mergerState.expandedLvl1.has(grp.id));
                    const canExpandGrp = viewMode === "sub_layers" && grp.children.length > 0;
                    const grpWDetached = st.overrides && st.overrides[grp.id] !== undefined;
                    const grpOpDetached = st.op_overrides && st.op_overrides[grp.id] !== undefined;
                    const grpDetached = grpWDetached || grpOpDetached;
                    const grpVal = getEffectiveVal(st, grp.id);
                    const grpOp = getEffectiveOp(st, grp.id);

                    visibleRowIdsInOrder.push(grp.id);

                    const grpRow = createSliderRow({
                        rowId: grp.id,
                        label: grp.label,
                        indentPx: 4,
                        hasChildren: canExpandGrp,
                        isExpanded: isGrpExpanded,
                        onToggleExpand: () => {
                            if (this._mergerState.expandedLvl1.has(grp.id)) {
                                this._mergerState.expandedLvl1.delete(grp.id);
                            } else {
                                this._mergerState.expandedLvl1.add(grp.id);
                            }
                            saveStateToWidget();
                            renderMergerUI();
                        },
                        effectiveValue: grpVal,
                        effectiveOp: grpOp,
                        isDetached: grpDetached,
                        isOpDetached: grpOpDetached,
                        isSelected: this._mergerState.selectedRows.has(grp.id),
                        onChangeValue: (val, shouldRender) => {
                            st.overrides[grp.id] = val;
                            saveStateToWidget();
                            if (shouldRender) renderMergerUI();
                        },
                        onChangeOp: (opCode) => {
                            st.op_overrides[grp.id] = opCode;
                            if (opCode === "add_diff" && (!st.model_c || st.model_c === "none")) {
                                st.model_c = validSources[Math.min(2, validSources.length - 1)] || "m2";
                            }
                            saveStateToWidget();
                            renderMergerUI();
                        },
                        onReattach: () => {
                            delete st.overrides[grp.id];
                            delete st.op_overrides[grp.id];
                            saveStateToWidget();
                            renderMergerUI();
                        },
                    });
                    treeScroll.appendChild(grpRow);

                    if (viewMode !== "blocks" && isGrpExpanded) {
                        for (const sub of grp.children) {
                            const canExpandSub =
                                viewMode === "sub_layers" && sub.children && sub.children.length > 0;
                            const isSubExpanded = canExpandSub && this._mergerState.expandedLvl2.has(sub.id);
                            const subWDetached = st.overrides && st.overrides[sub.id] !== undefined;
                            const subOpDetached = st.op_overrides && st.op_overrides[sub.id] !== undefined;
                            const subDetached = subWDetached || subOpDetached;
                            const subVal = getEffectiveVal(st, grp.id, sub.id);
                            const subOp = getEffectiveOp(st, grp.id, sub.id);

                            visibleRowIdsInOrder.push(sub.id);

                            const subRow = createSliderRow({
                                rowId: sub.id,
                                label: sub.label,
                                indentPx: 20,
                                hasChildren: canExpandSub,
                                isExpanded: isSubExpanded,
                                onToggleExpand: () => {
                                    if (this._mergerState.expandedLvl2.has(sub.id)) {
                                        this._mergerState.expandedLvl2.delete(sub.id);
                                    } else {
                                        this._mergerState.expandedLvl2.add(sub.id);
                                    }
                                    saveStateToWidget();
                                    renderMergerUI();
                                },
                                effectiveValue: subVal,
                                effectiveOp: subOp,
                                isDetached: subDetached,
                                isOpDetached: subOpDetached,
                                isSelected: this._mergerState.selectedRows.has(sub.id),
                                onChangeValue: (val, shouldRender) => {
                                    st.overrides[sub.id] = val;
                                    saveStateToWidget();
                                    if (shouldRender) renderMergerUI();
                                },
                                onChangeOp: (opCode) => {
                                    st.op_overrides[sub.id] = opCode;
                                    if (opCode === "add_diff" && (!st.model_c || st.model_c === "none")) {
                                        st.model_c = validSources[Math.min(2, validSources.length - 1)] || "m2";
                                    }
                                    saveStateToWidget();
                                    renderMergerUI();
                                },
                                onReattach: () => {
                                    delete st.overrides[sub.id];
                                    delete st.op_overrides[sub.id];
                                    saveStateToWidget();
                                    renderMergerUI();
                                },
                            });
                            treeScroll.appendChild(subRow);

                            if (canExpandSub && isSubExpanded) {
                                for (const leaf of sub.children) {
                                    const leafWDetached = st.overrides && st.overrides[leaf.id] !== undefined;
                                    const leafOpDetached =
                                        st.op_overrides && st.op_overrides[leaf.id] !== undefined;
                                    const leafDetached = leafWDetached || leafOpDetached;
                                    const leafVal = getEffectiveVal(st, grp.id, sub.id, leaf.id);
                                    const leafOp = getEffectiveOp(st, grp.id, sub.id, leaf.id);

                                    visibleRowIdsInOrder.push(leaf.id);

                                    const leafRow = createSliderRow({
                                        rowId: leaf.id,
                                        label: leaf.label,
                                        indentPx: 36,
                                        hasChildren: false,
                                        isExpanded: false,
                                        effectiveValue: leafVal,
                                        effectiveOp: leafOp,
                                        isDetached: leafDetached,
                                        isOpDetached: leafOpDetached,
                                        isSelected: this._mergerState.selectedRows.has(leaf.id),
                                        onChangeValue: (val, shouldRender) => {
                                            st.overrides[leaf.id] = val;
                                            saveStateToWidget();
                                            if (shouldRender) renderMergerUI();
                                        },
                                        onChangeOp: (opCode) => {
                                            st.op_overrides[leaf.id] = opCode;
                                            if (opCode === "add_diff" && (!st.model_c || st.model_c === "none")) {
                                                st.model_c =
                                                    validSources[Math.min(2, validSources.length - 1)] || "m2";
                                            }
                                            saveStateToWidget();
                                            renderMergerUI();
                                        },
                                        onReattach: () => {
                                            delete st.overrides[leaf.id];
                                            delete st.op_overrides[leaf.id];
                                            saveStateToWidget();
                                            renderMergerUI();
                                        },
                                    });
                                    treeScroll.appendChild(leafRow);
                                }
                            }
                        }
                    }
                }

                treeScroll.scrollTop = prevScrollTop;
            };

            this._renderMergerUI = renderMergerUI;
            renderMergerUI();
            this.setSize([680, 680]);
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
            if (message && this._mergerState) {
                let changed = false;
                if (message.detected_tier && message.detected_tier[0]) {
                    this._mergerState.blockCount = message.detected_tier[0];
                    changed = true;
                }
                if (message.leaf_keys && Array.isArray(message.leaf_keys[0]) && message.leaf_keys[0].length > 0) {
                    this._mergerState.customLeafKeys = message.leaf_keys[0];
                    changed = true;
                }
                if (message.model_names && message.model_names[0]) {
                    this._mergerState.backendModelNames = message.model_names[0];
                    changed = true;
                }
                if (message.stage_zero_fixes && message.stage_zero_fixes[0]) {
                    this._mergerState.stageZeroFixes = message.stage_zero_fixes[0];
                    changed = true;
                }
                if (changed && this._renderMergerUI) {
                    this._renderMergerUI();
                }
            }
        };
    },
});
