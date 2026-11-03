import gc
import json
import weakref
import torch
import comfy.model_management

from .loader_nodes import KittenLoRAModel
from .utils import (
    MAX_DYNAMIC_MODELS,
    ZERO_INIT_SUFFIXES,
    DynamicModelInputs,
    MergerSchemaPayload,
    convert_state_dict_to_tier,
    detect_block_tier,
    inspect_tensor_stats,
    materialize_patched_state_dict,
    natural_sort_key,
    parse_key_for_block,
    strip_diffusion_prefix,
    get_model_signature,
)

MAX_MERGE_STAGES = 32
MAX_TOTAL_MODEL_OUTPUTS = MAX_DYNAMIC_MODELS + MAX_MERGE_STAGES
ZERO_FIX_TOLERANCE = 1e-3

EMBEDDER_PREFIXES = (
    "pos_embedder.",
    "x_embedder.",
    "t_embedder.",
    "t_embedding_norm.",
)

ADAPTER_SUB_PREFIXES = (
    "llm_adapter.embed.",
    "llm_adapter.norm.",
    "llm_adapter.out_proj.",
    "llm_adapter.blocks.0.",
    "llm_adapter.blocks.1.",
    "llm_adapter.blocks.2.",
    "llm_adapter.blocks.3.",
    "llm_adapter.blocks.4.",
    "llm_adapter.blocks.5.",
)

BLOCK_SUB_LAYERS = (
    "adaln_modulation_self_attn",
    "adaln_modulation_cross_attn",
    "adaln_modulation_mlp",
    "self_attn",
    "cross_attn",
    "mlp",
)

FINAL_SUB_PREFIXES = (
    "final_layer.adaln_modulation.",
    "final_layer.linear.",
)

VALID_OPS = ("ws", "add", "sub", "add_diff", "cosine_a", "cosine_b")


class VirtualStageModel:
    """
    Lightweight placeholder for an intermediate stage (e.g. s1..s4) that has no downstream
    MODEL wires. Takes 0 MB of RAM while allowing ModelMergeInspector to evaluate it per-key.
    """
    def __init__(self, stage_id, display_name=None, is_lora=False):
        self._kitten_is_virtual_stage = True
        self._kitten_merged = True
        self._kitten_stage_id = stage_id
        self._kitten_source_id = None
        self._kitten_is_lora = is_lora
        self._kitten_display_name = display_name or stage_id
        self.patches = {}
        self.patches_uuid = stage_id
        self.model = self

    def state_dict(self):
        return {}

    def clone(self):
        return VirtualStageModel(
            self._kitten_stage_id,
            display_name=self._kitten_display_name,
            is_lora=self._kitten_is_lora,
        )


def resolve_key_hierarchy(clean_key):
    if clean_key.startswith("blocks."):
        parts = clean_key.split(".")
        if len(parts) >= 3:
            b_prefix = f"blocks.{parts[1]}."
            sub_name = parts[2]
            sub_prefix = f"{b_prefix}{sub_name}."
            return b_prefix, sub_prefix, clean_key
        if len(parts) >= 2:
            b_prefix = f"blocks.{parts[1]}."
            return b_prefix, None, clean_key

    if clean_key.startswith("llm_adapter."):
        for sub_p in ADAPTER_SUB_PREFIXES:
            if clean_key.startswith(sub_p):
                return "llm_adapter.", sub_p, clean_key
        return "llm_adapter.", None, clean_key

    for emb_p in EMBEDDER_PREFIXES:
        if clean_key.startswith(emb_p):
            return "embedders.", emb_p, clean_key

    if clean_key.startswith("final_layer."):
        for fin_p in FINAL_SUB_PREFIXES:
            if clean_key.startswith(fin_p):
                return "final_layer.", fin_p, clean_key
        return "final_layer.", None, clean_key

    return None, None, clean_key


def resolve_weight_and_op_for_key(clean_key, stage_data, runtime_overrides=None):
    overrides = stage_data.get("overrides") or {}
    op_overrides = stage_data.get("op_overrides") or {}
    global_ratio = float(stage_data.get("global_ratio", 0.5))
    global_op = stage_data.get("global_op", "ws")
    if global_op not in VALID_OPS:
        global_op = "ws"

    lvl1, lvl2, lvl3 = resolve_key_hierarchy(clean_key)

    w = global_ratio
    w_over = False

    if runtime_overrides:
        if lvl3 and lvl3 in runtime_overrides:
            w, w_over = float(runtime_overrides[lvl3]), True
        elif lvl2 and lvl2 in runtime_overrides:
            w, w_over = float(runtime_overrides[lvl2]), True
        elif lvl1 and lvl1 in runtime_overrides:
            w, w_over = float(runtime_overrides[lvl1]), True

    if not w_over:
        if lvl3 and lvl3 in overrides:
            w, w_over = float(overrides[lvl3]), True
        elif lvl2 and lvl2 in overrides:
            w, w_over = float(overrides[lvl2]), True
        elif lvl1 and lvl1 in overrides:
            w, w_over = float(overrides[lvl1]), True

    op = global_op
    op_over = False
    if lvl3 and lvl3 in op_overrides and op_overrides[lvl3] in VALID_OPS:
        op, op_over = op_overrides[lvl3], True
    elif lvl2 and lvl2 in op_overrides and op_overrides[lvl2] in VALID_OPS:
        op, op_over = op_overrides[lvl2], True
    elif lvl1 and lvl1 in op_overrides and op_overrides[lvl1] in VALID_OPS:
        op, op_over = op_overrides[lvl1], True

    return w, op, w_over, op_over


def get_cached_source_stats(src_id, sd, all_target_keys, stats_cache, device=None):
    if src_id in stats_cache:
        return stats_cache[src_id]

    key_norms = {}
    block_gates = {}
    block_all = {}
    sublayer_max = {}

    for k in all_target_keys:
        t = sd.get(k)
        w, _ = inspect_tensor_stats(t, device=device) if t is not None else (0.0, False)
        n = w or 0.0
        key_norms[k] = n

        clean_k = strip_diffusion_prefix(k)
        _, lvl2, _ = resolve_key_hierarchy(clean_k)
        if lvl2:
            sublayer_max[lvl2] = max(sublayer_max.get(lvl2, 0.0), n)

        _, b_idx, suffix = parse_key_for_block(k)
        if b_idx is not None:
            block_all.setdefault(b_idx, []).append(n)
            if suffix is not None and any(suffix.endswith(z) for z in ZERO_INIT_SUFFIXES):
                block_gates.setdefault(b_idx, []).append(n)

    entry = {
        "key_norms": key_norms,
        "block_gates": block_gates,
        "block_all": block_all,
        "sublayer_max": sublayer_max,
    }
    stats_cache[src_id] = entry
    return entry


def compute_stage_zero_fix_overrides(
    model_a_id,
    sd_a,
    model_b_id,
    sd_b,
    all_target_keys,
    stats_cache,
    is_a_lora=False,
    is_b_lora=False,
    device=None,
    zero_tol=ZERO_FIX_TOLERANCE,
):
    st_a = get_cached_source_stats(model_a_id, sd_a, all_target_keys, stats_cache, device=device)
    st_b = get_cached_source_stats(model_b_id, sd_b, all_target_keys, stats_cache, device=device)

    key_norms_a = st_a["key_norms"]
    key_norms_b = st_b["key_norms"]
    block_gates_a = st_a["block_gates"]
    block_gates_b = st_b["block_gates"]
    block_all_a = st_a["block_all"]
    block_all_b = st_b["block_all"]
    sublayer_max_a = st_a["sublayer_max"]
    sublayer_max_b = st_b["sublayer_max"]

    muted_blocks_a = set()
    muted_blocks_b = set()

    for b_idx, norms in block_all_a.items():
        if max(norms) <= zero_tol:
            muted_blocks_a.add(b_idx)
        elif not is_a_lora:
            gates = block_gates_a.get(b_idx, [])
            if gates and max(gates) <= zero_tol:
                muted_blocks_a.add(b_idx)

    for b_idx, norms in block_all_b.items():
        if max(norms) <= zero_tol:
            muted_blocks_b.add(b_idx)
        elif not is_b_lora:
            gates = block_gates_b.get(b_idx, [])
            if gates and max(gates) <= zero_tol:
                muted_blocks_b.add(b_idx)

    leaf_fixes = {}
    both_zero_leaves = set()
    lvl1_children = {}
    lvl2_children = {}

    for k in all_target_keys:
        clean_k = strip_diffusion_prefix(k)
        lvl1, lvl2, lvl3 = resolve_key_hierarchy(clean_k)
        if lvl1:
            lvl1_children.setdefault(lvl1, []).append(lvl3)
        if lvl2:
            lvl2_children.setdefault(lvl2, []).append(lvl3)

        _, b_idx, _ = parse_key_for_block(k)

        if is_a_lora:
            a_zero = (
                (b_idx is not None and b_idx in muted_blocks_a)
                or (lvl2 is not None and sublayer_max_a.get(lvl2, 0.0) <= zero_tol)
                or (k in sd_a and key_norms_a.get(k, 0.0) <= zero_tol)
            )
        else:
            a_zero = (key_norms_a.get(k, 0.0) <= zero_tol) or (b_idx is not None and b_idx in muted_blocks_a)

        if is_b_lora:
            b_zero = (
                (b_idx is not None and b_idx in muted_blocks_b)
                or (lvl2 is not None and sublayer_max_b.get(lvl2, 0.0) <= zero_tol)
                or (k in sd_b and key_norms_b.get(k, 0.0) <= zero_tol)
            )
        else:
            b_zero = (key_norms_b.get(k, 0.0) <= zero_tol) or (b_idx is not None and b_idx in muted_blocks_b)

        if a_zero and b_zero:
            both_zero_leaves.add(lvl3)
        elif b_zero and not a_zero:
            leaf_fixes[lvl3] = 1.0
        elif a_zero and not b_zero:
            leaf_fixes[lvl3] = 0.0

    if not leaf_fixes:
        return {}

    collapsed = dict(leaf_fixes)

    for lvl2, children in lvl2_children.items():
        active_children = [c for c in children if c not in both_zero_leaves]
        if not active_children:
            continue
        vals = {leaf_fixes.get(c) for c in active_children}
        if len(vals) == 1 and None not in vals:
            target_v = vals.pop()
            collapsed[lvl2] = target_v
            for c in children:
                collapsed.pop(c, None)

    for lvl1, children in lvl1_children.items():
        active_children = [c for c in children if c not in both_zero_leaves]
        if not active_children:
            continue
        vals = {leaf_fixes.get(c) for c in active_children}
        if len(vals) == 1 and None not in vals:
            target_v = vals.pop()
            collapsed[lvl1] = target_v
            for c in children:
                collapsed.pop(c, None)
                _, l2, _ = resolve_key_hierarchy(c)
                if l2:
                    collapsed.pop(l2, None)

    return collapsed


def compute_tensor_cosine_sim(ta, tb):
    a_flat = ta.reshape(-1)
    b_flat = tb.reshape(-1)
    norm_a = torch.linalg.norm(a_flat).item()
    norm_b = torch.linalg.norm(b_flat).item()
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    sim = torch.dot(a_flat, b_flat).item() / (norm_a * norm_b)
    return max(0.0, min(1.0, sim))


def combine_lineage_dicts(terms):
    combined = {}
    for coeff, lin in terms:
        if abs(coeff) <= 1e-9 or not lin:
            continue
        for src_m, src_w in lin.items():
            combined[src_m] = combined.get(src_m, 0.0) + coeff * src_w
    return {k: round(v, 6) for k, v in combined.items() if abs(v) > 1e-6}


def align_sd_keys_to_reference(sd, clean_to_ref_key):
    if not clean_to_ref_key:
        return sd
    aligned = {}
    for k, v in sd.items():
        clean_k = strip_diffusion_prefix(k)
        target_k = clean_to_ref_key.get(clean_k, k)
        aligned[target_k] = v
    return aligned


def eval_single_key_stage_recipe(ps, k, clean_k, b_idx, get_calc_tensor_fn, skipped_blocks_by_source):
    """
    Evaluates a single stage `ps` on a single key `k` on the GPU in `calc_dtype`.
    Returns (merged_gpu_tensor_or_None, is_exact_a, is_exact_b).
    """
    st = ps["st"]
    model_a_id = ps["model_a_id"]
    model_b_id = ps["model_b_id"]
    model_c_id = ps["model_c_id"]
    has_model_c = ps["has_model_c"]
    is_a_lora = ps["is_a_lora"]
    is_b_lora = ps["is_b_lora"]
    is_c_lora = ps["is_c_lora"]
    both_ab_lora = ps["both_ab_lora"]
    runtime_zero_ov = ps["runtime_zero_ov"]

    ta_calc = get_calc_tensor_fn(model_a_id)
    tb_calc = get_calc_tensor_fn(model_b_id)
    tc_calc = get_calc_tensor_fn(model_c_id) if has_model_c else None

    if ta_calc is None and tb_calc is None:
        return None, False, False

    if both_ab_lora:
        if ta_calc is None and tb_calc is not None:
            ta_calc = torch.zeros_like(tb_calc)
        elif tb_calc is None and ta_calc is not None:
            tb_calc = torch.zeros_like(ta_calc)
    else:
        if ta_calc is None and is_a_lora and tb_calc is not None:
            return tb_calc, False, True
        if tb_calc is None and is_b_lora and ta_calc is not None:
            return ta_calc, True, False
        if ta_calc is None:
            return tb_calc, False, True
        if tb_calc is None:
            return ta_calc, True, False

    if ta_calc.shape != tb_calc.shape or not ta_calc.is_floating_point():
        return ta_calc, True, False

    w, op, has_w_override, has_op_override = resolve_weight_and_op_for_key(
        clean_k, st, runtime_overrides=runtime_zero_ov
    )

    skip_a = skipped_blocks_by_source.get(model_a_id, set())
    skip_b = skipped_blocks_by_source.get(model_b_id, set())

    if b_idx is not None and not has_w_override and not has_op_override and op == "ws":
        b_in_a_skip = b_idx in skip_a
        b_in_b_skip = b_idx in skip_b
        if b_in_b_skip and not b_in_a_skip:
            w = 1.0
        elif b_in_a_skip and not b_in_b_skip:
            w = 0.0

    has_valid_c = has_model_c and (
        (tc_calc is not None and tc_calc.shape == ta_calc.shape) or is_c_lora
    )
    if has_valid_c and tc_calc is None:
        tc_calc = torch.zeros_like(ta_calc)

    if op == "ws":
        if not is_a_lora and is_b_lora:
            eff_b_w = 1.0 - float(w)
            if abs(eff_b_w) <= 1e-7:
                return ta_calc, True, False
            return ta_calc + eff_b_w * tb_calc, False, False
        elif is_a_lora and not is_b_lora:
            if abs(w) <= 1e-7:
                return tb_calc, False, True
            return tb_calc + float(w) * ta_calc, False, False
        else:
            if abs(w - 1.0) <= 1e-7:
                return ta_calc, True, False
            if abs(w) <= 1e-7:
                return tb_calc, False, True
            return torch.lerp(tb_calc, ta_calc, float(w)), False, False

    elif op == "add":
        if abs(w) <= 1e-7:
            return ta_calc, True, False
        return ta_calc + float(w) * tb_calc, False, False

    elif op == "sub":
        if abs(w) <= 1e-7:
            return ta_calc, True, False
        return ta_calc - float(w) * tb_calc, False, False

    elif op == "add_diff":
        if abs(w) <= 1e-7:
            return ta_calc, True, False
        if has_valid_c:
            return ta_calc + float(w) * (tb_calc - tc_calc), False, False
        return ta_calc + float(w) * tb_calc, False, False

    elif op in ("cosine_a", "cosine_b"):
        t_b = 1.0 - float(w)
        if abs(t_b) <= 1e-7:
            return ta_calc, True, False
        if has_valid_c:
            b_eff = ta_calc + (tb_calc - tc_calc)
        elif not is_a_lora and is_b_lora:
            b_eff = ta_calc + tb_calc
        else:
            b_eff = tb_calc

        sim = compute_tensor_cosine_sim(ta_calc, b_eff)
        cos_factor = (1.0 - sim) if op == "cosine_a" else sim
        k_blend = max(0.0, min(1.0, t_b * cos_factor))
        if abs(k_blend) <= 1e-7:
            return ta_calc, True, False
        return torch.lerp(ta_calc, b_eff, k_blend), False, False

    return ta_calc, True, False


class AnimaModelMerger:
    _allocated_patchers = []
    _allocated_schemas = []
    _last_merge_sig = None
    _last_merge_result = None
    _source_stats_cache = {}

    @classmethod
    def _evict_previous_allocations(cls, active_input_mps):
        active_ids = {id(mp) for mp in active_input_mps if mp is not None}
        for ref in cls._allocated_patchers:
            mp = ref()
            if mp is not None and id(mp) not in active_ids:
                if hasattr(mp, "patches") and isinstance(mp.patches, dict):
                    mp.patches.clear()
                if hasattr(mp, "_precomputed_sd"):
                    mp._precomputed_sd = None
                if hasattr(mp, "_cached_fp32_sd"):
                    mp._cached_fp32_sd = None
        cls._allocated_patchers.clear()

        for ref in cls._allocated_schemas:
            sch = ref()
            if sch is not None:
                if hasattr(sch, "_kitten_bundled_models"):
                    sch._kitten_bundled_models.clear()
                if hasattr(sch, "_kitten_stage_recipes"):
                    sch._kitten_stage_recipes.clear()
                if hasattr(sch, "_kitten_passthrough_models"):
                    sch._kitten_passthrough_models.clear()
        cls._allocated_schemas.clear()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_tier": (
                    [
                        "match_model_1",
                        "match_model_2",
                        "force_28_blocks (2B)",
                        "force_40_blocks (2.9B)",
                    ],
                    {
                        "default": "match_model_1",
                        "tooltip": "Target block architecture. Any 3.8B (52-block) input is automatically downscaled.",
                    },
                ),
                "downscale_absorb": (
                    ["auto", "surgical_0.0", "0.10", "0.15", "0.20", "0.25"],
                    {
                        "default": "auto",
                        "tooltip": "Woodbury subspace absorption when downscaling 52->40 or 40->28 blocks.",
                    },
                ),
                "upscale_fill_mode": (
                    ["skip_inserted", "zero_gates", "interpolate_neighbors"],
                    {
                        "default": "skip_inserted",
                        "tooltip": "How 12 inserted blocks are handled when upscaling 28->40 blocks.",
                    },
                ),
                "neighbor_ratio": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Front vs. back neighbor split ratio during block scaling.",
                    },
                ),
                "precision": (
                    ["fp32", "fp64", "bf16"],
                    {
                        "default": "fp32",
                        "tooltip": "Intermediate floating-point precision for merge calculations and connected Anima LoRA Loaders.",
                    },
                ),
                "recipe_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
            },
            "optional": DynamicModelInputs({
                "model_1": ("MODEL",),
            }),
        }

    RETURN_TYPES = ("STRING",) + ("MODEL",) * MAX_TOTAL_MODEL_OUTPUTS
    RETURN_NAMES = (
        ("merge_schema",)
        + tuple(f"model_{i}" for i in range(1, MAX_DYNAMIC_MODELS + 1))
        + tuple(f"model_s{i}" for i in range(1, MAX_MERGE_STAGES + 1))
    )
    FUNCTION = "merge_models"
    OUTPUT_NODE = True
    CATEGORY = "kitten_merger"

    @classmethod
    def IS_CHANGED(
        cls,
        target_tier="match_model_1",
        downscale_absorb="auto",
        upscale_fill_mode="skip_inserted",
        neighbor_ratio=0.5,
        precision="fp32",
        recipe_json="",
        **kwargs,
    ):
        functional_stages = []
        connected_stages = []
        if recipe_json and recipe_json.strip():
            try:
                parsed = json.loads(recipe_json)
                if isinstance(parsed, dict):
                    connected_stages = sorted(parsed.get("connectedStageIds") or [])
                    for s in parsed.get("stages") or []:
                        functional_stages.append({
                            "id": s.get("id"),
                            "name": s.get("name", ""),
                            "enabled": bool(s.get("enabled", True)),
                            "model_a": s.get("model_a") or s.get("base"),
                            "model_b": s.get("model_b") or s.get("donor"),
                            "model_c": s.get("model_c", "none"),
                            "global_op": s.get("global_op", "ws"),
                            "global_ratio": s.get("global_ratio", 0.5),
                            "fix_zeros": bool(s.get("fix_zeros", False)),
                            "overrides": s.get("overrides") or {},
                            "op_overrides": s.get("op_overrides") or {},
                        })
            except Exception:
                functional_stages = [recipe_json]

        return json.dumps(
            {
                "t": target_tier,
                "da": downscale_absorb,
                "uf": upscale_fill_mode,
                "nr": round(float(neighbor_ratio), 4),
                "p": precision,
                "cs": connected_stages,
                "st": functional_stages,
            },
            sort_keys=True,
        )

    def _determine_target_tier(self, target_tier_mode, tiers_by_id, m_ids):
        if target_tier_mode == "force_28_blocks (2B)":
            return 28
        if target_tier_mode == "force_40_blocks (2.9B)":
            return 40
        if target_tier_mode == "match_model_2" and "m2" in tiers_by_id:
            t = tiers_by_id["m2"]
            return 40 if t == 52 else (t or 40)

        first_id = "m1" if "m1" in tiers_by_id else m_ids[0]
        t = tiers_by_id.get(first_id, 40)
        return 40 if (t is None or t >= 40) else 28

    def _find_downstream_wired_stages(self, prompt, unique_id, json_model_output_count, stage_ids_in_order, recipe_connected_stages):
        """
        Determines which stages have downstream MODEL wires connected (either directly on
        AnimaModelMerger or on a connected ModelMergeInspector).
        """
        needed = set(recipe_connected_stages or [])

        if not prompt or not unique_id:
            return needed

        my_id = str(unique_id)
        inspector_node_ids = set()

        for node_id, node_info in prompt.items():
            if not isinstance(node_info, dict):
                continue
            inputs = node_info.get("inputs") or {}
            is_inspector = node_info.get("class_type") == "ModelMergeInspector"

            for inp_name, link_val in inputs.items():
                if not (isinstance(link_val, list) and len(link_val) == 2):
                    continue
                orig_id, orig_slot = str(link_val[0]), int(link_val[1])
                if orig_id != my_id:
                    continue

                if orig_slot == 0 and is_inspector:
                    inspector_node_ids.add(str(node_id))
                elif orig_slot > json_model_output_count:
                    st_idx = orig_slot - 1 - json_model_output_count
                    if 0 <= st_idx < len(stage_ids_in_order):
                        needed.add(stage_ids_in_order[st_idx])

        # Also check which stage output slots on any connected ModelMergeInspector are wired downstream
        if inspector_node_ids:
            for node_id, node_info in prompt.items():
                if not isinstance(node_info, dict):
                    continue
                inputs = node_info.get("inputs") or {}
                for inp_name, link_val in inputs.items():
                    if not (isinstance(link_val, list) and len(link_val) == 2):
                        continue
                    orig_id, orig_slot = str(link_val[0]), int(link_val[1])
                    if orig_id in inspector_node_ids and orig_slot > json_model_output_count:
                        st_idx = orig_slot - 1 - json_model_output_count
                        if 0 <= st_idx < len(stage_ids_in_order):
                            needed.add(stage_ids_in_order[st_idx])

        return needed

    def _find_downstream_wired_passthrough_models(self, prompt, unique_id, json_model_output_count, m_ids):
        """Returns base model IDs (m1..mN) whose passthrough output slots are wired downstream."""
        if not prompt or not unique_id:
            return set(m_ids)

        my_id = str(unique_id)
        wired_m = set()
        inspector_node_ids = set()

        for node_id, node_info in prompt.items():
            if not isinstance(node_info, dict):
                continue
            inputs = node_info.get("inputs") or {}
            is_inspector = node_info.get("class_type") == "ModelMergeInspector"
            for inp_name, link_val in inputs.items():
                if not (isinstance(link_val, list) and len(link_val) == 2):
                    continue
                orig_id, orig_slot = str(link_val[0]), int(link_val[1])
                if orig_id != my_id:
                    continue
                if orig_slot == 0 and is_inspector:
                    inspector_node_ids.add(str(node_id))
                elif 1 <= orig_slot <= json_model_output_count:
                    m_id = f"m{orig_slot}"
                    if m_id in m_ids:
                        wired_m.add(m_id)

        if inspector_node_ids:
            for node_id, node_info in prompt.items():
                if not isinstance(node_info, dict):
                    continue
                inputs = node_info.get("inputs") or {}
                for inp_name, link_val in inputs.items():
                    if not (isinstance(link_val, list) and len(link_val) == 2):
                        continue
                    orig_id, orig_slot = str(link_val[0]), int(link_val[1])
                    if orig_id in inspector_node_ids and 1 <= orig_slot <= json_model_output_count:
                        m_id = f"m{orig_slot}"
                        if m_id in m_ids:
                            wired_m.add(m_id)

        return wired_m

    def _build_model_patcher_from_sd(
        self,
        target_sd,
        template_mp,
        merged_flag=False,
        stage_id=None,
        source_id=None,
        display_name=None,
    ):
        if template_mp is None or getattr(template_mp, "_kitten_is_lora", False):
            lora_out = KittenLoRAModel(
                precomputed_sd={k: v.to(dtype=torch.bfloat16).cpu() for k, v in target_sd.items()},
                strength=1.0,
                name=display_name or stage_id or source_id or "merged_lora",
            )
            lora_out._kitten_merged = merged_flag
            lora_out._kitten_stage_id = stage_id
            lora_out._kitten_source_id = source_id
            AnimaModelMerger._allocated_patchers.append(weakref.ref(lora_out))
            return lora_out

        mp = template_mp.clone()
        mp._kitten_merged = merged_flag
        if stage_id is not None:
            mp._kitten_stage_id = stage_id
        if source_id is not None:
            mp._kitten_source_id = source_id
        if display_name is not None:
            mp._kitten_display_name = display_name

        template_base_sd = mp.model.state_dict()
        patches_to_set = {}

        for k, final_t in target_sd.items():
            if k not in template_base_sd:
                continue
            orig_base_t = template_base_sd[k]
            if orig_base_t.shape != final_t.shape:
                continue
            mp.patches.pop(k, None)
            if final_t is orig_base_t:
                continue
            final_cpu = (
                final_t
                if (final_t.device.type == "cpu" and final_t.dtype == orig_base_t.dtype)
                else final_t.to(device="cpu", dtype=orig_base_t.dtype).contiguous()
            )
            patches_to_set[k] = ("diff", (final_cpu,))

        if patches_to_set:
            mp.add_patches(patches_to_set, strength_patch=1.0, strength_model=0.0)

        AnimaModelMerger._allocated_patchers.append(weakref.ref(mp))
        return mp

    def merge_models(
        self,
        target_tier="match_model_1",
        downscale_absorb="auto",
        upscale_fill_mode="skip_inserted",
        neighbor_ratio=0.5,
        precision="fp32",
        recipe_json="",
        prompt=None,
        unique_id=None,
        **kwargs,
    ):
        models_dict = {}
        model_names_by_id = {}
        highest_connected_idx = 1
        for i in range(1, MAX_DYNAMIC_MODELS + 1):
            mp = kwargs.get(f"model_{i}")
            if mp is not None:
                m_id = f"m{i}"
                models_dict[m_id] = mp
                disp = getattr(mp, "_kitten_display_name", None)
                if disp:
                    model_names_by_id[m_id] = disp
                highest_connected_idx = max(highest_connected_idx, i)

        if not models_dict:
            raise ValueError("AnimaModelMerger: Connect at least model_1.")

        merge_sig = (
            tuple((m, get_model_signature(models_dict[m])) for m in sorted(models_dict.keys())),
            self.IS_CHANGED(
                target_tier=target_tier,
                downscale_absorb=downscale_absorb,
                upscale_fill_mode=upscale_fill_mode,
                neighbor_ratio=neighbor_ratio,
                precision=precision,
                recipe_json=recipe_json,
            ),
        )
        if AnimaModelMerger._last_merge_sig == merge_sig and AnimaModelMerger._last_merge_result is not None:
            return AnimaModelMerger._last_merge_result

        # Evict previous run's stage tensors before allocating anything for the new run
        self._evict_previous_allocations(models_dict.values())
        gc.collect()
        comfy.model_management.soft_empty_cache()

        m_ids = list(models_dict.keys())
        device = comfy.model_management.get_torch_device()
        calc_dtype = {
            "fp64": torch.float64,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }.get(precision, torch.float32)

        stages = []
        json_model_output_count = highest_connected_idx
        recipe_connected_stages = []
        if recipe_json and recipe_json.strip():
            try:
                parsed = json.loads(recipe_json)
                if isinstance(parsed, dict):
                    if isinstance(parsed.get("stages"), list):
                        stages = parsed["stages"]
                    if isinstance(parsed.get("modelOutputCount"), int) and parsed["modelOutputCount"] >= 1:
                        json_model_output_count = max(highest_connected_idx, parsed["modelOutputCount"])
                    if isinstance(parsed.get("connectedStageIds"), list):
                        recipe_connected_stages = parsed["connectedStageIds"]
            except json.JSONDecodeError:
                stages = []

        if not stages:
            prev_src = m_ids[0]
            if len(m_ids) == 1:
                stages.append({
                    "id": "s1",
                    "enabled": True,
                    "model_a": prev_src,
                    "model_b": prev_src,
                    "model_c": "none",
                    "global_op": "ws",
                    "global_ratio": 1.0,
                    "fix_zeros": False,
                    "overrides": {},
                    "op_overrides": {},
                })
            else:
                for idx, next_m in enumerate(m_ids[1:], start=1):
                    s_id = f"s{idx}"
                    stages.append({
                        "id": s_id,
                        "enabled": True,
                        "model_a": prev_src,
                        "model_b": next_m,
                        "model_c": "none",
                        "global_op": "ws",
                        "global_ratio": 0.5,
                        "fix_zeros": False,
                        "overrides": {},
                        "op_overrides": {},
                    })
                    prev_src = s_id

        stage_ids_in_order = []
        stage_meta_by_id = {}
        known_sources = set(m_ids)
        last_stage_id = m_ids[0]

        for idx, st in enumerate(stages, start=1):
            s_id = st.get("id") or f"s{idx}"
            stage_ids_in_order.append(s_id)
            model_a_id = st.get("model_a") or st.get("base") or m_ids[0]
            model_b_id = st.get("model_b") or st.get("donor") or m_ids[min(1, len(m_ids) - 1)]
            model_c_id = st.get("model_c") or "none"

            if model_a_id not in known_sources:
                model_a_id = last_stage_id
            if model_b_id not in known_sources:
                model_b_id = m_ids[-1]
            has_model_c = model_c_id != "none" and model_c_id in known_sources

            stage_meta_by_id[s_id] = {
                "st": st,
                "enabled": bool(st.get("enabled", True)),
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
                "model_c_id": model_c_id,
                "has_model_c": has_model_c,
            }
            known_sources.add(s_id)
            last_stage_id = s_id

        wired_stage_ids = self._find_downstream_wired_stages(
            prompt,
            unique_id,
            json_model_output_count,
            stage_ids_in_order,
            recipe_connected_stages,
        )
        wired_passthrough_m_ids = self._find_downstream_wired_passthrough_models(
            prompt,
            unique_id,
            json_model_output_count,
            m_ids,
        )

        # A stage is needed if it is enabled OR wired downstream, plus any parent stages it depends on
        active_stage_ids = {
            s_id for s_id in stage_ids_in_order
            if stage_meta_by_id[s_id]["enabled"] or s_id in wired_stage_ids
        }
        queue = list(active_stage_ids)
        while queue:
            cur_s = queue.pop()
            meta = stage_meta_by_id.get(cur_s)
            if not meta:
                continue
            deps = [meta["model_a_id"], meta["model_b_id"]]
            if meta["has_model_c"]:
                deps.append(meta["model_c_id"])
            for dep in deps:
                if dep in stage_meta_by_id and dep not in active_stage_ids:
                    active_stage_ids.add(dep)
                    queue.append(dep)

        # Materialize full CPU ModelPatchers for wired stages and the last enabled stage
        materialized_stage_ids = set(wired_stage_ids)
        enabled_in_order = [s_id for s_id in stage_ids_in_order if s_id in active_stage_ids]
        if enabled_in_order:
            materialized_stage_ids.add(enabled_in_order[-1])

        # Determine which base models (m1..mN) are actually needed
        active_m_ids = set(wired_passthrough_m_ids)
        for s_id in active_stage_ids:
            meta = stage_meta_by_id[s_id]
            for dep in (meta["model_a_id"], meta["model_b_id"], meta["model_c_id"] if meta["has_model_c"] else None):
                if dep in models_dict:
                    active_m_ids.add(dep)

        # If all stages are disabled (passthrough mode), keep all connected base models active
        if not active_stage_ids:
            active_m_ids = set(m_ids)
        if target_tier == "match_model_1" and "m1" in models_dict:
            active_m_ids.add("m1")
        elif target_tier == "match_model_2" and "m2" in models_dict:
            active_m_ids.add("m2")
        if not active_m_ids:
            active_m_ids.add(m_ids[0])

        raw_sds = {}
        tiers_by_id = {}
        source_is_lora = {}

        for m in m_ids:
            mp = models_dict[m]
            is_lora = bool(getattr(mp, "_kitten_is_lora", False))
            source_is_lora[m] = is_lora
            if m not in active_m_ids:
                continue
            if is_lora:
                raw_sds[m] = mp.materialize_state_dict(dtype=torch.bfloat16, device=device)
            else:
                raw_sds[m] = materialize_patched_state_dict(mp, device=device)
            tier, _ = detect_block_tier(raw_sds[m].keys())
            tiers_by_id[m] = tier

        active_m_list = [m for m in m_ids if m in active_m_ids]
        resolved_tier = self._determine_target_tier(target_tier, tiers_by_id, active_m_list)

        non_lora_ids = [m for m in active_m_list if not source_is_lora[m]]
        template_m_id = None
        for m in non_lora_ids:
            if tiers_by_id.get(m) == resolved_tier:
                template_m_id = m
                break
        if template_m_id is None and non_lora_ids:
            template_m_id = non_lora_ids[0]
            resolved_tier = tiers_by_id[template_m_id] if tiers_by_id.get(template_m_id) in (28, 40) else 40
        if template_m_id is None:
            template_m_id = active_m_list[0]

        template_mp = models_dict[template_m_id]

        aligned_sds = {}
        skipped_blocks_by_source = {}

        for m in active_m_list:
            orig_tier = tiers_by_id[m]
            conv_sd, skipped_set = convert_state_dict_to_tier(
                raw_sds.pop(m),
                orig_tier,
                resolved_tier,
                absorb_mode=downscale_absorb,
                upscale_fill_mode=upscale_fill_mode,
                ratio=neighbor_ratio,
                device=device,
                is_lora=source_is_lora[m],
            )
            aligned_sds[m] = conv_sd
            skipped_blocks_by_source[m] = skipped_set

        del raw_sds

        clean_to_ref_key = {}
        for k in aligned_sds[template_m_id].keys():
            clean_to_ref_key[strip_diffusion_prefix(k)] = k
        for m in active_m_list:
            for k in aligned_sds[m].keys():
                ck = strip_diffusion_prefix(k)
                if ck not in clean_to_ref_key:
                    clean_to_ref_key[ck] = f"diffusion_model.{ck}"
            aligned_sds[m] = align_sd_keys_to_reference(aligned_sds[m], clean_to_ref_key)

        union_target_keys = set()
        for m in active_m_list:
            union_target_keys |= set(aligned_sds[m].keys())
        all_target_keys = sorted(union_target_keys, key=natural_sort_key)
        clean_leaf_keys = [strip_diffusion_prefix(k) for k in all_target_keys]

        sources_sd = dict(aligned_sds)
        sources_lineage = {
            m: {k: {m: 1.0} for k in aligned_sds[m].keys()}
            for m in active_m_list
        }

        stage_summaries = []
        stage_summaries_by_id = {}
        stage_zero_fixes = {}
        stats_cache = AnimaModelMerger._source_stats_cache

        prepared_stages = []

        for s_id in stage_ids_in_order:
            meta = stage_meta_by_id[s_id]
            st = meta["st"]
            model_a_id = meta["model_a_id"]
            model_b_id = meta["model_b_id"]
            model_c_id = meta["model_c_id"]
            has_model_c = meta["has_model_c"]

            is_a_lora = source_is_lora.get(model_a_id, False)
            is_b_lora = source_is_lora.get(model_b_id, False)
            is_c_lora = source_is_lora.get(model_c_id, False) if has_model_c else False
            both_ab_lora = is_a_lora and is_b_lora
            out_stage_is_lora = both_ab_lora and (not has_model_c or is_c_lora)
            source_is_lora[s_id] = out_stage_is_lora

            if s_id not in active_stage_ids:
                continue

            sources_sd[s_id] = {}
            sources_lineage[s_id] = {}
            skipped_blocks_by_source[s_id] = set()

            if st.get("fix_zeros", False) and model_a_id in aligned_sds and model_b_id in aligned_sds:
                auto_zero_ov = compute_stage_zero_fix_overrides(
                    model_a_id,
                    aligned_sds[model_a_id],
                    model_b_id,
                    aligned_sds[model_b_id],
                    all_target_keys,
                    stats_cache,
                    is_a_lora=is_a_lora,
                    is_b_lora=is_b_lora,
                    device=device,
                )
                stage_zero_fixes[s_id] = auto_zero_ov
                runtime_zero_ov = auto_zero_ov
            else:
                runtime_zero_ov = None

            g_op = st.get("global_op", "ws")
            g_w = float(st.get("global_ratio", 0.5))
            ov_map = st.get("overrides") or {}
            op_ov_map = st.get("op_overrides") or {}
            total_ov_count = len(set(ov_map.keys()) | set(op_ov_map.keys()) | set((runtime_zero_ov or {}).keys()))
            ov_tag = f" [{total_ov_count} ov]" if total_ov_count > 0 else ""
            fz_tag = " [FZ]" if st.get("fix_zeros", False) else ""

            if total_ov_count > 0 and g_op == "ws":
                all_w_vals = [g_w] + [float(v) for v in ov_map.values()] + [float(v) for v in (runtime_zero_ov or {}).values()]
                min_w, max_w = min(all_w_vals), max(all_w_vals)
                w_disp = f"{min_w:.2f}..{max_w:.2f}" if abs(max_w - min_w) > 1e-3 else f"{g_w:.2f}"
                summary_str = f"{s_id} = ({w_disp})*{model_a_id} + (1-{w_disp})*{model_b_id}{ov_tag}{fz_tag}"
            elif g_op == "add":
                summary_str = f"{s_id} = {model_a_id} + {g_w:.2f}*{model_b_id}{ov_tag}{fz_tag}"
            elif g_op == "sub":
                summary_str = f"{s_id} = {model_a_id} - {g_w:.2f}*{model_b_id}{ov_tag}{fz_tag}"
            elif g_op == "add_diff":
                summary_str = f"{s_id} = {model_a_id} + {g_w:.2f}*({model_b_id} - {model_c_id}){ov_tag}{fz_tag}"
            elif g_op in ("cosine_a", "cosine_b"):
                c_sfx = f", C={model_c_id}" if has_model_c else ""
                summary_str = f"{s_id} = {g_op}({model_a_id}, {model_b_id}{c_sfx}, w={g_w:.2f}){ov_tag}{fz_tag}"
            else:
                summary_str = f"{s_id} = {g_w:.2f}*{model_a_id} + {(1.0 - g_w):.2f}*{model_b_id}{ov_tag}{fz_tag}"

            stage_summaries.append(summary_str)
            stage_summaries_by_id[s_id] = summary_str

            prepared_stages.append({
                "s_id": s_id,
                "st": st,
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
                "model_c_id": model_c_id,
                "has_model_c": has_model_c,
                "is_a_lora": is_a_lora,
                "is_b_lora": is_b_lora,
                "is_c_lora": is_c_lora,
                "both_ab_lora": both_ab_lora,
                "out_stage_is_lora": out_stage_is_lora,
                "runtime_zero_ov": runtime_zero_ov,
            })

        for k in (all_target_keys if prepared_stages else ()):
            clean_k = strip_diffusion_prefix(k)
            _, b_idx, _ = parse_key_for_block(k)

            gpu_calc_tensors = {}
            cpu_exact_refs = {m: aligned_sds[m].get(k) for m in active_m_list}

            def get_calc_tensor(src_id):
                if src_id in gpu_calc_tensors:
                    return gpu_calc_tensors[src_id]
                cpu_t = cpu_exact_refs.get(src_id)
                if cpu_t is None and src_id in sources_sd:
                    cpu_t = sources_sd[src_id].get(k)
                if cpu_t is None:
                    return None
                gt = cpu_t.to(device=device, dtype=calc_dtype)
                gpu_calc_tensors[src_id] = gt
                return gt

            for ps in prepared_stages:
                s_id = ps["s_id"]
                st = ps["st"]
                model_a_id = ps["model_a_id"]
                model_b_id = ps["model_b_id"]
                model_c_id = ps["model_c_id"]
                has_model_c = ps["has_model_c"]
                is_a_lora = ps["is_a_lora"]
                is_b_lora = ps["is_b_lora"]
                is_c_lora = ps["is_c_lora"]
                both_ab_lora = ps["both_ab_lora"]
                out_stage_is_lora = ps["out_stage_is_lora"]
                runtime_zero_ov = ps["runtime_zero_ov"]
                should_store_cpu = s_id in materialized_stage_ids

                lin_a = sources_lineage[model_a_id].get(k, {})
                lin_b = sources_lineage[model_b_id].get(k, {})
                lin_c = sources_lineage[model_c_id].get(k, {}) if has_model_c else None

                w, op, has_w_override, has_op_override = resolve_weight_and_op_for_key(
                    clean_k, st, runtime_overrides=runtime_zero_ov
                )

                skip_a = skipped_blocks_by_source.get(model_a_id, set())
                skip_b = skipped_blocks_by_source.get(model_b_id, set())
                if b_idx is not None and not has_w_override and not has_op_override and op == "ws":
                    b_in_a_skip = b_idx in skip_a
                    b_in_b_skip = b_idx in skip_b
                    if b_in_b_skip and not b_in_a_skip:
                        w = 1.0
                    elif b_in_a_skip and not b_in_b_skip:
                        w = 0.0
                    elif b_in_a_skip and b_in_b_skip:
                        skipped_blocks_by_source[s_id].add(b_idx)

                # Zero-copy fast path when weight is 100% Model A or 100% Model B (no GPU transfer needed!)
                ref_a = cpu_exact_refs.get(model_a_id)
                ref_b = cpu_exact_refs.get(model_b_id)

                if op == "ws" and not both_ab_lora and not is_a_lora and not is_b_lora:
                    if abs(w - 1.0) <= 1e-7 and ref_a is not None:
                        cpu_exact_refs[s_id] = ref_a
                        if model_a_id in gpu_calc_tensors:
                            gpu_calc_tensors[s_id] = gpu_calc_tensors[model_a_id]
                        if should_store_cpu:
                            sources_sd[s_id][k] = ref_a
                        sources_lineage[s_id][k] = dict(lin_a)
                        continue
                    if abs(w) <= 1e-7 and ref_b is not None:
                        cpu_exact_refs[s_id] = ref_b
                        if model_b_id in gpu_calc_tensors:
                            gpu_calc_tensors[s_id] = gpu_calc_tensors[model_b_id]
                        if should_store_cpu:
                            sources_sd[s_id][k] = ref_b
                        sources_lineage[s_id][k] = dict(lin_b)
                        continue

                merged_gpu, is_exact_a, is_exact_b = eval_single_key_stage_recipe(
                    ps, k, clean_k, b_idx, get_calc_tensor, skipped_blocks_by_source
                )
                if merged_gpu is None:
                    continue

                gpu_calc_tensors[s_id] = merged_gpu
                store_dtype = calc_dtype if not out_stage_is_lora else torch.bfloat16

                if is_exact_a and ref_a is not None and ref_a.dtype == store_dtype:
                    cpu_exact_refs[s_id] = ref_a
                    if should_store_cpu:
                        sources_sd[s_id][k] = ref_a
                elif is_exact_b and ref_b is not None and ref_b.dtype == store_dtype:
                    cpu_exact_refs[s_id] = ref_b
                    if should_store_cpu:
                        sources_sd[s_id][k] = ref_b
                elif should_store_cpu:
                    cpu_t = merged_gpu.to(device="cpu", dtype=store_dtype).contiguous()
                    cpu_exact_refs[s_id] = cpu_t
                    sources_sd[s_id][k] = cpu_t

                # Record lineage for merge_schema
                if is_exact_a:
                    sources_lineage[s_id][k] = dict(lin_a)
                elif is_exact_b:
                    sources_lineage[s_id][k] = dict(lin_b)
                elif op == "ws":
                    if not is_a_lora and is_b_lora:
                        sources_lineage[s_id][k] = combine_lineage_dicts([(1.0, lin_a), (1.0 - w, lin_b)])
                    elif is_a_lora and not is_b_lora:
                        sources_lineage[s_id][k] = combine_lineage_dicts([(w, lin_a), (1.0, lin_b)])
                    else:
                        sources_lineage[s_id][k] = combine_lineage_dicts([(w, lin_a), (1.0 - w, lin_b)])
                elif op == "add":
                    sources_lineage[s_id][k] = combine_lineage_dicts([(1.0, lin_a), (w, lin_b)])
                elif op == "sub":
                    sources_lineage[s_id][k] = combine_lineage_dicts([(1.0, lin_a), (-w, lin_b)])
                elif op == "add_diff":
                    if lin_c is not None:
                        sources_lineage[s_id][k] = combine_lineage_dicts([(1.0, lin_a), (w, lin_b), (-w, lin_c)])
                    else:
                        sources_lineage[s_id][k] = combine_lineage_dicts([(1.0, lin_a), (w, lin_b)])
                elif op in ("cosine_a", "cosine_b"):
                    sources_lineage[s_id][k] = combine_lineage_dicts([(w, lin_a), (1.0 - w, lin_b)])

            gpu_calc_tensors.clear()
            cpu_exact_refs.clear()

        passthrough_mps = {}
        for i in range(1, json_model_output_count + 1):
            m_id = f"m{i}"
            if m_id not in models_dict:
                passthrough_mps[m_id] = None
            elif m_id not in aligned_sds:
                cloned = models_dict[m_id].clone()
                cloned._kitten_merged = False
                cloned._kitten_source_id = m_id
                cloned._kitten_display_name = getattr(models_dict[m_id], "_kitten_display_name", None)
                passthrough_mps[m_id] = cloned
            elif source_is_lora[m_id]:
                passthrough_mps[m_id] = self._build_model_patcher_from_sd(
                    aligned_sds[m_id],
                    None,
                    merged_flag=False,
                    source_id=m_id,
                    display_name=getattr(models_dict[m_id], "_kitten_display_name", m_id),
                )
            elif tiers_by_id.get(m_id) == resolved_tier:
                cloned = models_dict[m_id].clone()
                cloned._kitten_merged = False
                cloned._kitten_source_id = m_id
                cloned._kitten_display_name = getattr(models_dict[m_id], "_kitten_display_name", None)
                passthrough_mps[m_id] = cloned
            else:
                passthrough_mps[m_id] = self._build_model_patcher_from_sd(
                    aligned_sds[m_id],
                    template_mp,
                    merged_flag=False,
                    source_id=m_id,
                    display_name=getattr(models_dict[m_id], "_kitten_display_name", None),
                )

        stage_custom_names = {
            s_id: (stage_meta_by_id[s_id]["st"].get("name") or "").strip()
            for s_id in stage_ids_in_order
        }
        stage_mps = {}
        virtual_stage_mps = {}
        for s_id in stage_ids_in_order:
            disp_name = stage_custom_names.get(s_id) or s_id
            stage_sd = sources_sd.pop(s_id, {})
            if s_id not in active_stage_ids:
                stage_mps[s_id] = None
                virtual_stage_mps[s_id] = None
            elif s_id in materialized_stage_ids and stage_sd:
                stage_template = None if source_is_lora.get(s_id, False) else template_mp
                mp_out = self._build_model_patcher_from_sd(
                    stage_sd,
                    stage_template,
                    merged_flag=True,
                    stage_id=s_id,
                    display_name=disp_name,
                )
                stage_mps[s_id] = mp_out
                virtual_stage_mps[s_id] = mp_out
            else:
                stage_mps[s_id] = None
                virtual_stage_mps[s_id] = VirtualStageModel(
                    s_id,
                    display_name=disp_name,
                    is_lora=source_is_lora.get(s_id, False),
                )
            del stage_sd

        sources_sd.clear()
        aligned_sds.clear()
        gc.collect()
        comfy.model_management.soft_empty_cache()

        palette = []
        palette_map = {}
        stage_key_maps = {}

        for s_id in enabled_in_order:
            s_kmap = {}
            for k, lin_dict in sources_lineage.get(s_id, {}).items():
                sig_str = json.dumps(lin_dict, sort_keys=True)
                if sig_str not in palette_map:
                    palette_map[sig_str] = len(palette)
                    palette.append(lin_dict)
                s_kmap[k] = palette_map[sig_str]
            stage_key_maps[s_id] = s_kmap

        # Bundle only active models and active stages for ModelMergeInspector
        inspector_m_ids = set()
        for s_id in enabled_in_order:
            meta = stage_meta_by_id[s_id]
            for dep in (meta["model_a_id"], meta["model_b_id"], meta["model_c_id"] if meta["has_model_c"] else None):
                if dep in models_dict:
                    inspector_m_ids.add(dep)
        if not inspector_m_ids:
            inspector_m_ids = set(active_m_list)

        bundled_models = []
        for i in range(1, json_model_output_count + 1):
            m_id = f"m{i}"
            if m_id in inspector_m_ids and passthrough_mps.get(m_id) is not None:
                bundled_models.append((m_id, passthrough_mps[m_id]))
        for s_id in enabled_in_order:
            if virtual_stage_mps.get(s_id) is not None:
                bundled_models.append((s_id, virtual_stage_mps[s_id]))

        output_slot_order = [f"m{i}" for i in range(1, json_model_output_count + 1)] + list(stage_ids_in_order)
        all_passthrough_and_stages = dict(passthrough_mps)
        all_passthrough_and_stages.update(stage_mps)

        last_active_stage_id = enabled_in_order[-1] if enabled_in_order else None
        schema_payload = MergerSchemaPayload(
            json.dumps({
                "kitten_merge_schema": 1,
                "target_tier": resolved_tier,
                "stages_summary": " | ".join(stage_summaries) if stage_summaries else "passthrough",
                "stage_summaries_by_id": stage_summaries_by_id,
                "palette": palette,
                "key_map": stage_key_maps.get(last_active_stage_id, {}) if last_active_stage_id else {},
                "stage_key_maps": stage_key_maps,
            }),
            bundled_models=bundled_models,
            stage_recipes=prepared_stages,
            is_lora_map=source_is_lora,
            output_slot_order=output_slot_order,
            passthrough_models=all_passthrough_and_stages,
        )
        AnimaModelMerger._allocated_schemas.append(weakref.ref(schema_payload))

        ordered_model_outputs = []
        for i in range(1, json_model_output_count + 1):
            ordered_model_outputs.append(passthrough_mps.get(f"m{i}", None))
        for s_id in stage_ids_in_order:
            ordered_model_outputs.append(stage_mps.get(s_id, None))

        while len(ordered_model_outputs) < MAX_TOTAL_MODEL_OUTPUTS:
            ordered_model_outputs.append(None)

        ret = {
            "ui": {
                "leaf_keys": [clean_leaf_keys],
                "detected_tier": [resolved_tier],
                "stage_zero_fixes": [stage_zero_fixes],
                "model_names": [model_names_by_id],
            },
            "result": (schema_payload,) + tuple(ordered_model_outputs[:MAX_TOTAL_MODEL_OUTPUTS]),
        }
        AnimaModelMerger._last_merge_sig = merge_sig
        AnimaModelMerger._last_merge_result = ret
        return ret
