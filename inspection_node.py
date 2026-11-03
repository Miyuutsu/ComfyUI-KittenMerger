import json
import os
import re
from datetime import datetime
import torch
import comfy.model_management
import folder_paths

from .merger_node import eval_single_key_stage_recipe
from .utils import (
    MAX_DYNAMIC_MODELS,
    TIER_SPECS,
    ZERO_INIT_SUFFIXES,
    DynamicModelInputs,
    detect_block_tier,
    fmt_val,
    get_model_signature,
    inspect_tensor_diff,
    inspect_tensor_stats,
    natural_sort_key,
    parse_key_for_block,
    resolve_patched_tensor,
    strip_diffusion_prefix,
)

MAX_INSPECTOR_OUTPUTS = 64


def normalize_model_ref(token):
    t = token.strip().lower()
    m = re.match(r"^(?:model_|m)(\d+)$", t)
    if m:
        return f"m{int(m.group(1))}"
    s = re.match(r"^(?:model_s|stage_|s)(\d+)$", t)
    if s:
        return f"s{int(s.group(1))}"
    return None


def parse_compare_schema(schema_str, valid_m_ids):
    if not schema_str or not schema_str.strip():
        return []

    rules = []
    for part in schema_str.split(","):
        raw = part.strip()
        if not raw or "=" not in raw:
            continue
        lhs_str, rhs_str = raw.split("=", 1)
        rhs_id = normalize_model_ref(rhs_str)
        if not rhs_id or rhs_id not in valid_m_ids:
            continue

        lhs_terms = []
        term_pattern = re.finditer(
            r"([+-]?)\s*(?:(\d+(?:\.\d+)?)\s*\*?\s*)?(model_s\d+|stage_\d+|s\d+|model_\d+|m\d+)",
            lhs_str,
            re.IGNORECASE,
        )
        for match in term_pattern:
            sign_str, coeff_str, ref_str = match.groups()
            m_id = normalize_model_ref(ref_str)
            if not m_id or m_id not in valid_m_ids:
                continue
            sign = -1.0 if sign_str == "-" else 1.0
            coeff = float(coeff_str) if coeff_str else 1.0
            lhs_terms.append((sign * coeff, m_id))

        if lhs_terms:
            rules.append({"raw": raw, "lhs": lhs_terms, "rhs": rhs_id})
    return rules


def parse_merger_json_schema(schema_str, models_dict):
    if not schema_str or not schema_str.strip().startswith("{"):
        return None
    try:
        data = json.loads(schema_str)
        if data.get("kitten_merge_schema") != 1:
            return None

        m_ids = list(models_dict.keys())
        palette = data.get("palette", [])
        stage_key_maps = data.get("stage_key_maps") or {}
        stage_summaries_by_id = data.get("stage_summaries_by_id") or {}
        default_key_map = data.get("key_map", {})

        source_remap = {}
        for insp_m, mp in models_dict.items():
            src_id = getattr(mp, "_kitten_source_id", None)
            if src_id:
                source_remap[src_id] = insp_m

        remapped_palette = []
        for entry in palette:
            remapped_entry = {}
            for src_m, coeff in entry.items():
                mapped_m = source_remap.get(src_m, src_m)
                remapped_entry[mapped_m] = remapped_entry.get(mapped_m, 0.0) + float(coeff)
            remapped_palette.append(remapped_entry)

        rules = []
        for insp_m in m_ids:
            mp = models_dict[insp_m]
            s_id = getattr(mp, "_kitten_stage_id", None)
            if s_id and s_id in stage_key_maps:
                s_summary = stage_summaries_by_id.get(s_id, s_id)
                rules.append({
                    "raw": f"Merger ({s_summary}) => {insp_m}",
                    "rhs": insp_m,
                    "palette": remapped_palette,
                    "key_map": stage_key_maps[s_id],
                    "is_kitten_json": True,
                })

        if not rules and default_key_map:
            target_m = m_ids[-1]
            for insp_m in m_ids:
                if getattr(models_dict[insp_m], "_kitten_merged", False):
                    target_m = insp_m
                    break
            rule_name = f"Merger Recipe ({data.get('stages_summary', 'stages')}) => {target_m}"
            rules.append({
                "raw": rule_name,
                "rhs": target_m,
                "palette": remapped_palette,
                "key_map": default_key_map,
                "is_kitten_json": True,
            })

        return rules
    except Exception:
        return None


def parse_donor_list(from_str, all_m_ids, target_m_id, lora_m_ids=None):
    lora_set = lora_m_ids or set()
    if not from_str or not from_str.strip() or from_str.strip().lower() == "all":
        return [m for m in all_m_ids if m != target_m_id and m not in lora_set]
    donors = []
    for tok in re.split(r"[,\s]+", from_str.strip()):
        m_id = normalize_model_ref(tok)
        if (
            m_id
            and m_id in all_m_ids
            and m_id != target_m_id
            and m_id not in lora_set
            and m_id not in donors
        ):
            donors.append(m_id)
    return donors


class ModelMergeInspector:
    _last_cache_sig = None
    _last_cache_data = None
    _model_stats_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diagnostics": ("BOOLEAN", {"default": True}),
                "summary": ("BOOLEAN", {"default": True}),
                "changed_blocks": ("BOOLEAN", {"default": False}),
                "changed_keys": ("BOOLEAN", {"default": False}),
                "zeroed_keys": ("BOOLEAN", {"default": False}),
                "all_keys": ("BOOLEAN", {"default": False}),
                "compare_mode": (
                    ["lineage", "verify"],
                    {
                        "default": "lineage",
                        "tooltip": "lineage: tracks changes from LHS parents to RHS child. verify: computes LHS tensor math and checks drift against RHS.",
                    },
                ),
                "compare_schema": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated expressions (e.g. 'm1 + m2 = m3, m3 - m1 = m4'). Leave blank for no comparison.",
                    },
                ),
                "replace_zeros": (
                    ["none", "blocks", "keys"],
                    {
                        "default": "none",
                        "tooltip": "Repair zeroed blocks or individual keys across connected models.",
                    },
                ),
                "replace_zeros_mode": (
                    ["fallback", "smart_average"],
                    {
                        "default": "fallback",
                        "tooltip": "fallback: use first non-zero donor. smart_average: average all non-zero donors.",
                    },
                ),
                "replace_zeros_from": (
                    "STRING",
                    {
                        "default": "all",
                        "tooltip": "Donor models to pull non-zero weights from (e.g. 'all' or 'm2, m3').",
                    },
                ),
                "change_threshold": (
                    "FLOAT",
                    {
                        "default": 0.001,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.001,
                        "tooltip": "Minimum weight difference to count as CHANGED.",
                    },
                ),
                "zero_threshold": (
                    "FLOAT",
                    {
                        "default": 0.001,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Weight at or below this value is treated as 0.",
                    },
                ),
                "save_to_output": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "model_merge_diff"}),
            },
            "optional": DynamicModelInputs({
                "merge_schema": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Connect merge_schema from Anima Model Merger to automatically pass all models, stages, and lineage without noodling individual model wires.",
                    },
                ),
                "model_1": ("MODEL",),
            }),
        }

    RETURN_TYPES = ("STRING",) + ("MODEL",) * MAX_INSPECTOR_OUTPUTS
    RETURN_NAMES = ("report",) + tuple(f"model_{i}" for i in range(1, MAX_INSPECTOR_OUTPUTS + 1))
    FUNCTION = "inspect_models"
    OUTPUT_NODE = True
    CATEGORY = "kitten_merger"

    def _scan_models(self, models_dict, rules, compare_mode, change_threshold, zero_threshold, stage_recipes=None):
        device = comfy.model_management.get_torch_device()
        m_ids = list(models_dict.keys())
        lora_m_ids = {m for m in m_ids if getattr(models_dict[m], "_kitten_is_lora", False)}
        virtual_m_ids = {m for m in m_ids if getattr(models_dict[m], "_kitten_is_virtual_stage", False)}
        base_m_ids = [m for m in m_ids if m not in lora_m_ids and m not in virtual_m_ids]
        recipe_by_s_id = {ps["s_id"]: ps for ps in (stage_recipes or [])}

        raw_sds = {m: models_dict[m].model.state_dict() for m in m_ids}

        clean_to_ref_key = {}
        ref_pool = base_m_ids if base_m_ids else m_ids
        for m in ref_pool:
            for k in raw_sds[m].keys():
                ck = strip_diffusion_prefix(k)
                if ck not in clean_to_ref_key:
                    clean_to_ref_key[ck] = k

        sds = {}
        for m in m_ids:
            if m in lora_m_ids and clean_to_ref_key:
                aligned = {}
                for k, v in raw_sds[m].items():
                    ck = strip_diffusion_prefix(k)
                    aligned[clean_to_ref_key.get(ck, k)] = v
                sds[m] = aligned
            else:
                sds[m] = raw_sds[m]

        key_sets = {m: set(sds[m].keys()) for m in m_ids}

        base_union_keys = set()
        for m in base_m_ids:
            base_union_keys |= key_sets[m]

        union_keys = set(base_union_keys)
        if not base_m_ids:
            for m in lora_m_ids:
                union_keys |= key_sets[m]

        unmatched_lora_keys = {}
        if base_m_ids:
            for m in lora_m_ids:
                extra = sorted(key_sets[m] - base_union_keys, key=natural_sort_key)
                if extra:
                    unmatched_lora_keys[m] = extra

        all_keys_list = sorted(union_keys, key=natural_sort_key)

        tiers = {}
        for m in m_ids:
            tiers[m], _ = detect_block_tier(key_sets[m])

        first_tier = tiers[base_m_ids[0]] if base_m_ids else (tiers[m_ids[0]] if m_ids else None)
        for vm in virtual_m_ids:
            if tiers.get(vm) is None:
                tiers[vm] = first_tier

        tier_label = TIER_SPECS.get(first_tier, f"Custom ({first_tier} blocks)" if first_tier else "Unknown")

        blocks_data = {}
        key_records = []
        changed_key_count = 0
        unchanged_key_count = 0
        zero_keys_per_model = {m: 0 for m in m_ids}
        nan_keys = {m: [] for m in m_ids}

        missing_keys = {}
        if len(base_m_ids) > 1:
            for m in base_m_ids:
                missing = sorted(base_union_keys - key_sets[m], key=natural_sort_key)
                if missing:
                    missing_keys[m] = missing

        rule_stats = {
            r["raw"]: {"changed_keys": 0, "drift_sum": 0.0, "drift_max": 0.0, "compared": 0}
            for r in rules
        }

        model_stat_maps = {}
        for m in m_ids:
            if m in virtual_m_ids:
                continue
            m_sig = (get_model_signature(models_dict[m]), getattr(models_dict[m], "_kitten_is_lora", False))
            if m_sig not in ModelMergeInspector._model_stats_cache:
                ModelMergeInspector._model_stats_cache[m_sig] = {}
            model_stat_maps[m] = ModelMergeInspector._model_stats_cache[m_sig]

        for m in m_ids:
            if m in virtual_m_ids:
                continue
            m_sig = (get_model_signature(models_dict[m]), getattr(models_dict[m], "_kitten_is_lora", False))
            if m_sig not in ModelMergeInspector._model_stats_cache:
                ModelMergeInspector._model_stats_cache[m_sig] = {}
            model_stat_maps[m] = ModelMergeInspector._model_stats_cache[m_sig]

        skipped_empty = {}

        for k in all_keys_list:
            clean_k = strip_diffusion_prefix(k)
            _, b_idx, suffix = parse_key_for_block(k)

            tensors = {}
            calc_tensors = {}

            def get_calc_tensor(m_id):
                if m_id in calc_tensors:
                    return calc_tensors[m_id]
                if m_id in virtual_m_ids:
                    ps = recipe_by_s_id.get(m_id)
                    if ps is None:
                        calc_tensors[m_id] = None
                        return None
                    gt, _, _ = eval_single_key_stage_recipe(
                        ps, k, clean_k, b_idx, get_calc_tensor, skipped_empty
                    )
                    calc_tensors[m_id] = gt
                    return gt

                if m_id not in tensors:
                    tensors[m_id] = (
                        resolve_patched_tensor(models_dict[m_id], sds[m_id], k, device=device)
                        if k in sds[m_id]
                        else None
                    )
                base_t = tensors[m_id]
                if base_t is None:
                    calc_tensors[m_id] = None
                    return None
                gt = (
                    base_t
                    if (base_t.device == device and base_t.dtype == torch.float32)
                    else base_t.detach().to(device=device, dtype=torch.float32)
                )
                calc_tensors[m_id] = gt
                return gt

            weights = {}
            zero_in = []

            for m in m_ids:
                if m in virtual_m_ids:
                    gt = get_calc_tensor(m)
                    w, has_nan = inspect_tensor_stats(gt, device=device) if gt is not None else (None, False)
                elif k in sds[m]:
                    stat_cache = model_stat_maps[m]
                    if k in stat_cache:
                        w, has_nan = stat_cache[k]
                    else:
                        gt = get_calc_tensor(m)
                        w, has_nan = inspect_tensor_stats(gt, device=device)
                        stat_cache[k] = (w, has_nan)
                else:
                    w, has_nan = (0.0, False) if m in lora_m_ids else (None, False)

                weights[m] = w
                if has_nan:
                    nan_keys[m].append(k)
                if w is not None and abs(w) <= zero_threshold:
                    zero_keys_per_model[m] += 1
                    zero_in.append(m)

            rule_diffs = {}
            is_changed = False

            for r in rules:
                raw_r = r["raw"]
                rhs_m = r["rhs"]
                t_rhs = get_calc_tensor(rhs_m)
                if t_rhs is None or not t_rhs.is_floating_point():
                    continue

                if r.get("is_kitten_json"):
                    p_idx = r["key_map"].get(k)
                    if p_idx is None or p_idx >= len(r["palette"]):
                        continue
                    lhs_pairs = [(float(c), m_id) for m_id, c in r["palette"][p_idx].items()]
                else:
                    lhs_pairs = r["lhs"]

                parent_max_diff = 0.0
                has_parent = False
                for coeff, lhs_m in lhs_pairs:
                    if abs(coeff) <= 1e-9 or lhs_m in lora_m_ids:
                        continue
                    t_lhs = get_calc_tensor(lhs_m)
                    if t_lhs is not None and t_lhs.shape == t_rhs.shape:
                        d = inspect_tensor_diff(t_lhs, t_rhs, device=device)
                        parent_max_diff = max(parent_max_diff, d)
                        has_parent = True

                if has_parent and parent_max_diff >= change_threshold:
                    is_changed = True

                if compare_mode == "verify":
                    expected = None
                    valid_verify = True
                    for coeff, lhs_m in lhs_pairs:
                        if abs(coeff) <= 1e-9:
                            continue
                        t_lhs = get_calc_tensor(lhs_m)
                        if t_lhs is None:
                            if lhs_m in lora_m_ids:
                                continue
                            valid_verify = False
                            break
                        if t_lhs.shape != t_rhs.shape:
                            valid_verify = False
                            break
                        term = t_lhs * float(coeff)
                        expected = term if expected is None else (expected + term)

                    if valid_verify and expected is not None:
                        r_diff = inspect_tensor_diff(expected, t_rhs, device=device)
                        rule_diffs[raw_r] = r_diff
                        st = rule_stats[raw_r]
                        st["compared"] += 1
                        st["drift_sum"] += r_diff
                        st["drift_max"] = max(st["drift_max"], r_diff)
                        if r_diff >= change_threshold:
                            st["changed_keys"] += 1
                else:
                    if has_parent:
                        rule_diffs[raw_r] = parent_max_diff
                        st = rule_stats[raw_r]
                        st["compared"] += 1
                        st["drift_sum"] += parent_max_diff
                        st["drift_max"] = max(st["drift_max"], parent_max_diff)
                        if parent_max_diff >= change_threshold:
                            st["changed_keys"] += 1

            if is_changed:
                changed_key_count += 1
            else:
                unchanged_key_count += 1

            is_gate = suffix is not None and any(suffix.endswith(z) for z in ZERO_INIT_SUFFIXES)
            is_norm = suffix is not None and ("_norm." in suffix or ".norm" in suffix)

            rec = {
                "key": k,
                "block": b_idx,
                "suffix": suffix,
                "is_gate": is_gate,
                "is_norm": is_norm,
                "weights": weights,
                "rule_diffs": rule_diffs,
                "changed": is_changed,
                "zero_any": len(zero_in) > 0,
                "zero_in": zero_in,
            }
            key_records.append(rec)

            if b_idx is not None:
                b_entry = blocks_data.setdefault(
                    b_idx,
                    {
                        "proj": {m: [] for m in m_ids},
                        "gate": {m: [] for m in m_ids},
                        "changed_keys": 0,
                        "records": [],
                    },
                )
                b_entry["records"].append(rec)
                if is_changed:
                    b_entry["changed_keys"] += 1

                for m in m_ids:
                    if m in lora_m_ids:
                        if k in sds[m] and weights[m] is not None:
                            b_entry["proj"][m].append(weights[m])
                    else:
                        w = weights[m]
                        if w is not None:
                            if not is_norm:
                                b_entry["proj"][m].append(w)
                            if is_gate:
                                b_entry["gate"][m].append(w)

            tensors.clear()
            calc_tensors.clear()

        return {
            "m_ids": m_ids,
            "lora_m_ids": lora_m_ids,
            "base_m_ids": base_m_ids,
            "tiers": tiers,
            "tier_label": tier_label,
            "total_keys": len(all_keys_list),
            "changed_key_count": changed_key_count,
            "unchanged_key_count": unchanged_key_count,
            "zero_keys_per_model": zero_keys_per_model,
            "nan_keys": nan_keys,
            "missing_keys": missing_keys,
            "unmatched_lora_keys": unmatched_lora_keys,
            "blocks_data": blocks_data,
            "key_records": key_records,
            "rule_stats": rule_stats,
        }

    def _solve_zeros(
        self,
        models_dict,
        data,
        replace_zeros,
        replace_zeros_mode,
        replace_zeros_from,
        zero_blocks_list,
    ):
        m_ids = data["m_ids"]
        lora_m_ids = data.get("lora_m_ids", set())
        if replace_zeros == "none" or len(m_ids) < 2:
            return {
                m: (None if getattr(mp, "_kitten_is_virtual_stage", False) else mp)
                for m, mp in models_dict.items()
            }, []

        device = comfy.model_management.get_torch_device()
        sds = {m: models_dict[m].model.state_dict() for m in m_ids}
        repaired_models = {}
        repair_logs = []

        for m in m_ids:
            if m in lora_m_ids or getattr(models_dict[m], "_kitten_is_virtual_stage", False):
                repaired_models[m] = (
                    None if getattr(models_dict[m], "_kitten_is_virtual_stage", False) else models_dict[m]
                )
                continue

            donors = parse_donor_list(replace_zeros_from, m_ids, m, lora_m_ids=lora_m_ids)
            donors = [d for d in donors if not getattr(models_dict.get(d), "_kitten_is_virtual_stage", False)]
            if not donors:
                repaired_models[m] = models_dict[m]
                continue

            keys_to_repair = []
            if replace_zeros == "blocks":
                muted_blocks = set(zero_blocks_list.get(m, []))
                if not muted_blocks:
                    repaired_models[m] = models_dict[m]
                    continue
                for b_idx in sorted(muted_blocks):
                    active_donors = [d for d in donors if b_idx not in zero_blocks_list.get(d, [])]
                    if not active_donors:
                        continue
                    for rec in data["blocks_data"][b_idx]["records"]:
                        keys_to_repair.append((rec["key"], active_donors))
            elif replace_zeros == "keys":
                for rec in data["key_records"]:
                    if m in rec["zero_in"]:
                        active_donors = [d for d in donors if d not in rec["zero_in"] and rec["weights"].get(d) is not None]
                        if active_donors:
                            keys_to_repair.append((rec["key"], active_donors))

            if not keys_to_repair:
                repaired_models[m] = models_dict[m]
                continue

            cloned_mp = models_dict[m].clone()
            patched_count = 0

            for key, valid_donors in keys_to_repair:
                if key not in sds[m]:
                    continue
                base_dtype = sds[m][key].dtype
                chosen_donors = valid_donors[:1] if replace_zeros_mode == "fallback" else valid_donors

                donor_tensors = []
                for d_id in chosen_donors:
                    dt = resolve_patched_tensor(models_dict[d_id], sds[d_id], key, device=device)
                    if dt is not None and dt.shape == sds[m][key].shape:
                        donor_tensors.append(dt.detach().to(device=device, dtype=torch.float32))

                if not donor_tensors:
                    continue

                if len(donor_tensors) == 1:
                    rep_t = donor_tensors[0]
                else:
                    rep_t = torch.stack(donor_tensors, dim=0).mean(dim=0)

                rep_cpu = rep_t.to(device="cpu", dtype=base_dtype).contiguous()
                cloned_mp.patches.pop(key, None)
                cloned_mp.add_patches({key: ("diff", (rep_cpu,))}, strength_patch=1.0, strength_model=0.0)
                patched_count += 1

            repaired_models[m] = cloned_mp
            if patched_count > 0:
                donor_str = ", ".join(donors)
                repair_logs.append(
                    f"[SOLVE ZEROES] {m}: Repaired {patched_count} keys (scope={replace_zeros}, mode={replace_zeros_mode}, from=[{donor_str}])."
                )

        return repaired_models, repair_logs

    def inspect_models(
        self,
        diagnostics=True,
        summary=True,
        changed_blocks=False,
        changed_keys=False,
        zeroed_keys=False,
        all_keys=False,
        compare_mode="lineage",
        compare_schema="",
        replace_zeros="none",
        replace_zeros_mode="fallback",
        replace_zeros_from="all",
        change_threshold=0.001,
        zero_threshold=0.001,
        save_to_output=False,
        filename_prefix="model_merge_diff",
        merge_schema=None,
        **kwargs,
    ):
        models_dict = {}
        alias_map = {}
        using_bundled_schema = False

        bundled_list = getattr(merge_schema, "_kitten_bundled_models", None)
        if bundled_list:
            using_bundled_schema = True
            for slot_id, mp in bundled_list:
                if mp is not None:
                    models_dict[slot_id] = mp

        for i in range(1, MAX_DYNAMIC_MODELS + 1):
            mp = kwargs.get(f"model_{i}")
            if mp is not None:
                m_slot = f"m{i}"
                models_dict[m_slot] = mp
                st_id = getattr(mp, "_kitten_stage_id", None)
                src_id = getattr(mp, "_kitten_source_id", None)
                if st_id and st_id not in models_dict:
                    alias_map[st_id] = m_slot
                if src_id and src_id not in models_dict:
                    alias_map[src_id] = m_slot

        if not models_dict:
            empty_msg = "No models connected. Connect merge_schema from Anima Model Merger or connect at least model_1."
            return {
                "ui": {"sections": [{"HEADER": empty_msg}], "report_text": [empty_msg]},
                "result": (empty_msg,) + (None,) * MAX_INSPECTOR_OUTPUTS,
            }

        m_ids = list(models_dict.keys())
        valid_refs = set(m_ids) | set(alias_map.keys())

        # If the user typed a manual compare_schema, prioritize it over merge_schema's auto-rules!
        if compare_schema and compare_schema.strip():
            effective_schema = compare_schema.strip()
            raw_rules = parse_compare_schema(effective_schema, valid_refs)
            rules = [
                {
                    "raw": r["raw"],
                    "lhs": [(c, alias_map.get(m, m)) for c, m in r["lhs"]],
                    "rhs": alias_map.get(r["rhs"], r["rhs"]),
                }
                for r in raw_rules
            ]
        elif merge_schema and str(merge_schema).strip():
            effective_schema = str(merge_schema).strip()
            json_rules = parse_merger_json_schema(effective_schema, models_dict)
            rules = json_rules if json_rules is not None else parse_compare_schema(effective_schema, valid_refs)
        else:
            effective_schema = ""
            rules = []

        sig = (
            tuple(
                (
                    m,
                    get_model_signature(models_dict[m]),
                    getattr(models_dict[m], "_kitten_stage_id", None),
                    getattr(models_dict[m], "_kitten_is_lora", False),
                )
                for m in m_ids
            ),
            compare_mode,
            effective_schema.strip(),
            round(change_threshold, 6),
            round(zero_threshold, 6),
        )

        stage_recipes = getattr(merge_schema, "_kitten_stage_recipes", None)
        if ModelMergeInspector._last_cache_sig == sig and ModelMergeInspector._last_cache_data is not None:
            data = ModelMergeInspector._last_cache_data
        else:
            data = self._scan_models(
                models_dict,
                rules,
                compare_mode,
                change_threshold,
                zero_threshold,
                stage_recipes=stage_recipes,
            )
            ModelMergeInspector._last_cache_sig = sig
            ModelMergeInspector._last_cache_data = data

        lora_m_ids = data.get("lora_m_ids", set())
        blocks_data = data["blocks_data"]
        key_records = data["key_records"]
        zk = data["zero_keys_per_model"]

        def avg_or_zero(vals, gates, is_lora=False):
            if is_lora:
                if not vals or max(vals) <= zero_threshold:
                    return 0.0, True
                return sum(vals) / len(vals), False
            if not vals:
                return None, False
            if gates and max(gates) <= zero_threshold:
                return 0.0, True
            return sum(vals) / len(vals), False

        zero_blocks_list = {m: [] for m in m_ids}
        diluted_blocks = []

        for b_idx in sorted(blocks_data.keys()):
            bd = blocks_data[b_idx]
            muted_map = {}
            for m in m_ids:
                is_lora = m in lora_m_ids
                _, muted = avg_or_zero(bd["proj"][m], bd["gate"][m], is_lora=is_lora)
                muted_map[m] = muted
                if muted:
                    zero_blocks_list[m].append(b_idx)

            for r in rules:
                if r.get("is_kitten_json"):
                    lhs_m_ids = []
                    for rec in bd["records"]:
                        p_idx = r["key_map"].get(rec["key"])
                        if p_idx is not None and p_idx < len(r["palette"]):
                            for m_id in r["palette"][p_idx]:
                                if m_id in m_ids and m_id not in lora_m_ids and m_id not in lhs_m_ids:
                                    lhs_m_ids.append(m_id)
                else:
                    lhs_m_ids = [lhs_m for _, lhs_m in r["lhs"] if lhs_m not in lora_m_ids]

                rhs_m = r["rhs"]
                if len(lhs_m_ids) >= 2 and bd["changed_keys"] > 0:
                    muted_parents = [m for m in lhs_m_ids if muted_map.get(m, False)]
                    active_parents = [m for m in lhs_m_ids if not muted_map.get(m, True)]
                    if muted_parents and active_parents:
                        diluted_blocks.append((b_idx, r["raw"], muted_parents, rhs_m))

        repaired_models, repair_logs = self._solve_zeros(
            models_dict,
            data,
            replace_zeros,
            replace_zeros_mode,
            replace_zeros_from,
            zero_blocks_list,
        )

        sec_diag = ["=== DIAGNOSTICS & MERGE HEALTH ==="]
        issues_found = 0

        for m in m_ids:
            bad_list = data["nan_keys"][m]
            if bad_list:
                issues_found += 1
                sec_diag.append(
                    f"[CRITICAL] {m} contains {len(bad_list)} corrupted (NaN/Inf) tensors! First: {bad_list[0]}"
                )

        base_tiers = {m: data["tiers"][m] for m in data.get("base_m_ids", m_ids)}
        if len(set(base_tiers.values())) > 1:
            issues_found += 1
            tier_str = ", ".join(f"{m}={base_tiers[m]}" for m in base_tiers)
            sec_diag.append(f"[WARNING] Block count mismatch across base models: {tier_str}")

        for m, miss_list in data["missing_keys"].items():
            issues_found += 1
            sec_diag.append(f"[WARNING] {m} is missing {len(miss_list)} keys present in other models (e.g. {miss_list[0]}).")

        for m, extra_lora_list in data.get("unmatched_lora_keys", {}).items():
            issues_found += 1
            sec_diag.append(
                f"[WARNING] LoRA {m} contains {len(extra_lora_list)} keys not present in base models (first: {extra_lora_list[0]})."
            )

        for m in m_ids:
            z_list = zero_blocks_list[m]
            if z_list:
                issues_found += 1
                blk_str = ", ".join(f"{b:02d}" for b in z_list)
                if m in lora_m_ids:
                    sec_diag.append(f"[INFO] {m} (LoRA) has {len(z_list)} zero/unpopulated blocks: [{blk_str}]")
                else:
                    sec_diag.append(f"[INFO] {m} has {len(z_list)} zeroed-out blocks: [{blk_str}]")

        if diluted_blocks:
            issues_found += 1
            blk_nums = ", ".join(sorted({f"{b[0]:02d}" for b in diluted_blocks}))
            sec_diag.append(
                f"[WARNING] ZERO-BLOCK MERGE DETECTED on blocks: [{blk_nums}]\n"
                f"          A zeroed block was combined with an active block in your schema, diluting trained weights!"
            )

        if rules:
            for r in rules:
                raw_r = r["raw"]
                st = data["rule_stats"][raw_r]
                mean_d = (st["drift_sum"] / st["compared"]) if st["compared"] > 0 else 0.0
                if compare_mode == "verify":
                    status_tag = "PASS" if st["changed_keys"] == 0 else "DRIFT"
                    sec_diag.append(
                        f"[VERIFY:{status_tag}] '{raw_r}' -> {st['changed_keys']}/{st['compared']} keys drifted >= {change_threshold} "
                        f"(mean diff: {mean_d:.4f}, max diff: {st['drift_max']:.4f})"
                    )
                else:
                    sec_diag.append(
                        f"[LINEAGE] '{raw_r}' -> {st['changed_keys']}/{st['compared']} keys changed >= {change_threshold} "
                        f"(mean diff: {mean_d:.4f}, max diff: {st['drift_max']:.4f})"
                    )
        elif compare_schema.strip():
            sec_diag.append(
                f"[WARNING] Could not parse compare_schema '{compare_schema}'. Use format like 'm1 + m2 = m3'."
            )

        for log_line in repair_logs:
            sec_diag.append(log_line)

        if issues_found == 0 and not repair_logs:
            sec_diag.append("[OK] All connected models have matching keys, active blocks, and zero NaN/Inf corruption.")

        has_comparison = len(rules) > 0

        def format_weights_inline(w_map):
            return " | ".join(f"{m}: {fmt_val(w_map.get(m), zero_threshold):>6}" for m in m_ids)

        def format_key_line(rec, show_status=True):
            w_str = format_weights_inline(rec["weights"])
            tags = []
            if show_status and has_comparison:
                tags.append("CHANGED" if rec["changed"] else "SAME")
            if rec["zero_any"]:
                tags.append(f"=0 in {','.join(rec['zero_in'])}")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            return f"{rec['key']}  -->  {w_str}{tag_str}"

        def format_block_line(b_idx, bd):
            eff_map = {}
            muted_list = []
            for m in m_ids:
                is_lora = m in lora_m_ids
                eff, muted = avg_or_zero(bd["proj"][m], bd["gate"][m], is_lora=is_lora)
                eff_map[m] = eff
                if muted:
                    muted_list.append(f"{m}=0")

            w_str = format_weights_inline(eff_map)
            tags = []
            if has_comparison:
                tags.append("CHANGED" if bd["changed_keys"] > 0 else "SAME")
            if muted_list:
                tags.extend(muted_list)
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            return f"Block {b_idx:02d} | {w_str}{tag_str}"

        sec_summary = ["=== SUMMARY (BLOCKS) ==="]
        for b_idx in sorted(blocks_data.keys()):
            sec_summary.append(format_block_line(b_idx, blocks_data[b_idx]))

        sec_changed_blocks = ["=== CHANGED BLOCKS & KEYS ==="]
        if not has_comparison:
            sec_changed_blocks.append("No compare_schema specified. Enter a schema (e.g. 'm1 + m2 = m3') to track changes.")
        else:
            any_changed_block = False
            for b_idx in sorted(blocks_data.keys()):
                bd = blocks_data[b_idx]
                if bd["changed_keys"] > 0:
                    any_changed_block = True
                    sec_changed_blocks.append(format_block_line(b_idx, bd))
                    for rec in bd["records"]:
                        if rec["changed"]:
                            short_key = rec["suffix"].lstrip(".") if rec["suffix"] else rec["key"]
                            w_str = format_weights_inline(rec["weights"])
                            sec_changed_blocks.append(f"    .{short_key:<42} --> {w_str}")

            non_block_changed = [r for r in key_records if r["block"] is None and r["changed"]]
            if non_block_changed:
                any_changed_block = True
                sec_changed_blocks.append("Non-Block Layers (llm_adapter / embedders / final_layer):")
                for rec in non_block_changed:
                    sec_changed_blocks.append("    " + format_key_line(rec, show_status=False))
            if not any_changed_block:
                sec_changed_blocks.append(f"No blocks or keys changed by >= {change_threshold}.")

        sec_changed_keys = ["=== CHANGED KEYS ==="]
        if not has_comparison:
            sec_changed_keys.append("No compare_schema specified. Enter a schema (e.g. 'm1 + m2 = m3') to track changes.")
        else:
            changed_recs = [r for r in key_records if r["changed"]]
            if changed_recs:
                for rec in changed_recs:
                    sec_changed_keys.append(format_key_line(rec, show_status=False))
            else:
                sec_changed_keys.append(f"No individual keys changed by >= {change_threshold}.")

        sec_zero_keys = ["=== ZEROED KEYS ==="]
        zero_recs = [r for r in key_records if r["zero_any"]]
        if zero_recs:
            for rec in zero_recs:
                sec_zero_keys.append(format_key_line(rec, show_status=False))
        else:
            sec_zero_keys.append(f"No zeroed keys (<= {zero_threshold}) found.")

        sec_all_keys = ["=== ALL KEYS ==="]
        for rec in key_records:
            sec_all_keys.append(format_key_line(rec, show_status=True))

        conn_labels = []
        for m in m_ids:
            disp = getattr(models_dict[m], "_kitten_display_name", None)
            if m in lora_m_ids:
                conn_labels.append(f"{m} ({disp} [LoRA])" if disp and disp != m else f"{m} (LoRA)")
            else:
                conn_labels.append(f"{m} ({disp})" if disp and disp != m else m)

        zb_str = " | ".join(
            f"{m}: {len(zero_blocks_list[m])}{' (LoRA)' if m in lora_m_ids else ''}"
            for m in m_ids
        )
        zk_str = ", ".join(f"{m}: {zk[m]}" for m in m_ids)
        zero_summary_line = f"Zeroed Blocks: {zb_str}  (Zeroed Keys -> {zk_str})"

        if has_comparison:
            change_line = f"Key Changes ({compare_mode}) : {data['changed_key_count']} changed | {data['unchanged_key_count']} unchanged"
        else:
            change_line = "Key Changes  : Comparison disabled (compare_schema is blank)"

        sections_dict = {
            "HEADER": "\n".join(
                [
                    f"Architecture : {data['tier_label']} | Connected: {', '.join(conn_labels)} | Total Keys: {data['total_keys']}",
                    change_line,
                    zero_summary_line,
                    "=" * 72,
                ]
            ),
            "diagnostics": "\n".join(sec_diag),
            "summary": "\n".join(sec_summary),
            "changed_blocks": "\n".join(sec_changed_blocks),
            "changed_keys": "\n".join(sec_changed_keys),
            "zeroed_keys": "\n".join(sec_zero_keys),
            "all_keys": "\n".join(sec_all_keys),
        }

        active_parts = [sections_dict["HEADER"]]
        if diagnostics:
            active_parts.append(sections_dict["diagnostics"])
        if summary:
            active_parts.append(sections_dict["summary"])
        if changed_blocks:
            active_parts.append(sections_dict["changed_blocks"])
        if changed_keys:
            active_parts.append(sections_dict["changed_keys"])
        if zeroed_keys:
            active_parts.append(sections_dict["zeroed_keys"])
        if all_keys:
            active_parts.append(sections_dict["all_keys"])

        if len(active_parts) == 1:
            active_parts.append("No switches turned on. Toggle at least one switch above.")

        final_report = "\n\n".join(active_parts)

        if save_to_output:
            out_dir = folder_paths.get_output_directory()
            os.makedirs(out_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_prefix = re.sub(r"[^\w\-_]", "_", str(filename_prefix))[:50]
            if not clean_prefix:
                clean_prefix = "model_merge_diff"
            out_path = os.path.join(out_dir, f"{clean_prefix}_{timestamp}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(final_report + "\n")

        if using_bundled_schema:
            slot_order = getattr(merge_schema, "_kitten_output_slot_order", None) or m_ids
            fallback_mps = getattr(merge_schema, "_kitten_passthrough_models", None) or {}
            ordered_out = [
                repaired_models.get(slot_id, fallback_mps.get(slot_id, None))
                for slot_id in slot_order
            ]
            while len(ordered_out) < MAX_INSPECTOR_OUTPUTS:
                ordered_out.append(None)
            model_outputs = tuple(ordered_out[:MAX_INSPECTOR_OUTPUTS])
        else:
            model_outputs = tuple(
                repaired_models.get(f"m{i}", None) for i in range(1, MAX_INSPECTOR_OUTPUTS + 1)
            )

        return {
            "ui": {"sections": [sections_dict], "report_text": [final_report]},
            "result": (final_report,) + model_outputs,
        }
