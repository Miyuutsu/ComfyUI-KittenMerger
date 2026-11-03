import math
import re
import torch
import comfy.lora

MAX_DYNAMIC_MODELS = 32

MANIFEST_28_TO_40 = {
    "old_block_count": 28,
    "new_block_count": 40,
    "insertion_positions": [2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36],
    "inserted_to_source": {
        "2": 1, "5": 3, "8": 5, "11": 7, "14": 9, "17": 11,
        "21": 14, "24": 16, "27": 18, "30": 20, "33": 22, "36": 24,
    },
}

MANIFEST_40_TO_52 = {
    "old_block_count": 40,
    "new_block_count": 52,
    "insertion_positions": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47],
    "inserted_to_source": {
        "3": 2, "7": 5, "11": 8, "15": 11, "19": 14, "23": 17,
        "27": 20, "31": 23, "35": 26, "39": 29, "43": 32, "47": 35,
    },
}

TIER_SPECS = {
    28: "Anima 2B (28 blocks)",
    40: "Anima 2.9B (40 blocks)",
    52: "Anima 3.8B Pro52 (52 blocks)",
}

ZERO_INIT_SUFFIXES = (
    "adaln_modulation_self_attn.2.weight",
    "adaln_modulation_cross_attn.2.weight",
    "adaln_modulation_mlp.2.weight",
    "self_attn.output_proj.weight",
    "cross_attn.output_proj.weight",
    "mlp.layer2.weight",
    "norm1_linear_2.weight",
    "norm2_linear_2.weight",
    "norm3_linear_2.weight",
    "attn1_to_out.weight",
    "attn2_to_out.weight",
    "ff_net_2.weight",
)

SUBSPACE_PROJECTION_TRIPLETS = (
    (
        ".self_attn.output_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        "self_attn",
    ),
    (
        ".cross_attn.output_proj.weight",
        ".cross_attn.v_proj.weight",
        ".cross_attn.q_proj.weight",
        ".cross_attn.k_proj.weight",
        "cross_attn",
    ),
    (
        ".mlp.layer2.weight",
        ".mlp.layer1.weight",
        None,
        None,
        "mlp",
    ),
)


class MergerSchemaPayload(str):
    """
    String subclass for merge_schema that behaves like a normal JSON string for any text node,
    while carrying the ordered list of (slot_id, ModelPatcher) objects and stage recipes to ModelMergeInspector.
    """
    def __new__(
        cls,
        content,
        bundled_models=None,
        stage_recipes=None,
        is_lora_map=None,
        output_slot_order=None,
        passthrough_models=None,
    ):
        obj = super().__new__(cls, content)
        obj._kitten_bundled_models = bundled_models or []
        obj._kitten_stage_recipes = stage_recipes or []
        obj._kitten_is_lora_map = is_lora_map or {}
        obj._kitten_output_slot_order = output_slot_order or []
        obj._kitten_passthrough_models = passthrough_models or {}
        return obj

    def __deepcopy__(self, memo):
        return str(self)


class DynamicModelInputs(dict):
    """Allows ComfyUI's validator to accept dynamically created model_1..model_N inputs."""
    def __contains__(self, key):
        if super().__contains__(key):
            return True
        return isinstance(key, str) and bool(re.match(r"^model_\d+$", key))

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str) and re.match(r"^model_\d+$", key):
            return ("MODEL",)
        raise KeyError(key)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def strip_diffusion_prefix(key):
    for prefix in ("model.diffusion_model.", "diffusion_model.", "net."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def parse_key_for_block(key):
    if "llm_adapter" in key:
        return None, None, None
    match = re.search(r"^(.*(?:net|diffusion_model)\.blocks\.)(\d+)(\..*)$", key)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    match = re.search(r"^(.*_blocks_)(\d+)(_.*)$", key)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    match = re.search(r"^(.*?blocks\.)(block)?(\d+)(\..*)", key)
    if match:
        return match.group(1) + (match.group(2) or ""), int(match.group(3)), match.group(4)
    return None, None, None


def detect_block_tier(keys):
    max_block = -1
    for k in keys:
        _, b_idx, _ = parse_key_for_block(k)
        if b_idx is not None:
            max_block = max(max_block, b_idx)
    if max_block < 0:
        return None, -1
    if max_block <= 27:
        return 28, max_block
    if max_block <= 39:
        return 40, max_block
    if max_block <= 51:
        return 52, max_block
    return max_block + 1, max_block


def resolve_patched_tensor(model_patcher, base_sd, key, device=None):
    if model_patcher is None or key not in base_sd:
        return None

    weight = base_sd[key]
    patches = getattr(model_patcher, "patches", {})
    if key not in patches:
        return weight

    patch_list = patches[key]
    if len(patch_list) == 1:
        p = patch_list[0]
        if (
            isinstance(p, tuple)
            and len(p) >= 3
            and p[0] == 1.0
            and p[2] == 0.0
            and isinstance(p[1], tuple)
            and len(p[1]) >= 2
            and p[1][0] == "diff"
        ):
            return p[1][1][0]

    target_dev = device if device is not None else weight.device
    try:
        w_clone = weight.detach().to(device=target_dev, dtype=torch.float32).clone()
        if hasattr(comfy.lora, "calculate_weight"):
            patched = comfy.lora.calculate_weight(patch_list, w_clone, key, intermediate_dtype=torch.float32)
        elif hasattr(model_patcher, "calculate_weight"):
            patched = model_patcher.calculate_weight(patch_list, w_clone, key)
        else:
            patched = w_clone
        return patched
    except Exception:
        return weight


def materialize_patched_state_dict(model_patcher, device=None):
    """
    Returns a dict of resolved tensors (on CPU) for a ModelPatcher.
    Unpatched keys reference the underlying tensor directly without copying;
    patched keys are cast back to the base tensor's native dtype (e.g. bf16) on CPU.
    """
    base_sd = model_patcher.model.state_dict()
    patches = getattr(model_patcher, "patches", {})
    if not patches:
        return dict(base_sd)

    out = {}
    for k, base_t in base_sd.items():
        if k not in patches:
            out[k] = base_t
        else:
            t = resolve_patched_tensor(model_patcher, base_sd, k, device=device)
            if t is not None:
                if t.device.type == "cpu" and t.dtype == base_t.dtype:
                    out[k] = t
                else:
                    out[k] = t.detach().to(device="cpu", dtype=base_t.dtype).contiguous()
    return out


def inspect_tensor_stats(tensor, device=None):
    if tensor is None:
        return None, False
    if not tensor.is_floating_point():
        return 0.0, False
    if tensor.numel() == 0:
        return 0.0, False

    target_dev = device if device is not None else tensor.device
    t = tensor.detach().to(device=target_dev, dtype=torch.float32)
    norm = torch.linalg.norm(t).item()

    if not math.isfinite(norm):
        has_bad = not bool(torch.isfinite(t).all().item())
        if has_bad:
            return 0.0, True
        norm = torch.linalg.norm(t.double()).item()

    if norm < 1e-7:
        return 0.0, False
    return (norm / (t.numel() ** 0.5)) * 100.0, False


def inspect_tensor_diff(t1, t2, device=None):
    if t1 is None or t2 is None or t1.shape != t2.shape:
        return 0.0
    if not t1.is_floating_point():
        return 0.0
    if t1.numel() == 0:
        return 0.0

    target_dev = device if device is not None else t1.device
    diff = t1.detach().to(device=target_dev, dtype=torch.float32) - t2.detach().to(device=target_dev, dtype=torch.float32)
    norm = torch.linalg.norm(diff).item()
    if not math.isfinite(norm) or norm < 1e-7:
        return 0.0
    return (norm / (diff.numel() ** 0.5)) * 100.0


def fmt_val(val, zero_tol=1e-4):
    if val is None:
        return "MISSING"
    if abs(val) <= zero_tol:
        return "0"
    return f"{val:.3f}"


def get_model_signature(mp):
    if mp is None:
        return None
    patches = getattr(mp, "patches", {})
    patch_sig = tuple(
        (
            k,
            len(v),
            tuple(
                (
                    p[0],
                    id(p[1][1][0])
                    if (
                        isinstance(p[1], tuple)
                        and len(p[1]) >= 2
                        and isinstance(p[1][1], tuple)
                        and len(p[1][1]) >= 1
                    )
                    else None,
                )
                for p in v
                if isinstance(p, tuple) and len(p) >= 2
            ),
        )
        for k, v in sorted(patches.items())
    )
    return (id(mp.model), getattr(mp, "patches_uuid", None), hash(patch_sig))


def build_source_to_target_map(manifest):
    total_old = int(manifest["old_block_count"])
    total_new = int(manifest["new_block_count"])
    inserted_to_source = {int(k): int(v) for k, v in manifest.get("inserted_to_source", {}).items()}
    inserted_new_indices = set(manifest.get("insertion_positions", inserted_to_source))

    source_to_target = {}
    old_idx = 0
    for new_idx in range(total_new):
        if new_idx in inserted_new_indices:
            continue
        source_to_target[old_idx] = new_idx
        old_idx += 1
        if old_idx >= total_old:
            break
    return source_to_target


def compute_matrix_cosine(mat_a, mat_b):
    norm_a = torch.linalg.norm(mat_a).item()
    norm_b = torch.linalg.norm(mat_b).item()
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    cos = torch.sum(mat_a * mat_b).item() / (norm_a * norm_b)
    return max(0.0, min(1.0, cos))


def analyze_inserted_blocks(state_dict, manifest, device="cpu"):
    source_to_target = build_source_to_target_map(manifest)
    inserted_to_source = {int(k): int(v) for k, v in manifest.get("inserted_to_source", {}).items()}

    block_tensors = {}
    for k, v in state_dict.items():
        _, b_idx, suffix = parse_key_for_block(k)
        if b_idx is not None and v.is_floating_point():
            block_tensors.setdefault(b_idx, {})[suffix] = v

    gate_ratios = []
    drift_ratios = []
    cos_sims = []

    for ins_high_idx, src_low_idx in sorted(inserted_to_source.items()):
        parent_high_idx = source_to_target.get(src_low_idx)
        if ins_high_idx not in block_tensors or parent_high_idx not in block_tensors:
            continue

        ins_map = block_tensors[ins_high_idx]
        par_map = block_tensors[parent_high_idx]

        for suffix, ins_t in ins_map.items():
            if suffix not in par_map or ins_t.shape != par_map[suffix].shape:
                continue
            if "adaln_modulation" in suffix:
                continue

            ins_f = ins_t.detach().to(device=device, dtype=torch.float32)
            par_f = par_map[suffix].detach().to(device=device, dtype=torch.float32)
            par_norm = torch.linalg.norm(par_f).item()
            ins_norm = torch.linalg.norm(ins_f).item()
            if par_norm < 1e-8:
                continue

            is_zero_init = any(suffix.endswith(z) for z in ZERO_INIT_SUFFIXES)
            if is_zero_init:
                gate_ratios.append(ins_norm / par_norm)
            else:
                diff_norm = torch.linalg.norm(ins_f - par_f).item()
                drift_ratios.append(diff_norm / par_norm)
                if ins_norm > 1e-8:
                    cos = torch.sum(ins_f * par_f).item() / (par_norm * ins_norm)
                    cos_sims.append(max(0.0, cos))

    mean_gate = sum(gate_ratios) / len(gate_ratios) if gate_ratios else 0.0
    mean_drift = sum(drift_ratios) / len(drift_ratios) if drift_ratios else 0.0
    mean_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0

    if mean_gate < 0.02:
        return 0.0
    if mean_drift < 0.10 and mean_gate < 0.25:
        return round(min(0.18, mean_gate * 0.35), 3)
    return round(min(0.22, max(0.12, 0.25 * mean_cos)), 3)


def compute_subspace_aligned_block_deltas(ins_suffix_map, nat_suffix_map, weight, device):
    if weight <= 0.0:
        return {}

    block_deltas = {}
    for out_sfx, in_sfx, q_sfx, k_sfx, subblock_type in SUBSPACE_PROJECTION_TRIPLETS:
        if out_sfx not in ins_suffix_map or out_sfx not in nat_suffix_map:
            continue
        if in_sfx not in ins_suffix_map or in_sfx not in nat_suffix_map:
            continue

        w_out_ins = ins_suffix_map[out_sfx].to(device=device, dtype=torch.float32)
        w_out_nat = nat_suffix_map[out_sfx].to(device=device, dtype=torch.float32)
        w_in_ins = ins_suffix_map[in_sfx].to(device=device, dtype=torch.float32)
        w_in_nat = nat_suffix_map[in_sfx].to(device=device, dtype=torch.float32)

        if w_out_ins.shape != w_out_nat.shape or w_in_ins.shape != w_in_nat.shape:
            continue

        ins_out_norm = torch.linalg.norm(w_out_ins).item()
        nat_out_norm = torch.linalg.norm(w_out_nat).item()
        if ins_out_norm < 1e-8 or nat_out_norm < 1e-8:
            continue

        if subblock_type in ("self_attn", "cross_attn"):
            q_align = 1.0
            k_align = 1.0
            if q_sfx and q_sfx in ins_suffix_map and q_sfx in nat_suffix_map:
                q_align = compute_matrix_cosine(
                    ins_suffix_map[q_sfx].to(device=device, dtype=torch.float32),
                    nat_suffix_map[q_sfx].to(device=device, dtype=torch.float32),
                )
            if k_sfx and k_sfx in ins_suffix_map and k_sfx in nat_suffix_map:
                k_align = compute_matrix_cosine(
                    ins_suffix_map[k_sfx].to(device=device, dtype=torch.float32),
                    nat_suffix_map[k_sfx].to(device=device, dtype=torch.float32),
                )

            qk_gate = math.sqrt(max(0.0, q_align * k_align))
            if subblock_type == "cross_attn":
                qk_gate *= 0.50 * k_align

            if qk_gate < 0.03:
                continue

            gram_nat = w_in_nat @ w_in_nat.T
            reg = 1e-2 * (torch.trace(gram_nat).item() / max(1, gram_nat.shape[0]))
            gram_reg = gram_nat + reg * torch.eye(gram_nat.shape[0], device=device, dtype=torch.float32)
            cross_cov = w_in_ins @ w_in_nat.T
            proj_basis = torch.linalg.solve(gram_reg, cross_cov.T).T
            projected_out = qk_gate * (w_out_ins @ proj_basis)
        else:
            in_ins_norm = torch.linalg.norm(w_in_ins, dim=1).clamp(min=1e-8)
            in_nat_norm = torch.linalg.norm(w_in_nat, dim=1).clamp(min=1e-8)
            row_cos = torch.sum(w_in_ins * w_in_nat, dim=1) / (in_ins_norm * in_nat_norm)
            diag_gate = torch.clamp(row_cos, min=0.0) ** 2
            scale_ratio = (in_ins_norm / in_nat_norm).clamp(max=1.5)
            proj_diag = w_out_ins * (diag_gate * scale_ratio).unsqueeze(0)

            gram_in = w_in_nat.T @ w_in_nat
            reg_in = 1e-2 * (torch.trace(gram_in).item() / max(1, gram_in.shape[0]))
            gram_in_reg = gram_in + reg_in * torch.eye(gram_in.shape[0], device=device, dtype=torch.float32)
            circuit_ins = w_out_ins @ w_in_ins
            z_sol = torch.linalg.solve(gram_in_reg, circuit_ins.T)
            proj_woodbury = z_sol.T @ w_in_nat.T

            mean_diag_align = torch.mean(torch.clamp(row_cos, min=0.0)).item()
            projected_out = (mean_diag_align * proj_diag) + ((1.0 - mean_diag_align) * proj_woodbury)

        proj_norm = torch.linalg.norm(projected_out).item()
        if proj_norm < 1e-8:
            continue

        if proj_norm > nat_out_norm:
            projected_out = projected_out * (nat_out_norm / proj_norm)

        nat_row_norm_sq = torch.sum(w_out_nat * w_out_nat, dim=1, keepdim=True).clamp(min=1e-12)
        dot_row = torch.sum(projected_out * w_out_nat, dim=1, keepdim=True)
        p_para = (dot_row / nat_row_norm_sq) * w_out_nat
        p_perp = projected_out - p_para

        effective_update = weight * (p_perp + 0.35 * torch.clamp(dot_row, min=0.0) / nat_row_norm_sq * w_out_nat)
        combined = w_out_nat + effective_update

        nat_row_norm = torch.sqrt(nat_row_norm_sq)
        max_allowed_row_norm = math.sqrt(1.0 + (weight ** 2)) * nat_row_norm
        comb_row_norm = torch.linalg.norm(combined, dim=1, keepdim=True).clamp(min=1e-8)
        clamp_scale = torch.clamp(max_allowed_row_norm / comb_row_norm, max=1.0)
        target_out = combined * clamp_scale

        block_deltas[out_sfx] = (target_out - w_out_nat).contiguous().cpu()

    return block_deltas


def downscale_dict_single_stage(state_dict, manifest, absorb_arg, ratio, device):
    low_count = int(manifest["old_block_count"])
    source_to_target = build_source_to_target_map(manifest)
    target_to_source = {v: k for k, v in source_to_target.items()}
    inserted_to_source = {int(k): int(v) for k, v in manifest.get("inserted_to_source", {}).items()}
    insertion_positions = set(manifest.get("insertion_positions", inserted_to_source))

    if str(absorb_arg).lower() == "auto":
        absorb_factor = analyze_inserted_blocks(state_dict, manifest, device=device)
    elif str(absorb_arg).startswith("surgical"):
        absorb_factor = 0.0
    else:
        try:
            absorb_factor = float(absorb_arg)
        except ValueError:
            absorb_factor = 0.0

    new_state_dict = {}
    keys_by_high_block = {}

    for k, v in state_dict.items():
        prefix, b_idx, suffix = parse_key_for_block(k)
        if b_idx is not None:
            keys_by_high_block.setdefault(b_idx, []).append((prefix, suffix, k))
        else:
            new_state_dict[k] = v

    for high_idx, keys in sorted(keys_by_high_block.items()):
        if high_idx in insertion_positions or high_idx not in target_to_source:
            continue
        low_idx = target_to_source[high_idx]
        for prefix, suffix, orig_key in keys:
            new_key = f"{prefix}{low_idx}{suffix}"
            new_state_dict[new_key] = state_dict[orig_key]

    if absorb_factor > 0.0:
        for ins_high_idx in sorted(insertion_positions):
            if ins_high_idx not in keys_by_high_block:
                continue
            front_low_idx = inserted_to_source.get(ins_high_idx)
            if front_low_idx is None:
                continue
            back_low_idx = front_low_idx + 1

            ins_entries = keys_by_high_block[ins_high_idx]
            if not ins_entries:
                continue
            block_prefix = ins_entries[0][0]
            ins_suffix_map = {suffix: state_dict[ins_key] for _, suffix, ins_key in ins_entries}

            front_nat_map = {
                suffix: new_state_dict[f"{block_prefix}{front_low_idx}{suffix}"]
                for suffix in ins_suffix_map
                if f"{block_prefix}{front_low_idx}{suffix}" in new_state_dict
            }
            back_nat_map = (
                {
                    suffix: new_state_dict[f"{block_prefix}{back_low_idx}{suffix}"]
                    for suffix in ins_suffix_map
                    if f"{block_prefix}{back_low_idx}{suffix}" in new_state_dict
                }
                if back_low_idx < low_count
                else {}
            )

            front_absorb = absorb_factor * ratio
            back_absorb = absorb_factor * (1.0 - ratio)

            if front_absorb > 0.0 and front_nat_map:
                f_deltas = compute_subspace_aligned_block_deltas(ins_suffix_map, front_nat_map, front_absorb, device)
                for out_sfx, d_tensor in f_deltas.items():
                    target_k = f"{block_prefix}{front_low_idx}{out_sfx}"
                    orig_t = new_state_dict[target_k]
                    new_state_dict[target_k] = (orig_t.float() + d_tensor).to(dtype=orig_t.dtype).contiguous().cpu()

            if back_absorb > 0.0 and back_nat_map:
                b_deltas = compute_subspace_aligned_block_deltas(ins_suffix_map, back_nat_map, back_absorb, device)
                for out_sfx, d_tensor in b_deltas.items():
                    target_k = f"{block_prefix}{back_low_idx}{out_sfx}"
                    orig_t = new_state_dict[target_k]
                    new_state_dict[target_k] = (orig_t.float() + d_tensor).to(dtype=orig_t.dtype).contiguous().cpu()

    return new_state_dict


def upscale_dict_single_stage(state_dict, manifest, fill_mode, ratio, device, is_lora=False):
    source_to_target = build_source_to_target_map(manifest)
    inserted_to_source = {int(k): int(v) for k, v in manifest.get("inserted_to_source", {}).items()}
    insertion_positions = manifest.get("insertion_positions", list(inserted_to_source))

    new_state_dict = {}
    keys_by_old_block = {}

    for k, v in state_dict.items():
        prefix, b_idx, suffix = parse_key_for_block(k)
        if b_idx is not None:
            keys_by_old_block.setdefault(b_idx, []).append((prefix, suffix, k))
        else:
            new_state_dict[k] = v

    for old_idx, keys in keys_by_old_block.items():
        if old_idx not in source_to_target:
            continue
        new_idx = source_to_target[old_idx]
        for prefix, suffix, orig_key in keys:
            new_key = f"{prefix}{new_idx}{suffix}"
            new_state_dict[new_key] = state_dict[orig_key]

    zero_gates = fill_mode in ("skip_inserted", "zero_gates")

    for new_idx in insertion_positions:
        front_old_idx = inserted_to_source.get(new_idx)
        if front_old_idx is None:
            continue
        back_old_idx = front_old_idx + 1

        for prefix, suffix, orig_front_key in keys_by_old_block.get(front_old_idx, []):
            new_key = f"{prefix}{new_idx}{suffix}"
            orig_t = state_dict[orig_front_key]

            should_zero = zero_gates and (is_lora or any(suffix.endswith(z) for z in ZERO_INIT_SUFFIXES))
            if should_zero:
                if not is_lora:
                    new_state_dict[new_key] = torch.zeros_like(orig_t).cpu()
                continue

            front_tensor = orig_t.to(device=device, dtype=torch.float32)
            orig_back_key = None
            for p, s, k in keys_by_old_block.get(back_old_idx, []):
                if p == prefix and s == suffix:
                    orig_back_key = k
                    break

            if orig_back_key and orig_back_key in state_dict:
                back_tensor = state_dict[orig_back_key].to(device=device, dtype=torch.float32)
                blended = (front_tensor * ratio) + (back_tensor * (1.0 - ratio))
            else:
                blended = front_tensor

            new_state_dict[new_key] = blended.to(dtype=orig_t.dtype).contiguous().cpu()

    skipped_blocks = set(insertion_positions) if fill_mode == "skip_inserted" else set()
    return new_state_dict, skipped_blocks


def convert_state_dict_to_tier(
    state_dict,
    current_tier,
    target_tier,
    absorb_mode="auto",
    upscale_fill_mode="skip_inserted",
    ratio=0.5,
    device="cpu",
    is_lora=False,
):
    if current_tier == target_tier or current_tier is None:
        return state_dict, set()

    sd = state_dict
    skipped = set()

    if current_tier > target_tier:
        if current_tier == 52 and target_tier in (40, 28):
            sd = downscale_dict_single_stage(sd, MANIFEST_40_TO_52, absorb_mode, ratio, device)
            current_tier = 40
        if current_tier == 40 and target_tier == 28:
            sd = downscale_dict_single_stage(sd, MANIFEST_28_TO_40, absorb_mode, ratio, device)
            current_tier = 28
        return sd, skipped

    if current_tier == 28 and target_tier == 40:
        sd, skipped = upscale_dict_single_stage(
            sd, MANIFEST_28_TO_40, upscale_fill_mode, ratio, device, is_lora=is_lora
        )

    return sd, skipped
