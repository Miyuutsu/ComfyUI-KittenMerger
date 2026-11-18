import json
import math
import os
import time
from datetime import UTC, datetime
import torch
from safetensors.torch import save_file

import comfy.model_management
import comfy.utils
import folder_paths

from .loader_nodes import DTYPE_MAP
from .utils import (
    ANY_TYPE,
    MAX_DYNAMIC_MODELS,
    DynamicUniversalInputs,
    materialize_patched_state_dict,
    send_kitten_progress,
    strip_diffusion_prefix,
)


def _get_kronecker_factor(dim, max_factor=8):
    for f in range(min(dim, max_factor), 1, -1):
        if dim % f == 0:
            return f
    return 1


def _normalize_kohya_lora_prefix(full_weight_key):
    base = full_weight_key[:-7] if full_weight_key.endswith(".weight") else full_weight_key
    for prefix in ("model.diffusion_model.", "diffusion_model.", "net."):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    if base.startswith("lora_unet_"):
        return base
    return f"lora_unet_{base.replace('.', '_')}"


def _compute_gavish_donoho_svht(singular_values, m, n):
    beta = min(m, n) / max(m, n)
    omega = 0.56 * (beta ** 3) - 0.95 * (beta ** 2) + 1.82 * beta + 1.43
    median_sv = torch.median(singular_values)
    tau = 0.25 * omega * median_sv
    return torch.clamp(singular_values - tau, min=1e-12)


def _decompose_loha(delta_f, rank, iterations=3):
    m, n = delta_f.shape
    eff_rank = min(rank, m, n)

    sign_mat = torch.sign(delta_f)
    sign_mat[sign_mat == 0] = 1.0
    sqrt_abs = torch.sqrt(torch.abs(delta_f) + 1e-12)

    init_1 = sign_mat * sqrt_abs
    init_2 = sqrt_abs

    u1, s1, vh1 = torch.linalg.svd(init_1, full_matrices=False)
    s1_sqrt = torch.sqrt(torch.clamp(s1[:eff_rank], min=1e-12))
    w1_a = u1[:, :eff_rank] * s1_sqrt.unsqueeze(0)
    w1_b = s1_sqrt.unsqueeze(1) * vh1[:eff_rank, :]

    u2, s2, vh2 = torch.linalg.svd(init_2, full_matrices=False)
    s2_sqrt = torch.sqrt(torch.clamp(s2[:eff_rank], min=1e-12))
    w2_a = u2[:, :eff_rank] * s2_sqrt.unsqueeze(0)
    w2_b = s2_sqrt.unsqueeze(1) * vh2[:eff_rank, :]

    for _ in range(iterations):
        m2 = w2_a @ w2_b
        denom2 = (m2 * m2) + 1e-4
        target_m1 = (delta_f * m2) / denom2
        u1, s1, vh1 = torch.linalg.svd(target_m1, full_matrices=False)
        s1_sqrt = torch.sqrt(torch.clamp(s1[:eff_rank], min=1e-12))
        w1_a = u1[:, :eff_rank] * s1_sqrt.unsqueeze(0)
        w1_b = s1_sqrt.unsqueeze(1) * vh1[:eff_rank, :]

        m1 = w1_a @ w1_b
        denom1 = (m1 * m1) + 1e-4
        target_m2 = (delta_f * m1) / denom1
        u2, s2, vh2 = torch.linalg.svd(target_m2, full_matrices=False)
        s2_sqrt = torch.sqrt(torch.clamp(s2[:eff_rank], min=1e-12))
        w2_a = u2[:, :eff_rank] * s2_sqrt.unsqueeze(0)
        w2_b = s2_sqrt.unsqueeze(1) * vh2[:eff_rank, :]

    return w1_a, w1_b, w2_a, w2_b, eff_rank


def _decompose_delta_matrix(
    delta_tensor,
    lora_format,
    rank,
    lokr_factor,
    out_dtype,
    device,
    clamp_lora=False,
):
    if delta_tensor is None or not delta_tensor.is_floating_point():
        return {}

    if lora_format == "full_diff (lossless step)" or delta_tensor.ndim != 2:
        return {"diff": delta_tensor.detach().to(device="cpu", dtype=out_dtype).contiguous()}

    delta_f = delta_tensor.detach().to(device=device, dtype=torch.float32)
    delta_norm = torch.linalg.norm(delta_f).item()
    if not math.isfinite(delta_norm) or delta_norm < 1e-7:
        return {}

    m, n = delta_f.shape

    if lora_format == "lokr":
        m1 = _get_kronecker_factor(m, lokr_factor)
        n1 = _get_kronecker_factor(n, lokr_factor)
        m2 = m // m1
        n2 = n // n1

        rearranged = delta_f.reshape(m1, m2, n1, n2).permute(0, 2, 1, 3).reshape(m1 * n1, m2 * n2)
        u_k, s_k, vh_k = torch.linalg.svd(rearranged, full_matrices=False)
        kron_principal_energy = s_k[0].item()

        s0_sqrt = math.sqrt(max(kron_principal_energy, 1e-12))
        w1 = (u_k[:, 0] * s0_sqrt).reshape(m1, n1)
        w2_full = (vh_k[0, :] * s0_sqrt).reshape(m2, n2)

        eff_rank = min(rank, m2, n2)
        u2, s2, vh2 = torch.linalg.svd(w2_full, full_matrices=False)
        s2_sqrt = torch.sqrt(torch.clamp(s2[:eff_rank], min=1e-12))

        w2_a = u2[:, :eff_rank] * s2_sqrt.unsqueeze(0)
        w2_b = s2_sqrt.unsqueeze(1) * vh2[:eff_rank, :]

        return {
            "lokr_w1": w1.to(dtype=out_dtype).contiguous().cpu(),
            "lokr_w2_a": w2_a.to(dtype=out_dtype).contiguous().cpu(),
            "lokr_w2_b": w2_b.to(dtype=out_dtype).contiguous().cpu(),
            "alpha": torch.tensor(float(eff_rank), dtype=torch.float32),
        }

    if lora_format == "loha":
        w1_a, w1_b, w2_a, w2_b, eff_rank = _decompose_loha(delta_f, rank)
        return {
            "hada_w1_a": w1_a.to(dtype=out_dtype).contiguous().cpu(),
            "hada_w1_b": w1_b.to(dtype=out_dtype).contiguous().cpu(),
            "hada_w2_a": w2_a.to(dtype=out_dtype).contiguous().cpu(),
            "hada_w2_b": w2_b.to(dtype=out_dtype).contiguous().cpu(),
            "alpha": torch.tensor(float(eff_rank), dtype=torch.float32),
        }

    # Default: "lora (SVD)"
    eff_rank = min(rank, m, n)
    u, s, vh = torch.linalg.svd(delta_f, full_matrices=False)
    s_clean = _compute_gavish_donoho_svht(s, m, n)[:eff_rank]

    if clamp_lora:
        m1 = _get_kronecker_factor(m, lokr_factor)
        n1 = _get_kronecker_factor(n, lokr_factor)
        m2 = m // m1
        n2 = n // n1
        rearranged = delta_f.reshape(m1, m2, n1, n2).permute(0, 2, 1, 3).reshape(m1 * n1, m2 * n2)
        _, s_k, _ = torch.linalg.svd(rearranged, full_matrices=False)
        kron_principal_energy = s_k[0].item()

        svd_norm = torch.linalg.norm(s_clean).item()
        if svd_norm > kron_principal_energy and svd_norm > 1e-8:
            target_svd_norm = 0.65 * kron_principal_energy + 0.35 * svd_norm
            s_clean = s_clean * (target_svd_norm / svd_norm)

    s_sqrt = torch.sqrt(torch.clamp(s_clean, min=1e-12))
    lora_up = u[:, :eff_rank] * s_sqrt.unsqueeze(0)
    lora_down = s_sqrt.unsqueeze(1) * vh[:eff_rank, :]

    return {
        "lora_up.weight": lora_up.to(dtype=out_dtype).contiguous().cpu(),
        "lora_down.weight": lora_down.to(dtype=out_dtype).contiguous().cpu(),
        "alpha": torch.tensor(float(eff_rank), dtype=torch.float32),
    }


class KittenConvertedComponent:
    """
    Lightweight wrapper around non-ModelPatcher objects (e.g. CLIP, VAE, raw dicts)
    carrying a target dtype override for AnimaModelSaver without mutating the original object in RAM.
    """
    def __init__(self, inner, target_dtype, display_name=None):
        self.inner = inner
        self._kitten_forced_dtype = target_dtype
        self._kitten_display_name = display_name or getattr(inner, "_kitten_display_name", None)


class AnimaDtypeConverter:
    """
    Multi-input Precision Converter. Takes any number of dynamic inputs (MODEL, LoRA, CLIP/TE, VAE)
    and tags/converts them to the selected precision (bf16, fp16, fp32, fp64, fp8).
    """

    @classmethod
    def INPUT_TYPES(cls):
        dtype_choices = ["bf16", "fp16", "fp32", "fp64"]
        if "fp8_e4m3fn" in DTYPE_MAP:
            dtype_choices.append("fp8_e4m3fn")
        if "fp8_e5m2" in DTYPE_MAP:
            dtype_choices.append("fp8_e5m2")

        return {
            "required": {
                "dtype": (
                    dtype_choices,
                    {
                        "default": "fp32",
                        "tooltip": "Target floating-point precision applied to all connected inputs.",
                    },
                ),
            },
            "optional": DynamicUniversalInputs({
                "input_1": (ANY_TYPE,),
            }),
        }

    RETURN_TYPES = (ANY_TYPE,) * MAX_DYNAMIC_MODELS
    RETURN_NAMES = tuple(f"output_{i}" for i in range(1, MAX_DYNAMIC_MODELS + 1))
    FUNCTION = "convert_dtypes"
    CATEGORY = "kitten_merger/savers"

    def _wrap_with_dtype(self, obj, target_dtype):
        if obj is None:
            return None

        if isinstance(obj, KittenConvertedComponent):
            return KittenConvertedComponent(
                obj.inner,
                target_dtype,
                display_name=obj._kitten_display_name,
            )

        if getattr(obj, "_kitten_is_lora", False):
            cloned = obj.clone()
            cloned._kitten_forced_dtype = target_dtype
            return cloned

        if hasattr(obj, "clone") and hasattr(obj, "model") and hasattr(obj, "patches"):
            cloned = obj.clone()
            cloned._kitten_forced_dtype = target_dtype
            return cloned

        return KittenConvertedComponent(obj, target_dtype)

    def convert_dtypes(self, dtype="fp32", **kwargs):
        target_dtype = DTYPE_MAP.get(dtype, torch.float32)
        outputs = []
        for i in range(1, MAX_DYNAMIC_MODELS + 1):
            inp = kwargs.get(f"input_{i}")
            outputs.append(self._wrap_with_dtype(inp, target_dtype))
        return tuple(outputs)


def _classify_component(obj):
    raw = obj.inner if isinstance(obj, KittenConvertedComponent) else obj
    forced_dtype = getattr(obj, "_kitten_forced_dtype", None) or getattr(raw, "_kitten_forced_dtype", None)

    if raw is None:
        return "none", None, None

    if isinstance(raw, str):
        return "schema", raw, None

    if getattr(raw, "_kitten_is_lora", False):
        return "lora", raw, forced_dtype

    if getattr(raw, "_kitten_is_virtual_stage", False):
        return "virtual_stage", raw, forced_dtype

    if hasattr(raw, "model") and hasattr(raw, "patches"):
        return "model", raw, forced_dtype

    if hasattr(raw, "cond_stage_model") or (hasattr(raw, "patcher") and hasattr(raw, "get_sd")):
        return "clip", raw, forced_dtype

    if hasattr(raw, "first_stage_model") and hasattr(raw, "get_sd"):
        return "vae", raw, forced_dtype

    if isinstance(raw, dict):
        return "dict", raw, forced_dtype

    if hasattr(raw, "get_sd") and callable(raw.get_sd):
        return "generic_sd", raw, forced_dtype

    return "unknown", raw, forced_dtype


def _cast_state_dict(sd, forced_dtype):
    out = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        t = v.detach()
        if forced_dtype is not None and t.is_floating_point() and t.dtype != forced_dtype:
            t = t.to(device="cpu", dtype=forced_dtype).contiguous()
        else:
            t = t.cpu().contiguous()
        out[k] = t
    return out


class AnimaModelSaver:
    """
    Universal Model / Checkpoint / LoRA Saver & Muxer.
    Supports saving each connected input to its own separate file (default) or muxing
    compatible components into a single container, plus SVD / LoKr / LoHa / Lossless-Diff LoRA saving.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "diffusion_models/ComfyUI",
                        "tooltip": "Subfolder and filename prefix inside ComfyUI's output directory (e.g. 'diffusion_models/ComfyUI' -> output/diffusion_models/ComfyUI_00001_.safetensors).",
                    },
                ),
                "save_mode": (
                    ["separate_files", "single_container"],
                    {
                        "default": "separate_files",
                        "tooltip": "separate_files: saves each connected input to its own .safetensors file. single_container: muxes connected Model/CLIP/VAE inputs into one file (LoRAs are always saved separately).",
                    },
                ),
                "lora_format": (
                    ["lora (SVD)", "lokr", "loha", "full_diff (lossless step)"],
                    {
                        "default": "lora (SVD)",
                        "tooltip": "Format used when saving a LoRA input. Use 'lora (SVD)', 'lokr', or 'loha' for compact Kohya adapters, or 'full_diff (lossless step)' for zero-loss intermediate step saving.",
                    },
                ),
                "lora_rank": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Target rank when decomposing merged LoRAs into 'lora (SVD)', 'lokr', or 'loha'.",
                    },
                ),
                "lokr_factor": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Kronecker factor when lora_format is 'lokr'.",
                    },
                ),
                "clamp_lora": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled on lora (SVD), runs an extra Kronecker SVD pass per layer to clamp singular value energy. Default off for 2x faster SVD saving.",
                    },
                ),
                "dora_scale": (
                    ["passthrough", "merge", "drop"],
                    {
                        "default": "passthrough",
                        "tooltip": "passthrough: keeps the primary LoRA's original dora_scale untouched. merge: blends dora_scale across stages. drop: omits dora_scale when saving.",
                    },
                ),
            },
            "optional": DynamicUniversalInputs({
                "merge_schema": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Optional: Connect merge_schema from Anima Model Merger to embed the merge recipe into the .safetensors metadata header.",
                    },
                ),
                "input_1": (ANY_TYPE,),
            }),
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_muxed_model"
    OUTPUT_NODE = True
    CATEGORY = "kitten_merger/savers"

    def _extract_lora_sd(
        self,
        raw_obj,
        forced_dtype,
        lora_format,
        lora_rank,
        lokr_factor,
        clamp_lora,
        dora_scale_mode,
        device,
        unique_id=None,
        pbar=None,
    ):
        out_dtype = forced_dtype
        if out_dtype is None:
            if raw_obj._precomputed_sd:
                first_t = next(iter(raw_obj._precomputed_sd.values()), None)
                out_dtype = first_t.dtype if isinstance(first_t, torch.Tensor) else torch.bfloat16
            else:
                out_dtype = torch.bfloat16

        # Fast path: unmerged or single-LoRA-scaled standalone LoRA with existing low-rank (up, down) tensors
        if (
            raw_obj._precomputed_sd is None
            and raw_obj.lora_entries
            and lora_format == "lora (SVD)"
        ):
            out_sd = {}
            base_strength = float(getattr(raw_obj, "strength", 1.0))
            for k, entry in raw_obj.lora_entries.items():
                if not isinstance(entry, dict):
                    continue
                entry_scale = float(entry.get("scale", 1.0))
                eff_strength = base_strength * entry_scale
                if abs(eff_strength) <= 1e-7:
                    continue
                prefix = _normalize_kohya_lora_prefix(k)
                if "up" in entry and "down" in entry and entry["up"] is not None and entry["down"] is not None:
                    up_t = entry["up"].detach()
                    down_t = entry["down"].detach().to(device="cpu", dtype=out_dtype)
                    rank = down_t.shape[0]
                    raw_alpha = float(entry.get("alpha") if entry.get("alpha") is not None else rank)
                    if eff_strength < 0.0:
                        up_t = (-up_t.float()).to(device="cpu", dtype=out_dtype)
                        alpha = raw_alpha * abs(eff_strength)
                    else:
                        up_t = up_t.to(device="cpu", dtype=out_dtype)
                        alpha = raw_alpha * eff_strength
                    out_sd[f"{prefix}.lora_up.weight"] = up_t.contiguous()
                    out_sd[f"{prefix}.lora_down.weight"] = down_t.contiguous()
                    out_sd[f"{prefix}.alpha"] = torch.tensor(float(alpha), dtype=torch.float32)
                    if dora_scale_mode != "drop" and hasattr(raw_obj, "get_dora_scale"):
                        ds = raw_obj.get_dora_scale(k, mode=dora_scale_mode)
                        if ds is not None:
                            out_sd[f"{prefix}.dora_scale"] = ds.detach().to(device="cpu", dtype=out_dtype).contiguous()
                elif "diff" in entry and entry["diff"] is not None:
                    diff_t = (entry["diff"].detach().float() * eff_strength).to(dtype=out_dtype).contiguous().cpu()
                    out_sd[f"{prefix}.diff"] = diff_t
            if out_sd:
                return out_sd

        keys = sorted(raw_obj.get_key_set(), key=str)
        total_k = max(1, len(keys))
        out_sd = {}

        for idx, k in enumerate(keys, start=1):
            if idx % 16 == 0 or idx == total_k:
                send_kitten_progress(
                    unique_id,
                    f"Decomposing LoRA ({lora_format}, rank={lora_rank})",
                    f"Key {idx}/{total_k}",
                    idx,
                    total_k,
                )
                if pbar is not None:
                    pbar.update_absolute(int((idx / total_k) * 75), 100)

            delta_t = raw_obj.materialize_key(k, dtype=torch.float32, device=device)
            if delta_t is None:
                continue
            prefix = _normalize_kohya_lora_prefix(k)
            layer_dict = _decompose_delta_matrix(
                delta_t,
                lora_format=lora_format,
                rank=int(lora_rank),
                lokr_factor=int(lokr_factor),
                out_dtype=out_dtype,
                device=device,
                clamp_lora=bool(clamp_lora),
            )
            if layer_dict:
                for sfx, tensor in layer_dict.items():
                    out_sd[f"{prefix}.{sfx}"] = tensor
                if dora_scale_mode != "drop" and hasattr(raw_obj, "get_dora_scale"):
                    ds = raw_obj.get_dora_scale(k, mode=dora_scale_mode)
                    if ds is not None:
                        out_sd[f"{prefix}.dora_scale"] = ds.detach().to(device="cpu", dtype=out_dtype).contiguous()
            del delta_t

        return out_sd

    def _extract_component_sd(
        self,
        kind,
        raw_obj,
        forced_dtype,
        idx_in_kind,
        total_in_kind,
        is_multi_component_mux,
        lora_format,
        lora_rank,
        lokr_factor,
        clamp_lora,
        dora_scale_mode,
        device,
        unique_id=None,
        pbar=None,
    ):
        if kind == "lora":
            return self._extract_lora_sd(
                raw_obj,
                forced_dtype=forced_dtype,
                lora_format=lora_format,
                lora_rank=lora_rank,
                lokr_factor=lokr_factor,
                clamp_lora=clamp_lora,
                dora_scale_mode=dora_scale_mode,
                device=device,
                unique_id=unique_id,
                pbar=pbar,
            )

        if kind == "model":
            sd = materialize_patched_state_dict(raw_obj, device=device)
            casted = _cast_state_dict(sd, forced_dtype)
            aligned = {}
            for k, v in casted.items():
                clean_k = strip_diffusion_prefix(k)
                if clean_k.startswith("model_sampling."):
                    continue
                if is_multi_component_mux:
                    if total_in_kind > 1:
                        pfx = f"model.diffusion_model_{idx_in_kind}."
                    else:
                        pfx = "model.diffusion_model."
                    aligned[f"{pfx}{clean_k}"] = v
                else:
                    if total_in_kind > 1:
                        pfx = f"net_{idx_in_kind}."
                        aligned[f"{pfx}{clean_k}"] = v
                    else:
                        aligned[f"net.{clean_k}"] = v
            return aligned

        if kind == "clip":
            if hasattr(raw_obj, "patcher") and hasattr(raw_obj.patcher, "model"):
                sd = materialize_patched_state_dict(raw_obj.patcher, device=device)
            elif hasattr(raw_obj, "get_sd"):
                sd = raw_obj.get_sd()
            else:
                sd = {}
            casted = _cast_state_dict(sd, forced_dtype)
            aligned = {}
            has_known_te_prefix = any(
                k.startswith(("cond_stage_model.", "text_encoders.", "clip.", "t5xxl.", "qwen.", "conditioner."))
                for k in casted.keys()
            )
            for k, v in casted.items():
                if has_known_te_prefix or not is_multi_component_mux:
                    if total_in_kind > 1 and not k.startswith(("text_encoders.", "cond_stage_model.")):
                        aligned[f"text_encoders.te_{idx_in_kind}.{k}"] = v
                    else:
                        aligned[k] = v
                else:
                    pfx = f"text_encoders.te_{idx_in_kind}." if total_in_kind > 1 else "cond_stage_model."
                    aligned[f"{pfx}{k}"] = v
            return aligned

        if kind == "vae":
            if hasattr(raw_obj, "patcher") and hasattr(raw_obj.patcher, "model"):
                sd = materialize_patched_state_dict(raw_obj.patcher, device=device)
            elif hasattr(raw_obj, "get_sd"):
                sd = raw_obj.get_sd()
            else:
                sd = {}
            casted = _cast_state_dict(sd, forced_dtype)
            aligned = {}
            for k, v in casted.items():
                clean_v_k = k[len("first_stage_model."):] if k.startswith("first_stage_model.") else (
                    k[len("vae."):] if k.startswith("vae.") else k
                )
                if is_multi_component_mux:
                    pfx = f"first_stage_model_{idx_in_kind}." if total_in_kind > 1 else "first_stage_model."
                    aligned[f"{pfx}{clean_v_k}"] = v
                else:
                    pfx = f"vae_{idx_in_kind}." if total_in_kind > 1 else ""
                    aligned[f"{pfx}{clean_v_k}"] = v
            return aligned

        if kind == "dict":
            return _cast_state_dict(raw_obj, forced_dtype)

        if kind == "generic_sd":
            return _cast_state_dict(raw_obj.get_sd(), forced_dtype)

        return {}

    def _write_single_safetensors(self, sd, filename_prefix, is_lora, component_summaries, schema_str, prompt, extra_pnginfo):
        out_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir
        )
        os.makedirs(full_output_folder, exist_ok=True)

        while True:
            out_filename = f"{filename}_{counter:05}_.safetensors"
            out_path = os.path.join(full_output_folder, out_filename)
            if not os.path.exists(out_path):
                break
            counter += 1

        out_stem = out_filename[:-len(".safetensors")] if out_filename.endswith(".safetensors") else out_filename
        metadata = {
            "modelspec.architecture": "anima/lora" if is_lora else "anima",
            "modelspec.prediction_type": "sd3",
            "modelspec.resolution": "2048x2048",
            "modelspec.title": out_stem,
            "modelspec.date": datetime.now(UTC).astimezone().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if prompt is not None:
            try:
                metadata["prompt"] = json.dumps(prompt)
            except Exception:
                pass
        if extra_pnginfo is not None and isinstance(extra_pnginfo, dict):
            for k, v in extra_pnginfo.items():
                try:
                    metadata[k] = json.dumps(v)
                except Exception:
                    pass
        if schema_str:
            metadata["kitten_merge_schema"] = schema_str
        metadata["kitten_mux_components"] = " | ".join(component_summaries)

        save_file(sd, out_path, metadata=metadata)
        return out_path, out_filename

    def save_muxed_model(
        self,
        filename_prefix="diffusion_models/ComfyUI",
        save_mode="separate_files",
        lora_format="lora (SVD)",
        lora_rank=32,
        lokr_factor=8,
        clamp_lora=False,
        dora_scale="passthrough",
        merge_schema=None,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        **kwargs,
    ):
        t_start = time.perf_counter()
        device = comfy.model_management.get_torch_device()

        classified_inputs = []
        counts_by_kind = {}
        schema_str = str(merge_schema).strip() if merge_schema else ""

        for i in range(1, MAX_DYNAMIC_MODELS + 1):
            inp = kwargs.get(f"input_{i}")
            if inp is None:
                continue
            kind, raw_obj, forced_dtype = _classify_component(inp)
            if kind == "none":
                continue
            if kind == "schema":
                if not schema_str:
                    schema_str = str(raw_obj).strip()
                continue
            if kind == "virtual_stage":
                raise ValueError(
                    f"AnimaModelSaver: input_{i} is an unwired VirtualStageModel ({getattr(raw_obj, '_kitten_stage_id', 'stage')}). "
                    f"Wire the stage's MODEL output slot directly from Anima Model Merger so it materializes in RAM."
                )
            if kind == "unknown":
                continue
            counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
            classified_inputs.append((i, kind, raw_obj, forced_dtype, counts_by_kind[kind]))

        if not classified_inputs:
            msg = "AnimaModelSaver: No valid model/LoRA/CLIP/VAE inputs connected. Nothing saved."
            print(f"[KittenMerger] {msg}")  # noqa: T201
            return {"ui": {"saved_path": [msg]}, "result": (msg,)}

        pbar = comfy.utils.ProgressBar(100)
        saved_paths = []
        all_summaries = []

        if save_mode == "separate_files":
            total_items = len(classified_inputs)
            for step_idx, (slot_num, kind, raw_obj, forced_dtype, _) in enumerate(classified_inputs, start=1):
                send_kitten_progress(
                    unique_id,
                    f"Saving Input {step_idx}/{total_items} ({kind.upper()}: input_{slot_num})",
                    filename_prefix,
                    step_idx - 1,
                    total_items,
                )
                comp_sd = self._extract_component_sd(
                    kind=kind,
                    raw_obj=raw_obj,
                    forced_dtype=forced_dtype,
                    idx_in_kind=1,
                    total_in_kind=1,
                    is_multi_component_mux=False,
                    lora_format=lora_format,
                    lora_rank=lora_rank,
                    lokr_factor=lokr_factor,
                    clamp_lora=clamp_lora,
                    dora_scale_mode=dora_scale,
                    device=device,
                    unique_id=unique_id,
                    pbar=pbar,
                )
                if not comp_sd:
                    continue
                sample_dt = next(
                    (str(t.dtype).replace("torch.", "") for t in comp_sd.values() if isinstance(t, torch.Tensor) and t.ndim >= 2),
                    "empty",
                )
                disp = getattr(raw_obj, "_kitten_display_name", None) or f"input_{slot_num}"
                summary = f"{kind}:{disp}[{sample_dt}, {len(comp_sd)} keys]"
                out_path, _ = self._write_single_safetensors(
                    comp_sd,
                    filename_prefix=filename_prefix,
                    is_lora=(kind == "lora"),
                    component_summaries=[summary],
                    schema_str=schema_str,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )
                comp_sd.clear()
                saved_paths.append(out_path)
                all_summaries.append(summary)
                pbar.update_absolute(int((step_idx / total_items) * 100), 100)
        else:
            lora_inputs = [x for x in classified_inputs if x[1] == "lora"]
            mux_inputs = [x for x in classified_inputs if x[1] != "lora"]
            mux_kinds = {x[1] for x in mux_inputs}
            is_multi_component_mux = len(mux_kinds) > 1
            mux_counts = {}
            for _, k_type, _, _, _ in mux_inputs:
                mux_counts[k_type] = mux_counts.get(k_type, 0) + 1

            if mux_inputs:
                combined_sd = {}
                mux_summaries = []
                for step_idx, (slot_num, kind, raw_obj, forced_dtype, idx_in_kind) in enumerate(mux_inputs, start=1):
                    send_kitten_progress(
                        unique_id,
                        f"Step 1/2: Materializing {kind.upper()} (input_{slot_num})",
                        f"Component {step_idx}/{len(mux_inputs)}",
                        step_idx - 1,
                        len(mux_inputs),
                    )
                    comp_sd = self._extract_component_sd(
                        kind=kind,
                        raw_obj=raw_obj,
                        forced_dtype=forced_dtype,
                        idx_in_kind=idx_in_kind,
                        total_in_kind=mux_counts[kind],
                        is_multi_component_mux=is_multi_component_mux,
                        lora_format=lora_format,
                        lora_rank=lora_rank,
                        lokr_factor=lokr_factor,
                        clamp_lora=clamp_lora,
                        dora_scale_mode=dora_scale,
                        device=device,
                        unique_id=unique_id,
                        pbar=pbar,
                    )
                    sample_dt = next(
                        (str(t.dtype).replace("torch.", "") for t in comp_sd.values() if isinstance(t, torch.Tensor)),
                        "empty",
                    )
                    disp = getattr(raw_obj, "_kitten_display_name", None) or f"input_{slot_num}"
                    mux_summaries.append(f"{kind}:{disp}[{sample_dt}, {len(comp_sd)} keys]")
                    combined_sd.update(comp_sd)

                if combined_sd:
                    send_kitten_progress(unique_id, "Step 2/2: Writing Container", filename_prefix, 85, 100)
                    out_path, _ = self._write_single_safetensors(
                        combined_sd,
                        filename_prefix=filename_prefix,
                        is_lora=False,
                        component_summaries=mux_summaries,
                        schema_str=schema_str,
                        prompt=prompt,
                        extra_pnginfo=extra_pnginfo,
                    )
                    combined_sd.clear()
                    saved_paths.append(out_path)
                    all_summaries.extend(mux_summaries)

            for slot_num, kind, raw_obj, forced_dtype, _ in lora_inputs:
                comp_sd = self._extract_component_sd(
                    kind="lora",
                    raw_obj=raw_obj,
                    forced_dtype=forced_dtype,
                    idx_in_kind=1,
                    total_in_kind=1,
                    is_multi_component_mux=False,
                    lora_format=lora_format,
                    lora_rank=lora_rank,
                    lokr_factor=lokr_factor,
                    clamp_lora=clamp_lora,
                    dora_scale_mode=dora_scale,
                    device=device,
                    unique_id=unique_id,
                    pbar=pbar,
                )
                if comp_sd:
                    disp = getattr(raw_obj, "_kitten_display_name", None) or f"input_{slot_num}"
                    summary = f"lora:{disp}[{len(comp_sd)} keys]"
                    out_path, _ = self._write_single_safetensors(
                        comp_sd,
                        filename_prefix=filename_prefix,
                        is_lora=True,
                        component_summaries=[summary],
                        schema_str=schema_str,
                        prompt=prompt,
                        extra_pnginfo=extra_pnginfo,
                    )
                    comp_sd.clear()
                    saved_paths.append(out_path)
                    all_summaries.append(summary)

        elapsed = time.perf_counter() - t_start
        pbar.update_absolute(100, 100)
        result_str = "\n".join(saved_paths) if saved_paths else "Nothing saved."
        send_kitten_progress(unique_id, "done", f"Saved {len(saved_paths)} file(s) ({elapsed:.2f}s)", 100, 100)
        comfy.utils.common_options = {"verbose": True}
        print(  # noqa: T201
             f"[KittenMerger] AnimaModelSaver saved {len(saved_paths)} file(s) in {elapsed:.2f}s: "
             f"{', '.join(saved_paths)} ({' | '.join(all_summaries)})"
         )

        return {"ui": {"saved_path": [result_str]}, "result": (result_str,)}
