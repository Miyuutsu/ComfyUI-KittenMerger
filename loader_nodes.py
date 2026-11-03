import inspect
import os
import re
import uuid
import torch
from safetensors import safe_open

import folder_paths
import comfy.lora
import comfy.model_management
import comfy.sd
import comfy.utils

from .utils import natural_sort_key, strip_diffusion_prefix

NON_DIFFUSION_PREFIXES = (
    "first_stage_model.",
    "cond_stage_model.",
    "text_encoders.",
    "conditioner.",
    "vae.",
    "clip.",
    "t5xxl.",
    "qwen.",
)

DIFFUSION_ROOT_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "net.",
    "blocks.",
    "llm_adapter.",
    "pos_embedder.",
    "x_embedder.",
    "t_embedder.",
    "t_embedding_norm.",
    "final_layer.",
)

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}
if hasattr(torch, "float8_e4m3fn"):
    DTYPE_MAP["fp8_e4m3fn"] = torch.float8_e4m3fn
if hasattr(torch, "float8_e5m2"):
    DTYPE_MAP["fp8_e5m2"] = torch.float8_e5m2


def clean_model_display_name(raw_name):
    if not raw_name:
        return ""
    s = re.sub(r"^\[[^\]]+\]\s*", "", str(raw_name).strip())
    s = s.replace("\\", "/").split("/")[-1]
    s = re.sub(r"\.(safetensors|sft|ckpt|pt|pth|bin)$", "", s, flags=re.IGNORECASE)
    return s


def build_anima_lora_key_map(max_blocks=60):
    """
    Builds a mapping of Kohya/sd-scripts (lora_unet_*) and standard Anima diffusion keys.
    Low-rank matrices always map to `.weight` so `.bias` never overwrites `.weight`.
    """
    weight_targets = [
        "pos_embedder.proj.weight",
        "x_embedder.proj.weight",
        "x_embedder.weight",
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.2.weight",
        "t_embedding_norm.weight",
        "final_layer.adaln_modulation.1.weight",
        "final_layer.adaln_modulation.2.weight",
        "final_layer.linear.weight",
        "llm_adapter.embed.weight",
        "llm_adapter.norm.weight",
        "llm_adapter.out_proj.weight",
    ]

    for i in range(8):
        for sub in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.output_proj",
            "cross_attn.q_proj",
            "cross_attn.k_proj",
            "cross_attn.v_proj",
            "cross_attn.o_proj",
            "cross_attn.output_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "mlp.layer1",
            "mlp.layer2",
            "norm1",
            "norm2",
            "input_layernorm",
            "post_attention_layernorm",
        ):
            weight_targets.append(f"llm_adapter.blocks.{i}.{sub}.weight")

    block_weight_suffixes = (
        "adaln_modulation_self_attn.1.weight",
        "adaln_modulation_self_attn.2.weight",
        "adaln_modulation_cross_attn.1.weight",
        "adaln_modulation_cross_attn.2.weight",
        "adaln_modulation_mlp.1.weight",
        "adaln_modulation_mlp.2.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_proj.weight",
        "self_attn.k_norm.weight",
        "self_attn.v_proj.weight",
        "self_attn.output_proj.weight",
        "cross_attn.q_proj.weight",
        "cross_attn.q_norm.weight",
        "cross_attn.k_proj.weight",
        "cross_attn.k_norm.weight",
        "cross_attn.v_proj.weight",
        "cross_attn.output_proj.weight",
        "mlp.layer1.weight",
        "mlp.layer2.weight",
    )

    for b in range(max_blocks):
        for sfx in block_weight_suffixes:
            weight_targets.append(f"blocks.{b}.{sfx}")

    key_map = {}
    underscore_to_weight = {}

    for clean_k in weight_targets:
        full_k = f"diffusion_model.{clean_k}"
        no_ext = clean_k[:-7]  # strip ".weight"
        under_name = no_ext.replace(".", "_")
        underscore_to_weight[under_name] = clean_k

        key_map[no_ext] = full_k
        key_map[f"diffusion_model.{no_ext}"] = full_k
        key_map[f"model.diffusion_model.{no_ext}"] = full_k
        key_map[f"net.{no_ext}"] = full_k
        key_map[f"transformer.{no_ext}"] = full_k
        key_map[f"lora_unet_{under_name}"] = full_k
        key_map[f"lora_transformer_{under_name}"] = full_k

    return key_map, underscore_to_weight


ANIMA_LORA_KEY_MAP, ANIMA_UNDERSCORE_WEIGHT_MAP = build_anima_lora_key_map()


def resolve_underscore_module_to_clean(u_part, is_bias=False):
    ext = ".bias" if is_bias else ".weight"
    if u_part in ANIMA_UNDERSCORE_WEIGHT_MAP:
        base_w = ANIMA_UNDERSCORE_WEIGHT_MAP[u_part]
        return (base_w[:-7] + ext) if is_bias else base_w

    # Dynamic regex fallback for any block index or sub-layer
    m_blk = re.match(
        r"^blocks_(\d+)_(adaln_modulation_self_attn|adaln_modulation_cross_attn|adaln_modulation_mlp|self_attn|cross_attn|mlp)_(.+)$",
        u_part,
    )
    if m_blk:
        b_idx, sub, rest = m_blk.groups()
        rest_dots = rest.replace("_proj", ".proj").replace(".", "_")
        if rest_dots in ("q_proj", "k_proj", "v_proj", "output_proj", "q_norm", "k_norm", "layer1", "layer2", "1", "2"):
            return f"blocks.{b_idx}.{sub}.{rest}{ext}"

    m_llm = re.match(r"^llm_adapter_blocks_(\d+)_(self_attn|cross_attn|mlp)_(.+)$", u_part)
    if m_llm:
        b_idx, sub, rest = m_llm.groups()
        return f"llm_adapter.blocks.{b_idx}.{sub}.{rest}{ext}"

    return f"{u_part.replace('_', '.')}{ext}"


def parse_direct_lora_pairs(lora_sd):
    """
    Direct parser for LoRA keys.
    Returns dict: { "diffusion_model.<clean_key>": {"up": Tensor, "down": Tensor, "alpha": float|None, "diff": Tensor|None} }
    """
    entries = {}

    def normalize_base_to_key(raw_base, is_bias=False):
        b = raw_base
        for pfx in ("model.diffusion_model.", "diffusion_model.", "net.", "transformer.", "model."):
            if b.startswith(pfx):
                b = b[len(pfx):]
                break
        if b.startswith(("lora_unet_", "lora_transformer_")):
            u_part = re.sub(r"^lora_(?:unet|transformer)_", "", b)
            clean_k = resolve_underscore_module_to_clean(u_part, is_bias=is_bias)
            return f"diffusion_model.{clean_k}"
        if is_bias:
            if b.endswith(".weight"):
                b = b[:-7] + ".bias"
            elif not b.endswith(".bias"):
                b = f"{b}.bias"
        else:
            if b.endswith(".bias"):
                b = b[:-5] + ".weight"
            elif not b.endswith(".weight"):
                b = f"{b}.weight"
        return f"diffusion_model.{b}"

    for k, v in lora_sd.items():
        if k.startswith(("lora_te", "text_encoder", "clip", "t5", "qwen", "vae.")):
            continue

        m_down = re.match(r"^(.+?)\.(?:lora_down|lora_A|lora\.down)(?:\.weight)?$", k)
        if m_down:
            target_k = normalize_base_to_key(m_down.group(1), is_bias=False)
            entries.setdefault(target_k, {})["down"] = v
            continue

        m_up = re.match(r"^(.+?)\.(?:lora_up|lora_B|lora\.up)(?:\.weight)?$", k)
        if m_up:
            target_k = normalize_base_to_key(m_up.group(1), is_bias=False)
            entries.setdefault(target_k, {})["up"] = v
            continue

        m_alpha = re.match(r"^(.+?)\.(?:alpha|lora_alpha)$", k)
        if m_alpha:
            target_k = normalize_base_to_key(m_alpha.group(1), is_bias=False)
            val = float(v.item()) if isinstance(v, torch.Tensor) and v.numel() == 1 else float(v)
            entries.setdefault(target_k, {})["alpha"] = val
            continue

        m_diff = re.match(r"^(.+?)\.diff$", k)
        if m_diff:
            target_k = normalize_base_to_key(m_diff.group(1), is_bias=False)
            entries.setdefault(target_k, {})["diff"] = v
            continue

        m_diff_b = re.match(r"^(.+?)\.(?:diff_b|lora_bias)$", k)
        if m_diff_b:
            target_k = normalize_base_to_key(m_diff_b.group(1), is_bias=True)
            entries.setdefault(target_k, {})["diff"] = v
            continue

    return entries


def materialize_single_lora_entry(patch_entry, strength, dtype, device):
    """
    Computes the full-rank delta tensor ΔW in `dtype` on `device` from either a
    direct parsed dict or a comfy.lora patch tuple.
    """
    if abs(strength) <= 1e-9:
        return None

    if isinstance(patch_entry, dict):
        if "diff" in patch_entry and patch_entry["diff"] is not None:
            return patch_entry["diff"].to(device=device, dtype=dtype) * float(strength)
        up = patch_entry.get("up")
        down = patch_entry.get("down")
        if up is None or down is None:
            return None
        rank = down.shape[0]
        alpha = patch_entry.get("alpha")
        scale = (float(alpha) / float(rank)) if alpha is not None else 1.0
        scale *= float(strength)

        up_t = up.to(device=device, dtype=dtype)
        down_t = down.to(device=device, dtype=dtype)
        if up_t.ndim == 2 and down_t.ndim == 2:
            return (up_t @ down_t) * scale
        return (up_t.flatten(1) @ down_t.flatten(1)).reshape(up_t.shape[0], *down_t.shape[1:]) * scale

    if isinstance(patch_entry, tuple) and len(patch_entry) >= 2:
        p_type = patch_entry[0]
        p_data = patch_entry[1]

        if p_type == "diff" and isinstance(p_data, (tuple, list)) and len(p_data) >= 1:
            return p_data[0].to(device=device, dtype=dtype) * float(strength)

        adapter = p_data if isinstance(p_data, tuple) else patch_entry
        if isinstance(adapter, tuple) and len(adapter) >= 2 and isinstance(adapter[0], str):
            a_type = adapter[0]
            a_vals = adapter[1]
            if a_type == "lora" and isinstance(a_vals, (tuple, list)) and len(a_vals) >= 2:
                up, down = a_vals[0], a_vals[1]
                alpha = a_vals[2] if len(a_vals) > 2 else None
                mid = a_vals[3] if len(a_vals) > 3 else None
                if mid is None:
                    rank = down.shape[0]
                    scale = (float(alpha) / float(rank)) if alpha is not None else 1.0
                    scale *= float(strength)
                    up_t = up.to(device=device, dtype=dtype)
                    down_t = down.to(device=device, dtype=dtype)
                    if up_t.ndim == 2 and down_t.ndim == 2:
                        return (up_t @ down_t) * scale
                    return (up_t.flatten(1) @ down_t.flatten(1)).reshape(up_t.shape[0], *down_t.shape[1:]) * scale

        if hasattr(comfy.lora, "calculate_weight"):
            try:
                if isinstance(adapter, tuple) and len(adapter) >= 2 and isinstance(adapter[1], (tuple, list)):
                    v = adapter[1]
                    if len(v) >= 2 and hasattr(v[0], "shape") and hasattr(v[1], "shape"):
                        out_shape = (v[0].shape[0], v[1].shape[1], *v[1].shape[2:])
                        zero_w = torch.zeros(out_shape, dtype=dtype, device=device)
                        patched = comfy.lora.calculate_weight(
                            [(strength, adapter, 1.0, None, None)],
                            zero_w,
                            "lora_key",
                            intermediate_dtype=dtype,
                        )
                        return patched.to(dtype=dtype)
            except Exception:
                pass

    return None


class _DummyInnerModel:
    def __init__(self, state_dict_fn):
        self._state_dict_fn = state_dict_fn

    def state_dict(self):
        return self._state_dict_fn()


class KittenLoRAModel:
    """
    A ModelPatcher-compatible wrapper representing a pure standalone LoRA (or merged LoRA).
    Can be wired directly into AnimaModelMerger or ModelMergeInspector on MODEL slots.
    Materializes ΔW tensors at the Merger's exact calculation precision (fp32/fp64/bf16).
    """

    def __init__(self, lora_entries=None, strength=1.0, precomputed_sd=None, name="lora"):
        self._kitten_is_lora = True
        self._kitten_merged = False
        self._kitten_stage_id = None
        self._kitten_source_id = None
        self._kitten_display_name = clean_model_display_name(name)
        self.name = name
        self.strength = float(strength)
        self.lora_entries = dict(lora_entries) if lora_entries else {}
        self._precomputed_sd = dict(precomputed_sd) if precomputed_sd is not None else None
        self._cached_fp32_sd = None
        self.patches = {}
        self.patches_uuid = uuid.uuid4()
        self.model = _DummyInnerModel(self._get_default_state_dict)

    def _get_default_state_dict(self):
        if self._precomputed_sd is not None:
            return self._precomputed_sd
        if self._cached_fp32_sd is None:
            device = comfy.model_management.get_torch_device()
            self._cached_fp32_sd = self.materialize_state_dict(dtype=torch.float32, device=device)
        return self._cached_fp32_sd

    def materialize_state_dict(self, dtype=torch.float32, device=None):
        if self._precomputed_sd is not None:
            return {
                k: v.to(dtype=dtype).contiguous().cpu()
                for k, v in self._precomputed_sd.items()
            }

        target_dev = device if device is not None else comfy.model_management.get_torch_device()
        out_sd = {}
        for k, entry in self.lora_entries.items():
            delta = materialize_single_lora_entry(entry, self.strength, dtype=dtype, device=target_dev)
            if delta is not None:
                out_sd[k] = delta.detach().cpu().contiguous()
        return out_sd

    def clone(self):
        cloned = KittenLoRAModel(
            lora_entries=self.lora_entries,
            strength=self.strength,
            precomputed_sd=dict(self._precomputed_sd) if self._precomputed_sd is not None else None,
            name=self.name,
        )
        cloned._kitten_merged = self._kitten_merged
        cloned._kitten_stage_id = self._kitten_stage_id
        cloned._kitten_source_id = self._kitten_source_id
        cloned._kitten_display_name = self._kitten_display_name
        cloned.patches = {k: list(v) for k, v in self.patches.items()}
        cloned.patches_uuid = self.patches_uuid
        return cloned

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        base_sd = self._get_default_state_dict()
        updated = dict(base_sd)
        for k, p_val in patches.items():
            if isinstance(p_val, tuple) and len(p_val) >= 2 and p_val[0] == "diff":
                diff_t = p_val[1][0]
                if k in updated and updated[k].shape == diff_t.shape:
                    updated[k] = (
                        updated[k].float() * float(strength_model) + diff_t.float() * float(strength_patch)
                    ).to(dtype=updated[k].dtype)
                else:
                    updated[k] = (diff_t.float() * float(strength_patch)).to(dtype=diff_t.dtype)
        self._precomputed_sd = updated
        self._cached_fp32_sd = updated
        self.patches_uuid = uuid.uuid4()
        return list(patches.keys())


class AnimaDiffusionLoader:
    """
    2-in-1 Diffusion / Checkpoint Loader that lists models from both `diffusion_models` (and `unet`)
    and `checkpoints` (`Stable-diffusion`), loading ONLY the diffusion model (DiT/UNet) weights
    without reading CLIP, T5, Qwen, or VAE weights into memory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        diff_files = folder_paths.get_filename_list("diffusion_models") or []
        ckpt_files = folder_paths.get_filename_list("checkpoints") or []

        combined = []
        for f in sorted(diff_files, key=natural_sort_key):
            combined.append(f"[diffusion_models] {f}")
        for f in sorted(ckpt_files, key=natural_sort_key):
            combined.append(f"[checkpoints] {f}")

        if not combined:
            combined = ["No models found in diffusion_models or checkpoints"]

        dtype_choices = ["auto (native)", "bf16", "fp16", "fp32"]
        if "fp8_e4m3fn" in DTYPE_MAP:
            dtype_choices.append("fp8_e4m3fn")
        if "fp8_e5m2" in DTYPE_MAP:
            dtype_choices.append("fp8_e5m2")

        return {
            "required": {
                "model_name": (
                    combined,
                    {
                        "tooltip": "Loads from either diffusion_models or checkpoints (Stable-diffusion). If you move a misplaced file between the two folders later, it will still be found automatically.",
                    },
                ),
                "weight_dtype": (
                    dtype_choices,
                    {
                        "default": "auto (native)",
                        "tooltip": "auto (native) keeps the file's native precision in RAM (usually bf16) while Anima Model Merger automatically upcasts to its own precision (fp32/fp64) during merge math.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "kitten_merger/loaders"

    def _resolve_path(self, model_name):
        m = re.match(r"^\[(diffusion_models|checkpoints)\]\s*(.+)$", str(model_name).strip())
        if m:
            primary_folder, rel_name = m.group(1), m.group(2)
            fallback_folder = "checkpoints" if primary_folder == "diffusion_models" else "diffusion_models"
        else:
            primary_folder, fallback_folder = "diffusion_models", "checkpoints"
            rel_name = str(model_name).strip()

        p = folder_paths.get_full_path(primary_folder, rel_name)
        if p and os.path.isfile(p):
            return p

        p_alt = folder_paths.get_full_path(fallback_folder, rel_name)
        if p_alt and os.path.isfile(p_alt):
            return p_alt

        raise FileNotFoundError(
            f"AnimaDiffusionLoader: Could not find '{rel_name}' in either 'diffusion_models' or 'checkpoints'."
        )

    def _load_diffusion_only_sd(self, ckpt_path):
        if ckpt_path.lower().endswith((".safetensors", ".sft")):
            sd = {}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                all_keys = list(f.keys())
                has_explicit_diff_prefix = any(
                    k.startswith(("model.diffusion_model.", "diffusion_model.", "net."))
                    for k in all_keys
                )

                for k in all_keys:
                    if k.startswith(NON_DIFFUSION_PREFIXES):
                        continue
                    if has_explicit_diff_prefix:
                        if not k.startswith(DIFFUSION_ROOT_PREFIXES):
                            continue
                    sd[k] = f.get_tensor(k)
            return sd, metadata

        loaded = comfy.utils.load_torch_file(ckpt_path, safe_load=True, return_metadata=True)
        if isinstance(loaded, tuple) and len(loaded) == 2:
            raw_sd, metadata = loaded
        else:
            raw_sd, metadata = loaded, None

        has_explicit_diff_prefix = any(
            k.startswith(("model.diffusion_model.", "diffusion_model.", "net."))
            for k in raw_sd.keys()
        )
        sd = {}
        for k, v in raw_sd.items():
            if k.startswith(NON_DIFFUSION_PREFIXES):
                continue
            if has_explicit_diff_prefix and not k.startswith(DIFFUSION_ROOT_PREFIXES):
                continue
            sd[k] = v
        del raw_sd
        return sd, metadata

    def load_model(self, model_name, weight_dtype="auto (native)"):
        ckpt_path = self._resolve_path(model_name)
        model_options = {}
        if weight_dtype in DTYPE_MAP:
            model_options["dtype"] = DTYPE_MAP[weight_dtype]

        sd, metadata = self._load_diffusion_only_sd(ckpt_path)

        model = None
        if hasattr(comfy.sd, "load_diffusion_model_state_dict"):
            sig = inspect.signature(comfy.sd.load_diffusion_model_state_dict)
            if "metadata" in sig.parameters:
                model = comfy.sd.load_diffusion_model_state_dict(
                    sd, model_options=model_options, metadata=metadata
                )
            else:
                model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options)

        if model is None:
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=False,
                output_clip=False,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                model_options=model_options,
            )
            model = out[0]

        model._kitten_follow_merger_precision = (weight_dtype == "auto (native)")
        model._kitten_display_name = clean_model_display_name(model_name)
        return (model,)


class AnimaLoRALoader:
    """
    Pure LoRA Loader that outputs a MODEL-compatible KittenLoRAModel (or patches an optional
    base_model if connected). When wired into Anima Model Merger, ΔW is computed directly at
    the Merger's calculation precision (fp32 / fp64 / bf16).
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_files = sorted(folder_paths.get_filename_list("loras") or [], key=natural_sort_key)
        if not lora_files:
            lora_files = ["No LoRAs found"]

        return {
            "required": {
                "lora_name": (
                    lora_files,
                    {
                        "tooltip": "Select a LoRA from models/loras. Connect the output directly to Anima Model Merger or Model Inspector as a MODEL wire.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Multiplier applied to the LoRA deltas.",
                    },
                ),
            },
            "optional": {
                "base_model": (
                    "MODEL",
                    {
                        "tooltip": "Leave disconnected to output a pure standalone LoRA for Anima Model Merger / Inspector. Connect a model (or another LoRA) to stack/apply directly.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "kitten_merger/loaders"

    def _parse_lora_file(self, lora_path):
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        entries = {}

        direct_entries = parse_direct_lora_pairs(lora_sd)
        for k, val in direct_entries.items():
            if "diff" in val or ("up" in val and "down" in val):
                entries[k] = val

        if hasattr(comfy.lora, "load_lora"):
            try:
                loaded_patches = comfy.lora.load_lora(lora_sd, ANIMA_LORA_KEY_MAP)
                for k, p_tuple in loaded_patches.items():
                    full_k = k if k.startswith("diffusion_model.") else f"diffusion_model.{strip_diffusion_prefix(k)}"
                    if full_k not in entries:
                        entries[full_k] = p_tuple
            except Exception:
                pass

        return entries, lora_sd

    def load_lora(self, lora_name, strength=1.0, base_model=None):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.isfile(lora_path):
            raise FileNotFoundError(f"AnimaLoRALoader: Could not find LoRA '{lora_name}'.")

        entries, raw_lora_sd = self._parse_lora_file(lora_path)
        clean_lora_name = clean_model_display_name(lora_name)

        if base_model is None:
            lora_model = KittenLoRAModel(
                lora_entries=entries,
                strength=strength,
                name=clean_lora_name,
            )
            return (lora_model,)

        if getattr(base_model, "_kitten_is_lora", False):
            device = comfy.model_management.get_torch_device()
            base_sd = base_model.materialize_state_dict(dtype=torch.float32, device=device)
            new_lora = KittenLoRAModel(lora_entries=entries, strength=strength, name=clean_lora_name)
            add_sd = new_lora.materialize_state_dict(dtype=torch.float32, device=device)

            combined_sd = dict(base_sd)
            for k, delta_t in add_sd.items():
                if k in combined_sd and combined_sd[k].shape == delta_t.shape:
                    combined_sd[k] = (combined_sd[k] + delta_t).contiguous()
                else:
                    combined_sd[k] = delta_t.contiguous()

            stacked = KittenLoRAModel(
                precomputed_sd=combined_sd,
                strength=1.0,
                name=f"{getattr(base_model, '_kitten_display_name', 'lora')}+{clean_lora_name}",
            )
            return (stacked,)

        key_map = {}
        if hasattr(comfy.lora, "model_lora_keys_unet"):
            key_map = comfy.lora.model_lora_keys_unet(base_model.model, key_map)
        key_map.update(ANIMA_LORA_KEY_MAP)

        loaded = comfy.lora.load_lora(raw_lora_sd, key_map)
        cloned = base_model.clone()
        cloned.add_patches(loaded, strength)
        base_disp = getattr(base_model, "_kitten_display_name", None)
        cloned._kitten_display_name = f"{base_disp}+{clean_lora_name}" if base_disp else clean_lora_name
        return (cloned,)
