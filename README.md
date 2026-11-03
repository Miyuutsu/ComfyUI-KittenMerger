# ComfyUI-KittenMerger

An advanced, high-performance ComfyUI custom node pack built for deep model inspection, granular multi-stage merging, on-the-fly architecture conversion, and precise weight manipulation for diffusion models (specialized for Anima 2B, 2.9B, and 3.8B architectures) and LoRAs.

---

## Features

### 1. Anima Model Merger (`AnimaModelMerger`)
* **Unlimited Dynamic Inputs & Outputs:** Automatically expands model input slots (`model_1`, `model_2`...) as you connect wires, and dynamically exposes model outputs and intermediate stage outputs (`model_s1`, `model_s2`...).
* **Multi-Stage Workflow Chaining:** Chain complex sequential merges (e.g., `s1 = m1 + m2`, `s2 = s1 + m3`, `s3 = s2 - m4`) inside a single node.
* **4-Level Hierarchical DOM UI:** A scrollable, collapsible tree interface allowing you to fine-tune weights and operations (`ws`, `add`, `sub`, `add_diff`, `cosine_a`, `cosine_b`) at four distinct levels:
  * **Global Stage Level**
  * **Block / Group Level** (`blocks.0.`, `embedders.`, `llm_adapter.`, `final_layer.`)
  * **Sub-Layer Level** (`self_attn.`, `cross_attn.`, `mlp.`, etc.)
  * **Leaf Tensor Level** (`q_proj.weight`, `k_norm.weight`, etc.)
* **Inheritance & Detachment:** Child nodes inherit from their parents until manually tweaked. Detached rows highlight in orange with an **`[R]` (Reattach)** button. Supports multi-row shift/ctrl selection and delta dragging.
* **1-Click "Fix Zeros":** Automatically detects unpopulated or dead zero layers in source models and maps them correctly during stage evaluations.
* **Legacy Import/Export:** Import/export stage JSON recipes, full multi-stage project recipes, or automatically parse legacy workflow JSONs.

### 2. Model Inspector (`ModelMergeInspector`)
* **Automatic Lineage & Verification:** Wire the `merge_schema` output from the Merger directly into the Inspector to automatically track per-key contributions and verify layer arithmetic.
* **Manual Compare Schema:** Supports flexible custom math expressions (e.g., `m1 + m2 = m3`, `0.5*m1 + 0.5*m2 = m3`) in both `lineage` and `verify` modes.
* **Solve Zeroes (`none` / `blocks` / `keys`):** Automatically repair corrupted or zeroed-out blocks and keys across connected models using `fallback` or `smart_average` donor strategies.

### 3. Specialized Loaders (`AnimaDiffusionLoader` & `AnimaLoRALoader`)
* **Anima Diffusion Loader (2-in-1):** Lists files from both `diffusion_models` and `checkpoints` (`Stable-diffusion`), loading *only* the DiT/UNet weights into memory while stripping heavy VAE, text encoders, and conditioner weights.
* **Anima LoRA Loader:** Loads standalone LoRAs as `MODEL`-compatible wrappers (`KittenLoRAModel`), materializing $\Delta W$ deltas directly at the merger's calculation precision (`fp32`, `fp64`, or `bf16`).

---

## Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/Miyuutsu/ComfyUI-KittenMerger
   ```
3. Restart ComfyUI.

---

## Project Structure

```text
ComfyUI-KittenMerger/
├── __init__.py         # Node class and display name mappings
├── utils.py            # Shared math, block parsers, manifests, and scaling
├── loader_nodes.py     # AnimaDiffusionLoader and AnimaLoRALoader
├── merger_node.py      # AnimaModelMerger backend execution logic
├── inspection_node.py  # ModelMergeInspector diagnostics and verification backend
└── web/
    ├── inspector.js    # Inspector dynamic slot sync and display UI
    └── merger.js       # Merger embedded 3-level tree UI and stage tabs
```

---

## License

Distributed under GNU General Public License v3.0. See `LICENSE` for more information.