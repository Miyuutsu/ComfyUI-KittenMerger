from .inspection_node import ModelMergeInspector
from .loader_nodes import AnimaDiffusionLoader, AnimaLoRALoader
from .merger_node import AnimaModelMerger

NODE_CLASS_MAPPINGS = {
    "ModelMergeInspector": ModelMergeInspector,
    "AnimaModelMerger": AnimaModelMerger,
    "AnimaDiffusionLoader": AnimaDiffusionLoader,
    "AnimaLoRALoader": AnimaLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelMergeInspector": "Model Inspector",
    "AnimaModelMerger": "Anima Model Merger",
    "AnimaDiffusionLoader": "Anima Model Loader (2-in-1)",
    "AnimaLoRALoader": "Anima LoRA Loader",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
