"""Feature extraction, normalization, and preprocessing orchestration."""

from .features import AefPcaModel, FeaturePack, build_feature_names, build_multimodal_feature_pack
from .normalize import normalize_array
from .pipeline import run_preprocessing

__all__ = [
    "AefPcaModel",
    "FeaturePack",
    "build_feature_names",
    "build_multimodal_feature_pack",
    "normalize_array",
    "run_preprocessing",
]
