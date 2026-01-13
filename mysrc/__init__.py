"""
MindEyeV2 Algonauts2023 対応モジュール

このパッケージは MindEyeV2 を Algonauts2023 データ形式に対応させるための
新規コードを含みます。元の src/ コードは変更せず、必要な部分を import して再利用します。

使用例:
    from mysrc import AlgonautsDataset, AlgonautsMindEye, config
    from mysrc.transfer_utils import load_pretrained_without_ridge
"""

try:
    from config import (
        DATA_ROOT,
        SRC_DIR,
        CHECKPOINT_DIR,
        OUTPUT_DIR,
        DUMMY_MODE,
        DEVICE,
        get_config,
    )

    from algonauts_dataset import (
        AlgonautsDataset,
        SUBJECT_DIMS,
        get_dataloader,
        get_total_vertices,
    )

    from models_algonauts import (
        AlgonautsRidge,
        AlgonautsMindEye,
    )

    from transfer_utils import (
        load_pretrained_without_ridge,
        freeze_layers,
        get_trainable_params,
        count_parameters,
    )
except ImportError:
    from .config import (
        DATA_ROOT,
        SRC_DIR,
        CHECKPOINT_DIR,
        OUTPUT_DIR,
        DUMMY_MODE,
        DEVICE,
        get_config,
    )

    from .algonauts_dataset import (
        AlgonautsDataset,
        SUBJECT_DIMS,
        get_dataloader,
        get_total_vertices,
    )

    from .models_algonauts import (
        AlgonautsRidge,
        AlgonautsMindEye,
    )

    from .transfer_utils import (
        load_pretrained_without_ridge,
        freeze_layers,
        get_trainable_params,
        count_parameters,
    )

from .dummy_models import (
    DummyCLIPImageEmbedder,
    DummyDiffusionEngine,
    DummyVAE,
    DummySampler,
    DummyConvNeXt,
    is_dummy_mode,
    create_dummy_models,
)

__version__ = "0.1.0"
__all__ = [
    # config
    "DATA_ROOT", "SRC_DIR", "CHECKPOINT_DIR", "OUTPUT_DIR",
    "DUMMY_MODE", "DEVICE", "get_config",
    # dataset
    "AlgonautsDataset", "SUBJECT_DIMS", "get_dataloader", "get_total_vertices",
    # models
    "AlgonautsRidge", "AlgonautsMindEye",
    # transfer
    "load_pretrained_without_ridge", "freeze_layers", "get_trainable_params", "count_parameters",
    # dummy
    "DummyCLIPImageEmbedder", "DummyDiffusionEngine", "DummyVAE",
    "DummySampler", "DummyConvNeXt", "is_dummy_mode", "create_dummy_models",
]
