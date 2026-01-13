"""
CLIP ユーティリティ

OpenCLIP ViT-B-32 を用いた軽量特徴抽出のヘルパー。
Colab等の制約下では pooled 特徴 (512) を 1 トークンとして扱う運用を推奨。
"""
from typing import Tuple

import torch


def load_openclip_vitb32(device: str = "cuda"):
    """OpenCLIP ViT-B-32 をロードして前処理を返す。
    注意: インストール済みの `open_clip` パッケージが必要。
    """
    try:
        import open_clip
    except ImportError as e:
        raise ImportError(
            "open_clip が未インストールです。`pip install open_clip_torch` を実行してください"
        )
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    return model, preprocess


def openclip_image_features(
    images: torch.Tensor,
    model,
    pooled: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """OpenCLIP で画像特徴を抽出。

    Args:
        images: (B, 3, H, W) テンソル（`preprocess` 済み推奨）
        model: `open_clip` のモデル
        pooled: True の場合は pooled 出力 (B,512) を使用
        normalize: True の場合は L2 正規化

    Returns:
        (B, 1, 512) テンソル（pooled=True）
    """
    with torch.no_grad():
        feats = model.encode_image(images)
        if normalize:
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    if pooled:
        return feats.unsqueeze(1)
    else:
        # OpenCLIP の標準 API ではトークン毎出力は未提供
        # 必要であればカスタムViTから中間特徴を抽出してください。
        raise NotImplementedError("token-level 出力は未サポートです。pooled=True を使用してください")
