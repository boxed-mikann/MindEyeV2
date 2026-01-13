"""
ダミーモデル定義

Colab等のメモリ制限環境でパイプライン動作確認を行うための
軽量スタブモデルを提供します。

主要スタブ:
- DummyCLIPImageEmbedder: ViT-bigG-14 の代替（~5GB → ~10MB）
- DummyDiffusionEngine: unCLIP/SDXL の代替（~5GB → ~1MB）
- DummyVAE: AutoencoderKL の代替
- DummyConvNeXt: ConvNeXt-XL の代替
"""

import os
from contextlib import nullcontext
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def is_dummy_mode() -> bool:
    """ダミーモードが有効かどうかを確認"""
    return os.environ.get("MINDEYE_DUMMY", "0") == "1"


# =============================================================================
# DummyCLIPImageEmbedder
# =============================================================================

class DummyCLIPImageEmbedder(nn.Module):
    """
    ViT-bigG-14 CLIP Image Embedder のスタブ
    
    本物: ~2.5B params, ~5GB FP16
    ダミー: ~640K params, ~2.5MB FP16
    
    入出力形状は本物と同一:
        Input:  (B, 3, 224, 224)
        Output: (B, 256, 1664)
    """
    
    def __init__(
        self,
        clip_emb_dim: int = 1664,
        clip_seq_dim: int = 256,
        token_dim: int = 64,
    ):
        super().__init__()
        self.clip_emb_dim = clip_emb_dim
        self.clip_seq_dim = clip_seq_dim
        self.token_dim = token_dim
        
        # 軽量なパッチ埋め込み + 共有トークン射影
        # 224x224 を 14x14 パッチで分割 → 16x16=256 トークン
        self.encoder = nn.Conv2d(3, token_dim, kernel_size=14, stride=14)  # (B, token_dim, 16, 16)
        self.proj_token = nn.Linear(token_dim, clip_emb_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """小さな値で初期化"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) - 入力画像
        
        Returns:
            (B, 256, 1664) - CLIP トークン埋め込み
        """
        # パッチ埋め込み
        x = self.encoder(x)  # (B, token_dim, 16, 16)
        # トークン次元へ整形
        x = x.flatten(2).transpose(1, 2)  # (B, 256, token_dim)
        # 共有線形射影で各トークンを clip_emb_dim に拡張
        x = self.proj_token(x)  # (B, 256, clip_emb_dim)
        return x
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """CLIPの encode メソッドをエミュレート"""
        return self.forward(x)


# =============================================================================
# DummySampler
# =============================================================================

class DummySampler:
    """
    Diffusion Sampler のスタブ
    実際のサンプリングは行わず、入力をそのまま返す
    """
    
    num_steps: int = 1
    guider: Any = None
    
    def __init__(self, num_steps: int = 1):
        self.num_steps = num_steps
    
    def __call__(
        self,
        denoiser,
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uc: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """サンプリングをスキップして入力を返す"""
        return x


# =============================================================================
# DummyDiffusionEngine
# =============================================================================

class DummyDiffusionEngine(nn.Module):
    """
    unCLIP / SDXL Diffusion Engine のスタブ
    
    本物: ~2.8B params, ~5.5GB FP16
    ダミー: ~数パラメータ, ~1MB
    
    ランダム画像を生成して返す
    """
    
    def __init__(self, output_size: int = 768):
        super().__init__()
        self.output_size = output_size
        # ダミーパラメータ（モデルとして認識されるように）
        self._dummy_param = nn.Parameter(torch.zeros(1))
        
        # ダミーコンポーネント
        self.sampler = DummySampler()
        self.model = None
    
    def ema_scope(self, *args, **kwargs):
        """EMAスコープのスタブ（何もしない）"""
        return nullcontext()
    
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        """
        VAE encode のスタブ
        (B, 3, H, W) -> (B, 4, H//8, W//8)
        """
        B, C, H, W = x.shape
        return torch.randn(B, 4, H // 8, W // 8, device=x.device, dtype=x.dtype)
    
    def decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        """
        VAE decode のスタブ
        (B, 4, H, W) -> (B, 3, H*8, W*8)
        """
        B = z.shape[0]
        device, dtype = z.device, z.dtype
        # ランダム画像を返す（デバッグ時に判別しやすいパターン）
        img = torch.rand(B, 3, self.output_size, self.output_size, device=device, dtype=dtype)
        # "DUMMY" パターンを追加（中央に灰色の帯）
        img[:, :, self.output_size//2-20:self.output_size//2+20, :] = 0.5
        return img
    
    def get_first_stage_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """first stage encoding のスタブ"""
        return self.encode_first_stage(x)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass（使われないが互換性のため）"""
        return self.decode_first_stage(x)


# =============================================================================
# DummyVAE
# =============================================================================

class DummyVAE(nn.Module):
    """
    AutoencoderKL (VAE) のスタブ
    
    本物: ~83M params, ~165MB FP16
    ダミー: ~数パラメータ
    """
    
    def __init__(self, latent_dim: int = 4, scale_factor: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.scale_factor = scale_factor
        self._dummy_param = nn.Parameter(torch.zeros(1))
    
    def encode(self, x: torch.Tensor) -> "DummyVAEOutput":
        """
        Encode: (B, 3, H, W) -> latent distribution
        """
        B, C, H, W = x.shape
        latent_h, latent_w = H // self.scale_factor, W // self.scale_factor
        mean = torch.randn(B, self.latent_dim, latent_h, latent_w, device=x.device, dtype=x.dtype)
        return DummyVAEOutput(mean)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode: (B, 4, h, w) -> (B, 3, H, W)
        """
        B, C, h, w = z.shape
        H, W = h * self.scale_factor, w * self.scale_factor
        return torch.rand(B, 3, H, W, device=z.device, dtype=z.dtype)


class DummyVAEOutput:
    """VAE encode の出力をエミュレート"""
    
    def __init__(self, mean: torch.Tensor):
        self.mean = mean
        self.latent_dist = self
    
    def sample(self) -> torch.Tensor:
        return self.mean
    
    @property
    def mode(self) -> torch.Tensor:
        return self.mean


# =============================================================================
# DummyConvNeXt
# =============================================================================

class DummyConvNeXt(nn.Module):
    """
    ConvNeXt-XL のスタブ（blurry_recon の知識蒸留用）
    
    本物: ~350M params, ~700MB FP16
    ダミー: ~数パラメータ
    
    入出力:
        Input:  (B, 3, 224, 224)
        Output: (B, feature_dim)
    """
    
    def __init__(self, feature_dim: int = 1024):
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Linear(3 * 224 * 224, feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, -1)
        return self.proj(x)


# =============================================================================
# DummyCaptioner
# =============================================================================

class DummyCaptioner(nn.Module):
    """
    GIT Caption Model のスタブ
    固定のダミーキャプションを返す
    """
    
    def __init__(self):
        super().__init__()
        self._dummy_param = nn.Parameter(torch.zeros(1))
        self.dummy_caption = "[DUMMY] A test image for pipeline verification"
    
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """ダミートークンを返す"""
        return torch.tensor([[0]], device=self._dummy_param.device)
    
    def get_caption(self, *args, **kwargs) -> str:
        """ダミーキャプションを返す"""
        return self.dummy_caption


# =============================================================================
# ファクトリ関数
# =============================================================================

def create_dummy_models(device: str = "cuda") -> Dict[str, nn.Module]:
    """
    全てのダミーモデルを作成して返す
    
    Returns:
        {
            "clip_img_embedder": DummyCLIPImageEmbedder,
            "diffusion_engine": DummyDiffusionEngine,
            "vae": DummyVAE,
            "convnext": DummyConvNeXt,
            "captioner": DummyCaptioner,
        }
    """
    return {
        "clip_img_embedder": DummyCLIPImageEmbedder().to(device),
        "diffusion_engine": DummyDiffusionEngine().to(device),
        "vae": DummyVAE().to(device),
        "convnext": DummyConvNeXt().to(device),
        "captioner": DummyCaptioner().to(device),
    }


def get_dummy_clip_features(
    batch_size: int,
    clip_seq_dim: int = 256,
    clip_emb_dim: int = 1664,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    ダミーのCLIP特徴を生成（学習時のターゲット用）
    
    Returns:
        (batch_size, clip_seq_dim, clip_emb_dim)
    """
    return torch.randn(batch_size, clip_seq_dim, clip_emb_dim, device=device, dtype=dtype)


# =============================================================================
# テスト
# =============================================================================

if __name__ == "__main__":
    print("Testing dummy models...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    
    # CLIP
    dummy_clip = DummyCLIPImageEmbedder().to(device)
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    out = dummy_clip(x)
    print(f"DummyCLIP: {x.shape} -> {out.shape}")
    assert out.shape == (batch_size, 256, 1664)
    
    # Diffusion Engine
    dummy_engine = DummyDiffusionEngine().to(device)
    z = torch.randn(batch_size, 4, 96, 96, device=device)
    img = dummy_engine.decode_first_stage(z)
    print(f"DummyEngine: {z.shape} -> {img.shape}")
    assert img.shape == (batch_size, 3, 768, 768)
    
    # VAE
    dummy_vae = DummyVAE().to(device)
    x = torch.randn(batch_size, 3, 256, 256, device=device)
    encoded = dummy_vae.encode(x)
    decoded = dummy_vae.decode(encoded.sample())
    print(f"DummyVAE encode: {x.shape} -> {encoded.mean.shape}")
    print(f"DummyVAE decode: {encoded.mean.shape} -> {decoded.shape}")
    
    # ConvNeXt
    dummy_convnext = DummyConvNeXt().to(device)
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    out = dummy_convnext(x)
    print(f"DummyConvNeXt: {x.shape} -> {out.shape}")
    
    # Memory usage
    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [dummy_clip, dummy_engine, dummy_vae, dummy_convnext]
    )
    print(f"\nTotal dummy params: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB FP32)")
    print("All tests passed!")
