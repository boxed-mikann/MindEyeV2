"""
Algonauts2023 対応モデル

元の src/models.py の BrainNetwork 等を再利用しつつ、
Algonauts2023 のデータ形式（LH+RH 連結頂点）に対応したモデルを提供します。

アーキテクチャ:
    Algonauts fMRI [39548] → AlgonautsRidge → [hidden_dim] → BrainNetwork → CLIP tokens

転移学習:
    - BrainNetwork: 既存ckptから完全に再利用可能
    - Ridge: 入力次元が異なるため新規初期化が必要
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union

import torch
import torch.nn as nn

# 元コードへのパスを追加
_SRC_DIR = Path(__file__).parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ローカルモジュール
try:
    from algonauts_dataset import SUBJECT_DIMS, get_total_vertices
except ImportError:
    from .algonauts_dataset import SUBJECT_DIMS, get_total_vertices


# =============================================================================
# AlgonautsRidge: Algonauts用Ridge層
# =============================================================================

class AlgonautsRidge(nn.Module):
    """
    Algonauts2023 用の Ridge 回帰層
    
    各被験者の頂点数に応じた線形変換を行う。
    元の MindEyeV2 の RidgeRegression と互換性のあるインターフェース。
    
    Args:
        subjects: 対応する被験者IDのリスト（例: ["subj01"]）
        out_features: 出力次元（hidden_dim と同じ）
    """
    
    def __init__(
        self,
        subjects: List[str] = ["subj01"],
        out_features: int = 1024,
    ):
        super().__init__()
        self.subjects = subjects
        self.out_features = out_features
        
        # 被験者ごとの入力次元を取得
        self.input_sizes = [get_total_vertices(subj) for subj in subjects]
        
        # 被験者ごとの線形層
        self.linears = nn.ModuleList([
            nn.Linear(in_size, out_features)
            for in_size in self.input_sizes
        ])
        
        # 被験者IDから線形層インデックスへのマッピング
        self.subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
    
    def forward(
        self,
        x: torch.Tensor,
        subject_idx: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, num_vertices) または (batch, num_vertices)
            subject_idx: 被験者インデックス
        
        Returns:
            (batch, 1, out_features)
        """
        # (batch, num_vertices) の場合は (batch, 1, num_vertices) に変換
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # (batch, 1, num_vertices) -> (batch, num_vertices)
        x = x.squeeze(1)
        
        # 線形変換
        x = self.linears[subject_idx](x)
        
        # (batch, out_features) -> (batch, 1, out_features)
        return x.unsqueeze(1)
    
    def get_subject_idx(self, subject: str) -> int:
        """被験者IDからインデックスを取得"""
        return self.subject_to_idx.get(subject, 0)


# =============================================================================
# AlgonautsMindEye: Algonauts対応の完全モデル
# =============================================================================

class AlgonautsMindEye(nn.Module):
    """
    Algonauts2023 対応 MindEye モデル
    
    元の BrainNetwork を内包し、Algonauts用の新しいRidge層と組み合わせる。
    
    アーキテクチャ:
        fMRI [D_algonauts] → Ridge → [hidden_dim] → BrainNetwork → CLIP tokens
    
    Args:
        subjects: 対応する被験者IDのリスト
        hidden_dim: 隠れ層の次元
        out_dim: BrainNetwork出力次元（clip_emb_dim * clip_seq_dim）
        seq_len: シーケンス長
        n_blocks: Mixerブロック数
        clip_emb_dim: CLIPトークンの埋め込み次元
        clip_seq_dim: CLIPトークンのシーケンス長
        use_prior: Diffusion Priorを使用するか
        blurry_recon: ぼやけ再構成を使用するか
        clip_scale: CLIPスケール
        drop: ドロップアウト率
    """
    
    def __init__(
        self,
        subjects: List[str] = ["subj01"],
        hidden_dim: int = 1024,
        out_dim: Optional[int] = None,
        seq_len: int = 1,
        n_blocks: int = 4,
        clip_emb_dim: int = 1664,
        clip_seq_dim: int = 256,
        use_prior: bool = True,
        blurry_recon: bool = False,
        clip_scale: float = 1.0,
        drop: float = 0.15,
    ):
        super().__init__()
        
        # 設定を保存
        self.subjects = subjects
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.clip_emb_dim = clip_emb_dim
        self.clip_seq_dim = clip_seq_dim
        self.use_prior = use_prior
        self.blurry_recon = blurry_recon
        
        # 出力次元（BrainNetworkのout_dim）
        if out_dim is None:
            out_dim = clip_emb_dim * clip_seq_dim  # 1664 * 256 = 425984
        self.out_dim = out_dim
        
        # Algonauts用Ridge層（新規）
        self.ridge = AlgonautsRidge(
            subjects=subjects,
            out_features=hidden_dim,
        )
        
        # BrainNetwork（元コードから import）
        # 注意: in_dim は hidden_dim と同じ値を使用
        try:
            from models import BrainNetwork
            self.backbone = BrainNetwork(
                h=hidden_dim,
                in_dim=hidden_dim,  # Ridge出力と同じ
                out_dim=out_dim,
                seq_len=seq_len,
                n_blocks=n_blocks,
                drop=drop,
                clip_size=clip_emb_dim,
                blurry_recon=blurry_recon,
                clip_scale=clip_scale,
            )
        except ImportError as e:
            raise ImportError(
                f"Could not import BrainNetwork from src/models.py. "
                f"Make sure src/ is in your Python path. Error: {e}"
            )
        
        # Diffusion Prior（オプション）
        self.diffusion_prior = None
        if use_prior:
            self._init_prior()
    
    def _init_prior(self):
        """Diffusion Priorを初期化"""
        try:
            from models import BrainDiffusionPrior
            from dalle2_pytorch.train_configs import DiffusionPriorNetworkConfig
            
            # デフォルトのPrior設定
            prior_config = DiffusionPriorNetworkConfig(
                dim=self.clip_emb_dim,
                depth=6,
                dim_head=64,
                heads=self.clip_emb_dim #64,
                #causal=False,
                #num_tokens=self.clip_seq_dim,
                #learned_query_mode="none",
            )
            
            self.diffusion_prior = BrainDiffusionPrior(
                net=prior_config.create(),
                image_embed_dim=self.clip_emb_dim,
                condition_on_text_encodings=False,
                timesteps=100,
                cond_drop_prob=0.2,
                image_embed_scale=None,
            )
        except ImportError as e:
            print(f"Warning: Could not initialize DiffusionPrior: {e}")
            self.diffusion_prior = None
    
    def forward(
        self,
        fmri: torch.Tensor,
        subject_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            fmri: (batch, num_vertices) または (batch, 1, num_vertices)
            subject_idx: 被験者インデックス
        
        Returns:
            backbone: (batch, seq_len, clip_emb_dim) - BrainNetwork出力
            clip_voxels: (batch, seq_len, clip_emb_dim) - CLIP投影
            blurry: ぼやけ再構成（有効時）または空テンソル
        """
        # Ridge: fMRI → hidden_dim
        x = self.ridge(fmri, subject_idx=subject_idx)
        
        # BrainNetwork: hidden_dim → CLIP tokens
        backbone, clip_voxels, blurry = self.backbone(x)
        
        return backbone, clip_voxels, blurry
    
    def encode_fmri(
        self,
        fmri: torch.Tensor,
        subject_idx: int = 0,
    ) -> torch.Tensor:
        """
        fMRIをCLIPトークンにエンコード（推論用簡易メソッド）
        
        Returns:
            (batch, clip_seq_dim, clip_emb_dim)
        """
        backbone, clip_voxels, _ = self.forward(fmri, subject_idx)
        return clip_voxels
    
    def get_ridge_params(self):
        """Ridge層のパラメータを取得（転移学習時の最適化対象）"""
        return self.ridge.parameters()
    
    def get_backbone_params(self):
        """BrainNetworkのパラメータを取得"""
        return self.backbone.parameters()
    
    def get_all_params(self):
        """全パラメータを取得"""
        return self.parameters()


# =============================================================================
# モデル作成ヘルパー
# =============================================================================

def create_algonauts_model(
    subjects: List[str] = ["subj01"],
    hidden_dim: int = 1024,
    seq_len: int = 1,
    n_blocks: int = 4,
    clip_emb_dim: int = 1664,
    clip_seq_dim: int = 256,
    use_prior: bool = False,
    blurry_recon: bool = False,
    device: str = "cuda",
) -> AlgonautsMindEye:
    """
    Algonauts対応モデルを作成
    
    Args:
        subjects: 被験者IDリスト
        hidden_dim: 隠れ層次元
        seq_len: シーケンス長
        n_blocks: Mixerブロック数
        clip_emb_dim: CLIP埋め込み次元
        clip_seq_dim: CLIPシーケンス長
        use_prior: Diffusion Priorを使用するか
        blurry_recon: ぼやけ再構成を使用するか
        device: デバイス
    
    Returns:
        AlgonautsMindEye インスタンス
    """
    model = AlgonautsMindEye(
        subjects=subjects,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        n_blocks=n_blocks,
        clip_emb_dim=clip_emb_dim,
        clip_seq_dim=clip_seq_dim,
        use_prior=use_prior,
        blurry_recon=blurry_recon,
    )
    
    return model.to(device)


# =============================================================================
# テスト
# =============================================================================

if __name__ == "__main__":
    print("Testing AlgonautsMindEye...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # subj01 の次元
    try:
        from algonauts_dataset import get_total_vertices
    except ImportError:
        from .algonauts_dataset import get_total_vertices
    
    num_vertices = get_total_vertices("subj01")
    print(f"subj01 total vertices: {num_vertices}")
    
    # モデル作成（Priorなし、軽量設定）
    model = create_algonauts_model(
        subjects=["subj01"],
        hidden_dim=256,  # 軽量テスト用
        use_prior=False,
        blurry_recon=False,
        device=device,
    )
    
    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    ridge_params = sum(p.numel() for p in model.ridge.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    
    print(f"Total params: {total_params:,}")
    print(f"Ridge params: {ridge_params:,}")
    print(f"Backbone params: {backbone_params:,}")
    
    # Forward テスト
    batch_size = 2
    fmri = torch.randn(batch_size, num_vertices, device=device)
    
    backbone, clip_voxels, blurry = model(fmri)
    
    print(f"\nInput fMRI shape: {fmri.shape}")
    print(f"Backbone output shape: {backbone.shape}")
    print(f"CLIP voxels shape: {clip_voxels.shape}")
    
    print("\nAll tests passed!")
