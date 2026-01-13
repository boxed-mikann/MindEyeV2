"""
設定管理モジュール

パス設定、デバイス設定、モード設定を一元管理します。
Colab / ローカル環境を自動検出して適切なパスを設定します。
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch


# =============================================================================
# 環境検出
# =============================================================================

def is_colab() -> bool:
    """Google Colab環境かどうかを判定"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_kaggle() -> bool:
    """Kaggle環境かどうかを判定"""
    return os.path.exists("/kaggle/input")


def detect_device() -> str:
    """利用可能なデバイスを検出"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# =============================================================================
# パス設定
# =============================================================================

# このファイルの場所を基準にパスを設定
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent

# 元コードへのパス
SRC_DIR = _PROJECT_ROOT / "src"

# sys.path に追加（元コードを import 可能にする）
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# データパス設定（環境に応じて変更）
# =============================================================================

if is_colab():
    # Google Colab
    _DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/algonauts_2023_challenge_data")
    _DEFAULT_CHECKPOINT_DIR = Path("/content/drive/MyDrive/mindeye_checkpoints")
    _DEFAULT_OUTPUT_DIR = Path("/content/outputs")
elif is_kaggle():
    # Kaggle
    _DEFAULT_DATA_ROOT = Path("/kaggle/input/algonauts-2023")
    _DEFAULT_CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
    _DEFAULT_OUTPUT_DIR = Path("/kaggle/working/outputs")
else:
    # ローカル環境（研究室PCなど）
    _DEFAULT_DATA_ROOT = Path("D:/data/algonauts_2023_challenge_data")
    _DEFAULT_CHECKPOINT_DIR = _PROJECT_ROOT / "train_logs"
    _DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "outputs"


# 環境変数で上書き可能
DATA_ROOT = Path(os.environ.get("ALGONAUTS_DATA_ROOT", str(_DEFAULT_DATA_ROOT)))
CHECKPOINT_DIR = Path(os.environ.get("MINDEYE_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))
OUTPUT_DIR = Path(os.environ.get("MINDEYE_OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR)))

# 元のMindEyeV2のHuggingFaceチェックポイント
PRETRAINED_CKPT_NAME = "multisubject_subj01_1024hid_nolow_300ep"
PRETRAINED_CKPT_DIR = CHECKPOINT_DIR / PRETRAINED_CKPT_NAME


# =============================================================================
# モード設定
# =============================================================================

# ダミーモード（メモリ節約用）: 環境変数 MINDEYE_DUMMY=1 で有効化
DUMMY_MODE = os.environ.get("MINDEYE_DUMMY", "0") == "1"

# デバイス設定
DEVICE = detect_device()


# =============================================================================
# 学習設定
# =============================================================================

@dataclass
class TrainConfig:
    """学習設定"""
    # データ
    subject: str = "subj01"
    data_root: Path = field(default_factory=lambda: DATA_ROOT)
    
    # モデル
    hidden_dim: int = 1024
    seq_len: int = 1
    n_blocks: int = 4
    clip_emb_dim: int = 1664
    clip_seq_dim: int = 256
    
    # 学習
    batch_size: int = 8
    num_epochs: int = 50
    max_lr: float = 3e-4
    weight_decay: float = 1e-2
    mixup_pct: float = 0.33
    
    # 損失スケール
    clip_scale: float = 1.0
    prior_scale: float = 30.0
    blur_scale: float = 0.5
    
    # 機能フラグ
    use_prior: bool = True
    blurry_recon: bool = False
    use_image_aug: bool = False
    
    # チェックポイント
    checkpoint_dir: Path = field(default_factory=lambda: CHECKPOINT_DIR)
    pretrained_ckpt: Optional[Path] = None
    
    # モード
    dummy_mode: bool = DUMMY_MODE
    device: str = DEVICE


@dataclass
class LightTrainConfig(TrainConfig):
    """軽量学習設定（Colab T4向け）"""
    hidden_dim: int = 512
    batch_size: int = 4
    num_epochs: int = 10
    use_prior: bool = False
    blurry_recon: bool = False


@dataclass 
class DummyTrainConfig(TrainConfig):
    """ダミー学習設定（動作確認用）"""
    hidden_dim: int = 256
    batch_size: int = 2
    num_epochs: int = 1
    use_prior: bool = False
    blurry_recon: bool = False
    dummy_mode: bool = True


# =============================================================================
# 設定取得関数
# =============================================================================

def get_config(mode: str = "standard") -> TrainConfig:
    """
    学習モードに応じた設定を取得
    
    Args:
        mode: "dummy" | "light" | "standard"
    
    Returns:
        TrainConfig インスタンス
    """
    if mode == "dummy":
        return DummyTrainConfig()
    elif mode == "light":
        return LightTrainConfig()
    else:
        return TrainConfig()


def set_data_root(path: str | Path) -> None:
    """データルートを設定（グローバル変数を更新）"""
    global DATA_ROOT
    DATA_ROOT = Path(path)


def set_checkpoint_dir(path: str | Path) -> None:
    """チェックポイントディレクトリを設定"""
    global CHECKPOINT_DIR, PRETRAINED_CKPT_DIR
    CHECKPOINT_DIR = Path(path)
    PRETRAINED_CKPT_DIR = CHECKPOINT_DIR / PRETRAINED_CKPT_NAME


def set_dummy_mode(enabled: bool) -> None:
    """ダミーモードを設定"""
    global DUMMY_MODE
    DUMMY_MODE = enabled
    os.environ["MINDEYE_DUMMY"] = "1" if enabled else "0"


# =============================================================================
# 確認用
# =============================================================================

def print_config():
    """現在の設定を表示"""
    print("=" * 60)
    print("MindEyeV2 Algonauts Configuration")
    print("=" * 60)
    print(f"Environment:     {'Colab' if is_colab() else 'Kaggle' if is_kaggle() else 'Local'}")
    print(f"Device:          {DEVICE}")
    print(f"Dummy Mode:      {DUMMY_MODE}")
    print(f"Data Root:       {DATA_ROOT}")
    print(f"Checkpoint Dir:  {CHECKPOINT_DIR}")
    print(f"Output Dir:      {OUTPUT_DIR}")
    print(f"SRC Dir:         {SRC_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
