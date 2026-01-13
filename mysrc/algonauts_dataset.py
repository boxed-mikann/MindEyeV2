"""
Algonauts2023 データセット

Algonauts2023 Challenge のデータ形式に対応した PyTorch Dataset を提供します。

データ形式:
- fMRI: LH (左半球) + RH (右半球) の頂点データを連結
- 画像: training_images/ または test_images/ 内の PNG ファイル
- 既にz-score正規化済み、反復平均済み

参考: https://algonauts.csail.mit.edu/challenge.html
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Callable, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# =============================================================================
# 被験者ごとの頂点数
# =============================================================================

# (LH頂点数, RH頂点数) - Algonauts2023 README より
SUBJECT_DIMS: Dict[str, Tuple[int, int]] = {
    "subj01": (19004, 20544),
    "subj02": (19004, 20544),
    "subj03": (19004, 20544),
    "subj04": (19004, 20544),
    "subj05": (19004, 20544),
    "subj06": (18978, 20220),  # 欠損データあり
    "subj07": (19004, 20544),
    "subj08": (18981, 20530),  # 欠損データあり
}

# 被験者ごとの訓練/テスト画像数
SUBJECT_TRAIN_COUNTS: Dict[str, int] = {
    "subj01": 9841,
    "subj02": 9841,
    "subj03": 9082,
    "subj04": 8779,
    "subj05": 9841,
    "subj06": 9082,
    "subj07": 9841,
    "subj08": 8779,
}

SUBJECT_TEST_COUNTS: Dict[str, int] = {
    "subj01": 159,
    "subj02": 159,
    "subj03": 293,
    "subj04": 395,
    "subj05": 159,
    "subj06": 293,
    "subj07": 159,
    "subj08": 395,
}


def get_total_vertices(subject: str) -> int:
    """被験者の総頂点数（LH + RH）を取得"""
    if subject not in SUBJECT_DIMS:
        raise ValueError(f"Unknown subject: {subject}. Valid subjects: {list(SUBJECT_DIMS.keys())}")
    lh, rh = SUBJECT_DIMS[subject]
    return lh + rh


def get_subject_dims(subject: str) -> Tuple[int, int]:
    """被験者の(LH頂点数, RH頂点数)を取得"""
    if subject not in SUBJECT_DIMS:
        raise ValueError(f"Unknown subject: {subject}. Valid subjects: {list(SUBJECT_DIMS.keys())}")
    return SUBJECT_DIMS[subject]


# =============================================================================
# 画像変換
# =============================================================================

def default_image_transform(image: Image.Image) -> torch.Tensor:
    """
    デフォルトの画像変換（224x224, 正規化）
    
    torchvision がインストールされていれば使用、なければ手動実装
    """
    try:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)
    except ImportError:
        # torchvision なしの場合
        import numpy as np
        image = image.resize((224, 224), Image.BILINEAR)
        arr = np.array(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        arr = (arr - mean) / std
        return torch.from_numpy(arr)


# =============================================================================
# AlgonautsDataset
# =============================================================================

class AlgonautsDataset(Dataset):
    """
    Algonauts2023 Challenge データセット
    
    Args:
        data_root: algonauts_2023_challenge_data のルートパス
        subject: 被験者ID（"subj01" - "subj08"）
        split: "train" または "test"
        transform: 画像変換関数（None の場合はデフォルト変換）
        load_images: 画像をロードするかどうか（False の場合はパスのみ）
    
    Returns (__getitem__):
        {
            "fmri": Tensor(total_vertices,),  # LH + RH 連結済み
            "image": Tensor(3, 224, 224),     # 変換後画像（load_images=True時）
            "image_path": str,                # 画像ファイルパス
            "index": int,                     # データインデックス
            "subject": str,                   # 被験者ID
        }
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        subject: str = "subj01",
        split: str = "train",
        transform: Optional[Callable] = None,
        load_images: bool = True,
    ):
        self.data_root = Path(data_root)
        self.subject = subject
        self.split = split
        self.transform = transform if transform is not None else default_image_transform
        self.load_images = load_images
        
        # パス設定
        self.subject_dir = self.data_root / subject
        
        if split == "train":
            self.fmri_dir = self.subject_dir / "training_split" / "training_fmri"
            self.image_dir = self.subject_dir / "training_split" / "training_images"
        elif split == "test":
            self.fmri_dir = None  # テストセットにはfMRIなし
            self.image_dir = self.subject_dir / "test_split" / "test_images"
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'test'")
        
        # データ存在確認
        if not self.subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {self.subject_dir}")
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # fMRIデータをロード（訓練時のみ）
        if split == "train":
            self._load_fmri()
        else:
            self.fmri_lh = None
            self.fmri_rh = None
            self.fmri_combined = None
        
        # 画像リストを取得
        self._load_image_list()
        
        # 次元情報
        self.lh_dim, self.rh_dim = get_subject_dims(subject)
        self.total_dim = self.lh_dim + self.rh_dim
    
    def _load_fmri(self) -> None:
        """fMRIデータをメモリにロード"""
        lh_path = self.fmri_dir / "lh_training_fmri.npy"
        rh_path = self.fmri_dir / "rh_training_fmri.npy"
        
        if not lh_path.exists() or not rh_path.exists():
            raise FileNotFoundError(f"fMRI files not found in {self.fmri_dir}")
        
        print(f"Loading fMRI data for {self.subject}...")
        self.fmri_lh = np.load(lh_path).astype(np.float32)  # (N, LH_vertices)
        self.fmri_rh = np.load(rh_path).astype(np.float32)  # (N, RH_vertices)
        
        # LH + RH を連結
        self.fmri_combined = np.concatenate([self.fmri_lh, self.fmri_rh], axis=1)
        print(f"  LH shape: {self.fmri_lh.shape}, RH shape: {self.fmri_rh.shape}")
        print(f"  Combined shape: {self.fmri_combined.shape}")
    
    def _load_image_list(self) -> None:
        """画像ファイルリストを取得"""
        prefix = "train" if self.split == "train" else "test"
        
        # ファイル名でソート（train-0001, train-0002, ... の順）
        self.image_files = sorted(
            [f for f in self.image_dir.iterdir() if f.suffix == ".png"],
            key=lambda x: int(x.stem.split("_")[0].split("-")[1])
        )
        
        print(f"Found {len(self.image_files)} {self.split} images")
        
        # 画像数とfMRI数の整合性確認（訓練時のみ）
        if self.split == "train" and self.fmri_combined is not None:
            if len(self.image_files) != self.fmri_combined.shape[0]:
                raise ValueError(
                    f"Image count ({len(self.image_files)}) != fMRI count ({self.fmri_combined.shape[0]})"
                )
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        # 画像パス
        image_path = self.image_files[idx]
        
        # 結果辞書
        result = {
            "image_path": str(image_path),
            "index": idx,
            "subject": self.subject,
        }
        
        # fMRI（訓練時のみ）
        if self.fmri_combined is not None:
            result["fmri"] = torch.from_numpy(self.fmri_combined[idx])
        
        # 画像
        if self.load_images:
            image = Image.open(image_path).convert("RGB")
            result["image"] = self.transform(image)
        
        return result
    
    def get_fmri_stats(self) -> Dict[str, np.ndarray]:
        """fMRIデータの統計量を取得（デバッグ用）"""
        if self.fmri_combined is None:
            return {}
        
        return {
            "mean": self.fmri_combined.mean(axis=0),
            "std": self.fmri_combined.std(axis=0),
            "min": self.fmri_combined.min(axis=0),
            "max": self.fmri_combined.max(axis=0),
            "global_mean": self.fmri_combined.mean(),
            "global_std": self.fmri_combined.std(),
        }


# =============================================================================
# DataLoader ヘルパー
# =============================================================================

def get_dataloader(
    data_root: Union[str, Path],
    subject: str = "subj01",
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    load_images: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    DataLoader を作成するヘルパー関数
    
    Args:
        data_root: データルートパス
        subject: 被験者ID
        split: "train" or "test"
        batch_size: バッチサイズ
        shuffle: シャッフルするかどうか
        num_workers: データロードのワーカー数
        transform: 画像変換関数
        load_images: 画像をロードするかどうか
        pin_memory: pin_memory を使用するかどうか
    
    Returns:
        DataLoader インスタンス
    """
    dataset = AlgonautsDataset(
        data_root=data_root,
        subject=subject,
        split=split,
        transform=transform,
        load_images=load_images,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=split == "train",
    )


# =============================================================================
# テスト/デバッグ用
# =============================================================================

def test_dataset(data_root: Union[str, Path], subject: str = "subj01"):
    """データセットの動作確認"""
    print("=" * 60)
    print(f"Testing AlgonautsDataset for {subject}")
    print("=" * 60)
    
    # 訓練データ
    print("\n--- Training Split ---")
    train_dataset = AlgonautsDataset(data_root, subject=subject, split="train")
    print(f"Dataset length: {len(train_dataset)}")
    print(f"Total fMRI dim: {train_dataset.total_dim}")
    
    # サンプル取得
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"fMRI shape: {sample['fmri'].shape}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Image path: {sample['image_path']}")
    
    # 統計
    stats = train_dataset.get_fmri_stats()
    print(f"fMRI global mean: {stats['global_mean']:.4f}")
    print(f"fMRI global std: {stats['global_std']:.4f}")
    
    # DataLoader テスト
    print("\n--- DataLoader Test ---")
    loader = get_dataloader(data_root, subject=subject, batch_size=4)
    batch = next(iter(loader))
    print(f"Batch fMRI shape: {batch['fmri'].shape}")
    print(f"Batch image shape: {batch['image'].shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    # コマンドラインからテスト実行
    import sys
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        # デフォルトパス
        try:
            from config import DATA_ROOT
        except ImportError:
            from .config import DATA_ROOT
        data_root = DATA_ROOT
    
    test_dataset(data_root)
