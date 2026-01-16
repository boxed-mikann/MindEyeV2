"""
転移学習ユーティリティ

既存の MindEyeV2 チェックポイントから重みをロードし、
Algonauts対応モデルに転移学習を行うためのユーティリティを提供します。

転移戦略:
- Ridge層: 入力次元が異なるため新規初期化（転移不可）
- BrainNetwork: 完全に再利用可能
- Diffusion Prior: 完全に再利用可能
"""

import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import os


# =============================================================================
# チェックポイント操作
# =============================================================================

def load_checkpoint(ckpt_path):
    """チェックポイントをロード（ディレクトリまたはファイル）"""
    if os.path.isdir(ckpt_path):
        # ディレクトリの場合は中のファイルを探す
        candidates = ["last.pth", "model.pth", "checkpoint.pth", "best.pth"]
        for name in candidates:
            path = os.path.join(ckpt_path, name)
            if os.path.exists(path):
                ckpt_path = path
                break
        else:
            # .pthファイルを探す
            pth_files = [f for f in os.listdir(ckpt_path) if f.endswith(".pth")]
            if pth_files:
                ckpt_path = os.path.join(ckpt_path, pth_files[0])
            else:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    # PyTorch 2.6+ requires weights_only=False for checkpoints with custom objects
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    return checkpoint


def get_state_dict_from_checkpoint(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    """
    チェックポイントからstate_dictを抽出
    
    異なる形式のチェックポイントに対応:
    - {"model_state_dict": {...}} 形式
    - {"state_dict": {...}} 形式
    - 直接 state_dict 形式
    """
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and any(k.startswith(("backbone", "ridge", "diffusion")) for k in checkpoint.keys()):
        return checkpoint
    else:
        raise ValueError("Could not find state_dict in checkpoint")


def filter_state_dict(
    state_dict: Dict[str, torch.Tensor],
    exclude_patterns: List[str] = None,
    include_patterns: List[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    state_dict をフィルタリング
    
    Args:
        state_dict: 元のstate_dict
        exclude_patterns: 除外するキーのパターン（正規表現）
        include_patterns: 含めるキーのパターン（正規表現、Noneの場合は全て）
    
    Returns:
        フィルタリング後のstate_dict
    """
    filtered = {}
    
    for key, value in state_dict.items():
        # 除外パターンチェック
        if exclude_patterns:
            if any(re.search(pat, key) for pat in exclude_patterns):
                continue
        
        # 含めるパターンチェック
        if include_patterns:
            if not any(re.search(pat, key) for pat in include_patterns):
                continue
        
        filtered[key] = value
    
    return filtered


# =============================================================================
# 転移学習
# =============================================================================

def load_pretrained_without_ridge(
    model: nn.Module,
    ckpt_path: Union[str, Path],
    freeze_backbone: bool = True,
    freeze_prior: bool = True,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    既存チェックポイントからRidge層以外をロード
    
    Args:
        model: AlgonautsMindEye モデル
        ckpt_path: 既存ckptのパス
        freeze_backbone: BrainNetworkをfreezeするか
        freeze_prior: Diffusion Priorをfreezeするか
        strict: strict loadingを使用するか
        verbose: ログを出力するか
    
    Returns:
        (loaded_keys, missing_keys): ロードしたキーと見つからなかったキー
    """
    # チェックポイントをロード---
    checkpoint = load_checkpoint(ckpt_path)
    state_dict = get_state_dict_from_checkpoint(checkpoint)
    
    # Ridge層を除外
    state_dict = filter_state_dict(
        state_dict,
        exclude_patterns=[r"^ridge\.", r"^linears\."]
    )
    
    if verbose:
        print(f"Loaded {len(state_dict)} keys (excluding ridge)")
    
    # モデルにロード
    result = model.load_state_dict(state_dict, strict=False)
    
    loaded_keys = [k for k in state_dict.keys() if k not in result.missing_keys]
    
    if verbose:
        print(f"Successfully loaded: {len(loaded_keys)} keys")
        if result.missing_keys:
            # Ridge関連以外のmissing keysを表示
            non_ridge_missing = [k for k in result.missing_keys if not k.startswith("ridge")]
            if non_ridge_missing:
                print(f"Missing (non-ridge): {non_ridge_missing}")
        if result.unexpected_keys:
            print(f"Unexpected keys: {result.unexpected_keys}")
    
    # Freeze
    if freeze_backbone:
        freeze_layers(model, ["backbone"])
        if verbose:
            print("Froze backbone layers")
    
    if freeze_prior and hasattr(model, "diffusion_prior") and model.diffusion_prior is not None:
        freeze_layers(model, ["diffusion_prior"])
        if verbose:
            print("Froze diffusion_prior layers")
    
    return loaded_keys, result.missing_keys


def load_backbone_only(
    model: nn.Module,
    ckpt_path: Union[str, Path],
    freeze: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    BrainNetwork（backbone）のみをロード
    
    Args:
        model: AlgonautsMindEye モデル
        ckpt_path: 既存ckptのパス
        freeze: ロード後にfreezeするか
        verbose: ログを出力するか
    
    Returns:
        ロードしたキーのリスト
    """
    checkpoint = load_checkpoint(ckpt_path)
    state_dict = get_state_dict_from_checkpoint(checkpoint)
    
    # backbone関連のみ抽出
    state_dict = filter_state_dict(
        state_dict,
        include_patterns=[r"^backbone\."]
    )
    
    if verbose:
        print(f"Found {len(state_dict)} backbone keys")
    
    # ロード
    result = model.load_state_dict(state_dict, strict=False)
    loaded_keys = [k for k in state_dict.keys() if k not in result.missing_keys]
    
    if verbose:
        print(f"Loaded {len(loaded_keys)} backbone keys")
    
    # Freeze
    if freeze:
        freeze_layers(model, ["backbone"])
        if verbose:
            print("Froze backbone layers")
    
    return loaded_keys


# =============================================================================
# パラメータ操作
# =============================================================================

def freeze_layers(
    model: nn.Module,
    layer_names: List[str],
) -> None:
    """
    指定したレイヤーをfreeze（requires_grad=False）
    
    Args:
        model: モデル
        layer_names: freezeするレイヤー名のリスト
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if name.startswith(layer_name):
                param.requires_grad = False
                break


def unfreeze_layers(
    model: nn.Module,
    layer_names: List[str],
) -> None:
    """
    指定したレイヤーをunfreeze（requires_grad=True）
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if name.startswith(layer_name):
                param.requires_grad = True
                break


def get_trainable_params(
    model: nn.Module,
    mode: str = "ridge_only",
) -> List[nn.Parameter]:
    """
    学習対象パラメータを取得
    
    Args:
        model: モデル
        mode: 
            "ridge_only": Ridge層のみ（転移学習時推奨）
            "ridge_and_proj": Ridge + 最終投影層
            "all_unfrozen": freezeされていない全パラメータ
            "all": 全パラメータ
    
    Returns:
        パラメータのリスト
    """
    if mode == "ridge_only":
        return [p for n, p in model.named_parameters() if "ridge" in n]
    
    elif mode == "ridge_and_proj":
        patterns = ["ridge", "backbone_linear", "clip_proj"]
        return [
            p for n, p in model.named_parameters()
            if any(pat in n for pat in patterns)
        ]
    
    elif mode == "all_unfrozen":
        return [p for p in model.parameters() if p.requires_grad]
    
    elif mode == "all":
        return list(model.parameters())
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> Dict[str, int]:
    """
    パラメータ数をカウント
    
    Returns:
        {
            "total": 総パラメータ数,
            "trainable": 学習可能パラメータ数,
            "frozen": freeze済みパラメータ数,
            "by_layer": レイヤー別パラメータ数,
        }
    """
    total = 0
    trainable = 0
    by_layer = {}
    
    for name, param in model.named_parameters():
        n_params = param.numel()
        total += n_params
        
        if param.requires_grad:
            trainable += n_params
        
        # レイヤー名（最初のドットまで）でグループ化
        layer_name = name.split(".")[0]
        if layer_name not in by_layer:
            by_layer[layer_name] = {"total": 0, "trainable": 0}
        by_layer[layer_name]["total"] += n_params
        if param.requires_grad:
            by_layer[layer_name]["trainable"] += n_params
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "by_layer": by_layer,
    }


def print_parameter_summary(model: nn.Module) -> None:
    """パラメータサマリーを表示"""
    stats = count_parameters(model)
    
    print("=" * 60)
    print("Parameter Summary")
    print("=" * 60)
    print(f"Total:     {stats['total']:>12,}")
    print(f"Trainable: {stats['trainable']:>12,}")
    print(f"Frozen:    {stats['frozen']:>12,}")
    print("-" * 60)
    print("By Layer:")
    for layer_name, layer_stats in stats["by_layer"].items():
        status = "🟢" if layer_stats["trainable"] > 0 else "🔒"
        print(f"  {status} {layer_name:20} {layer_stats['total']:>10,} ({layer_stats['trainable']:,} trainable)")
    print("=" * 60)


# =============================================================================
# チェックポイント保存
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    save_path: Union[str, Path],
    extra_info: Optional[Dict] = None,
) -> None:
    """
    チェックポイントを保存
    
    Args:
        model: モデル
        optimizer: オプティマイザ（Noneの場合は保存しない）
        epoch: エポック番号
        save_path: 保存先パス
        extra_info: 追加情報
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if extra_info:
        checkpoint.update(extra_info)
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to: {save_path}")


# =============================================================================
# テスト
# =============================================================================

if __name__ == "__main__":
    print("Testing transfer_utils...")
    
    # ダミーモデルでテスト
    try:
        from models_algonauts import AlgonautsMindEye
    except ImportError:
        from .models_algonauts import AlgonautsMindEye
    
    model = AlgonautsMindEye(
        subjects=["subj01"],
        hidden_dim=256,
        use_prior=False,
        blurry_recon=False,
    )
    
    # パラメータサマリー
    print("\n--- Initial State ---")
    print_parameter_summary(model)
    
    # Freeze テスト
    freeze_layers(model, ["backbone"])
    print("\n--- After Freezing Backbone ---")
    print_parameter_summary(model)
    
    # Trainable params 取得
    ridge_params = get_trainable_params(model, mode="ridge_only")
    print(f"\nRidge-only trainable params: {sum(p.numel() for p in ridge_params):,}")
    
    all_unfrozen = get_trainable_params(model, mode="all_unfrozen")
    print(f"All unfrozen params: {sum(p.numel() for p in all_unfrozen):,}")
    
    print("\nAll tests passed!")
