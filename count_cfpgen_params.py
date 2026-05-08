import argparse
from pathlib import Path
import sys
from typing import Any


def count_params(module: Any) -> int:
    return sum(p.numel() for p in module.parameters())


def human_int(n: int) -> str:
    return f"{n:,}"


def ratio_str(part: int, whole: int) -> str:
    if whole == 0:
        return "N/A"
    return f"{part / whole * 100:.4f}%"


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError as e:
        raise SystemExit(
            "缺少依赖 torch，无法加载 .ckpt 并统计参数量。请先安装 PyTorch，例如：\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "或在你已有 torch 的训练/推理环境中运行本脚本。"
        ) from e

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/AIRvePFS/dair/chenxr-data/repo/cfpgen/byprot-checkpoints/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb/checkpoints/step_109999.0-loss_0.89.ckpt",
    )
    parser.add_argument("--fast", action="store_true", help="仅基于 checkpoint state_dict 统计，不加载 byprot 模型")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # 1. 直接读取原始权重文件
    raw_ckpt = torch.load(args.ckpt, map_location="cpu")

    # 兼容不同的保存格式（有时候权重藏在 'state_dict' 或 'model' 键值下）
    if "state_dict" in raw_ckpt:
        ckpt_state_dict = raw_ckpt["state_dict"]
    elif "model" in raw_ckpt:
        ckpt_state_dict = raw_ckpt["model"]
    else:
        ckpt_state_dict = raw_ckpt

    # 2. 查找是否有 motif_head 相关的参数
    motif_keys_in_ckpt = [k for k in ckpt_state_dict.keys() if "motif_head" in k]

    if len(motif_keys_in_ckpt) > 0:
        print(f"✅ Checkpoint 文件中包含 'motif_head' 的权重，共 {len(motif_keys_in_ckpt)} 个张量。")
        # 打印前两个 key 看看名字长什么样
        print("例如:", motif_keys_in_ckpt[:2])
    else:
        print("❌ 警告：Checkpoint 文件中根本没有 'motif_head' 相关的参数！")

    ckpt_path = args.ckpt
    total_params_model = sum(v.numel() for v in ckpt_state_dict.values() if torch.is_tensor(v))
    net_prefix = "model.net."
    total_params_net = sum(
        v.numel()
        for k, v in ckpt_state_dict.items()
        if k.startswith(net_prefix) and torch.is_tensor(v)
    )

    model_cfg = None
    base_ckpt = None
    if isinstance(raw_ckpt, dict) and "hyper_parameters" in raw_ckpt:
        hp = raw_ckpt["hyper_parameters"]
        if isinstance(hp, dict) and "model" in hp:
            model_cfg = hp["model"]
            try:
                base_ckpt = model_cfg["net"]["pretrained_model_name_or_path"]
            except Exception:
                base_ckpt = None

    adaln_params = 0
    func_cross_attn_params = 0
    go_guided_params = 0
    fsr_params = 0

    for k, v in ckpt_state_dict.items():
        if not (k.startswith(net_prefix) and torch.is_tensor(v)):
            continue
        if ".adaLN_modulation." in k:
            adaln_params += v.numel()
            go_guided_params += v.numel()
            continue

        if (
            ".func_proj." in k
            or ".cross_attn_ln." in k
            or ".cross_attn." in k
            or k.endswith(".cross_res_scale")
        ):
            func_cross_attn_params += v.numel()
            go_guided_params += v.numel()
            continue

        if (
            ".motif_proj." in k
            or ".motif_cross_attn_ln." in k
            or ".motif_cross_attn." in k
            or k.endswith(".motif_cross_res_scale")
        ):
            fsr_params += v.numel()
            continue

    print(f"ckpt: {ckpt_path}")
    print(f"total params (model): {human_int(total_params_model)}")
    print(f"total params (model.net): {human_int(total_params_net)}")
    if base_ckpt is not None:
        print(f"base ckpt (from hparams): {base_ckpt}")
    else:
        print("base ckpt: N/A (not found in checkpoint hparams)")

    print(f"adaLN_modulation params: {human_int(adaln_params)} ({ratio_str(adaln_params, total_params_net)} of model.net)")
    print(
        f"func cross-attn params: {human_int(func_cross_attn_params)} "
        f"({ratio_str(func_cross_attn_params, total_params_net)} of model.net)"
    )
    print(f"GO-guided params (adaLN_modulation + func cross-attn): {human_int(go_guided_params)}")
    print(f"FSR params (motif cross-attn): {human_int(fsr_params)}")

    hidden_size = None
    num_layers = None
    use_diff_modulation = None
    use_func_cross_attn = None
    if model_cfg is not None:
        try:
            use_diff_modulation = bool(model_cfg["use_diff_modulation"])
        except Exception:
            pass
        try:
            use_func_cross_attn = bool(model_cfg["use_func_cross_attn"])
        except Exception:
            pass
        try:
            from transformers import AutoConfig
            net_name = model_cfg["net"]["name"]
            hf_cfg = AutoConfig.from_pretrained(net_name)
            hidden_size = getattr(hf_cfg, "hidden_size", None)
            num_layers = getattr(hf_cfg, "num_hidden_layers", None)
        except Exception:
            pass
    if hidden_size is not None and num_layers is not None:
        adaln_out = 12 * hidden_size if use_diff_modulation else 6 * hidden_size
        adaln_per_layer = hidden_size * adaln_out + adaln_out
        expected_adaln = adaln_per_layer * num_layers
        print(
            f"[theory] hidden_size={hidden_size}, layers={num_layers}, "
            f"use_diff_modulation={use_diff_modulation}, use_func_cross_attn={use_func_cross_attn}"
        )
        print(f"[theory] adaLN per-layer params: {human_int(adaln_per_layer)}")
        print(f"[theory] adaLN total params: {human_int(expected_adaln)}")


if __name__ == "__main__":
    main()
