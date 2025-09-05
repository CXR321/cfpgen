from byprot.models.lm.cfp_gen import CondDiffusionProteinLanguageModel, CondDiffusionProteinLanguageModel2
import matplotlib.pyplot as plt
import numpy as np

# 定义要比较的模型路径和标签
model_configs = [
    {
        'path': "byprot-checkpoints-temp/cfpgen_general_dataset_stage1_dplm2_dm_ca_me_wandb/checkpoints/step_49999.0-loss_0.69.ckpt",
        'label': 'Model 1',
        'has_motif': True
    },
    # {
    #     'path': "byprot-checkpoints/cfpgen_general_dataset_stage1_dplm2_dm_ca_dc_me_wandb/checkpoints/step_149999.0-loss_0.65.ckpt",
    #     'label': 'Model 2',
    #     'has_motif': True
    # },
    {
        'path': "byprot-checkpoints-temp/cfpgen_general_dataset_stage1_dplm2_dm_ca_me_wandb/checkpoints/step_89999.0-loss_0.68.ckpt",
        'label': 'Model 2',
        'has_motif': True
    },
    {
        'path': "byprot-checkpoints/cfpgen_general_dataset_stage1_dplm2_diff-modulation_func-cross-attn_wandb/checkpoints/step_29999.0-loss_0.71.ckpt",
        'label': 'Baseline',
        'has_motif': False  # Baseline 没有 motif scale
    }
]

# 存储所有模型的 scale 数据
all_scale_motif_data = []
all_scale_func_data = []

# 加载每个模型并提取 scale 参数
for config in model_configs:
    print(f"Loading model: {config['label']}")
    model = CondDiffusionProteinLanguageModel2.from_pretrained(config['path'])
    model = model.eval()
    
    # 提取 func scale 参数
    scale_func_list = [layer.cross_res_scale.detach().numpy() for layer in model.net.esm.encoder.layer]
    all_scale_func_data.append((scale_func_list, config['label'] + ' func'))
    
    # 只有有 motif scale 的模型才提取
    if config['has_motif']:
        scale_motif_list = [layer.motif_cross_res_scale.detach().numpy() for layer in model.net.esm.encoder.layer]
        all_scale_motif_data.append((scale_motif_list, config['label'] + ' motif'))

# 创建可视化
plt.figure(figsize=(12, 8))

# 绘制 motif scale 数据（只有有 motif 的模型）
for i, (data, label) in enumerate(all_scale_motif_data):
    plt.plot(data, marker='o', markersize=4, label=label, linestyle='-', linewidth=2)

# 绘制 func scale 数据（所有模型）
for i, (data, label) in enumerate(all_scale_func_data):
    plt.plot(data, marker='s', markersize=4, label=label, linestyle='--', linewidth=2)

plt.xlabel('Layer Index')
plt.ylabel('Scale Value')
plt.title('Comparison of Cross-Residual Scale Parameters Across Models')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图像
plt.savefig('cross_res_scale_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 可选：打印统计信息
print("\nScale Value Statistics:")
print("=" * 50)
for i, config in enumerate(model_configs):
    func_mean = np.mean(all_scale_func_data[i][0])
    print(f"{config['label']}:")
    print(f"  Func scale mean: {func_mean:.4f}")
    
    # 只有有 motif scale 的模型才显示
    if config['has_motif']:
        motif_mean = np.mean(all_scale_motif_data[i][0])
        print(f"  Motif scale mean: {motif_mean:.4f}")
    print()