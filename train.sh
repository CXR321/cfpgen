export CUDA_VISIBLE_DEVICES=5,7,6,2
# export CUDA_VISIBLE_DEVICES=5

# max_tokens=8192
# max_tokens=4096
max_tokens=2048
# max_tokens=1024
accumulate_grad_batches=2

# exp=cfpgen/cfpgen_650m_stage1
exp=cfpgen/cfpgen_650m_stage1_dplm2

# model_name=cfpgen_general_dataset_stage1
# model_name=cfpgen_general_dataset_stage1_dplm2_diff-modulation_func-cross-attn_wandb
# model_name=cfpgen_general_dataset_stage1_dplm2_diff-modulation_wandb
# model_name=cfpgen_general_dataset_stage1_dplm2_dm_ca_dc_wandb
# model_name=cfpgen_general_dataset_stage1_dplm2_dm_ca_dc_me_wandb
model_name=cfpgen_general_dataset_stage1_dplm2_dm_ca_me_drop6_wandb
# model_name=debug

python train.py \
    experiment=${exp} \
    name=${model_name} \
    datamodule.max_tokens=${max_tokens} \
    trainer.accumulate_grad_batches=${accumulate_grad_batches} 