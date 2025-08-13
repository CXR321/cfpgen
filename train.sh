# export CUDA_VISIBLE_DEVICES=2,7,5,6
export CUDA_VISIBLE_DEVICES=7

# max_tokens=8192
# max_tokens=4096
max_tokens=1024
accumulate_grad_batches=1

# exp=cfpgen/cfpgen_650m_stage1
exp=cfpgen/cfpgen_650m_stage1_dplm2

# model_name=cfpgen_general_dataset_stage1
model_name=cfpgen_general_dataset_stage1_dplm2_debug_wandb

python train.py \
    experiment=${exp} \
    name=${model_name} \
    datamodule.max_tokens=${max_tokens} \
    trainer.accumulate_grad_batches=${accumulate_grad_batches} 