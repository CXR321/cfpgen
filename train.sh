export CUDA_VISIBLE_DEVICES=2,6,7,5
# export CUDA_VISIBLE_DEVICES=0,6,7
# export CUDA_VISIBLE_DEVICES=6,7
# export CUDA_VISIBLE_DEVICES=1

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
# model_name=cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_dc-0.05_mf_cf_wandb
# model_name=cfpgen_general_dataset_stage1_dplm2_dm_ca_dc_pow2weight_wandb
# model_name=cfpgen_general_dataset_stage1_dplm2_motifonly

# model_name=cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-sn-pn_wandb

# model_name=cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_dc-0.25-30_sn-pnwandb
# model_name=cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_dc2-0.25-30_me-sn-pnwandb

# model_name=cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_clloss-0.1_attnloss-0.1_sn-pnwandb
# model_name=cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_clloss-1.0-t0.05-cos1024_sn-pnwandb
# model_name=cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb
model_name=cfpgen_general_dataset_stage1_dplm2_goonly_alldata_struct_only

# model_name=cfpgen_general_dataset_stage1_dplm2_motifonly_structmaskNone_pfamNone
# model_name=debug


python train.py \
    experiment=${exp} \
    name=${model_name} \
    datamodule.max_tokens=${max_tokens} \
    trainer.accumulate_grad_batches=${accumulate_grad_batches} 