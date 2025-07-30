export CUDA_VISIBLE_DEVICES=0,2,5,6

max_tokens=8192
accumulate_grad_batches=32

exp=cfpgen/cfpgen_650m_stage1
model_name=cfpgen_general_dataset_stage1

python train.py \
    experiment=${exp} \
    name=${model_name} \
    datamodule.max_tokens=${max_tokens} \
    trainer.accumulate_grad_batches=${accumulate_grad_batches} 