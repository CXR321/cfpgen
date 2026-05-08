#!/bin/bash

# fold and then compute plddt and tm-score

fasta_path="xxxx.fasta"  # generated sequences based on backbone
gt_pkl_path="data-bin/uniprotKB/cfpgen_general_dataset/test_bb.pkl"
max_tokens=1024

echo 'Folding by ESMFold'
source ~/anaconda3/etc/profile.d/conda.sh
conda activate esm  # esmfold env

python analysis/cal_plddt_tmscore.py -i ${fasta_path} --max-tokens-per-batch ${max_tokens} -gt ${gt_pkl_path}

