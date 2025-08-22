import yaml
import argparse
from pprint import pprint
import torch
import os
from byprot import utils
from byprot.models.lm.cfp_gen import CondDiffusionProteinLanguageModel,CondDiffusionProteinLanguageModel2
import pandas as pd
from omegaconf import DictConfig
import pyrootutils
import pickle
from tqdm import tqdm
import random
import numpy as np
import time
import multiprocessing as mp
from torch.cuda.amp import autocast
import traceback

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_motif_gt_pos(target, model, motif_start_end, min_mask_ratio=0.05, max_mask_ratio=0.1):
    batch_size, sequence_length = target.shape
    masked_targets = []

    for i in range(batch_size):
        current_target = target[i].clone()

        non_special_sym_mask = (
                (current_target != model.pad_id) &
                (current_target != model.bos_id) &
                (current_target != model.eos_id)
        )
        effective_indices = torch.where(non_special_sym_mask)[0]

        if len(effective_indices) == 0:
            masked_targets.append(torch.full_like(current_target, fill_value=model.mask_id))
            continue

        total_length = len(effective_indices)
        retain_min_len = max(10, int(min_mask_ratio * total_length))
        retain_max_len = max(30, int(max_mask_ratio * total_length))

        start, end = motif_start_end[i]

        if start == 0 and end == 0:
            retain_length = torch.randint(retain_min_len, retain_max_len + 1, (1,)).item()
            retain_start_idx = torch.randint(0, total_length - retain_length + 1, (1,)).item()
            retain_start = effective_indices[retain_start_idx].item()
            retain_end = effective_indices[retain_start_idx + retain_length - 1].item()
        else:
            motif_length = end - start
            if motif_length < retain_min_len:
                retain_length = retain_min_len
            elif motif_length > retain_max_len:
                retain_length = retain_max_len
            else:
                retain_length = motif_length

            if end - start - retain_length > 0:
                retain_start = torch.randint(start, end - retain_length + 1, (1,)).item()
            else:
                retain_start = start

            retain_end = retain_start + retain_length - 1

        sequence_indices = torch.arange(sequence_length, device=target.device)
        mask = non_special_sym_mask & ((sequence_indices < retain_start) | (sequence_indices > retain_end))
        masked_target = current_target.clone()
        masked_target[mask] = model.mask_id

        masked_targets.append(masked_target)

    return torch.stack(masked_targets)


def get_initial(config, model, sample, length, tokenizer, device, sequence):

    go_labels = sample['go_f_mapped'] if 'go_f_mapped' in sample else sample['go_mapped']
    ipr_labels = sample['ipr_mapped']
    ec_labels = sample.get('EC_mapped', None)

    if config.get('use_seq_motif', False):
        motif_info = sample.get('motif', [])
        if motif_info:
            motif_start_end = [[motif['start'], motif['end']] for motif in motif_info][0]
        else:
            motif_start_end = [0, 0]
        length = len(sequence)

    # seq = ['<mask>'] * length

    # seq = [''.join(seq)]
    # init_seq = seq * 1 #config['num_seqs']

    seq_struct = tokenizer.all_tokens[50] * length
    seq_aa = "A" * length
    seq_struct = (
        tokenizer.struct_cls_token
        + seq_struct
        + tokenizer.struct_eos_token
    )
    seq_aa = tokenizer.aa_cls_token + seq_aa + tokenizer.aa_eos_token
    seq = (seq_struct, seq_aa)

    # init_seq = seq * 1 #config['num_seqs']
    input_tokens_batch = []

    for i in range(config['num_seqs']):
        # print(f"seq 0 : {seq[0]}")
        batch_struct = tokenizer.batch_encode_plus(
            [seq[0]],
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        batch_aatype = tokenizer.batch_encode_plus(
            [seq[1]],
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        input_tokens = torch.concat(
            [batch_struct["input_ids"], batch_aatype["input_ids"]], dim=1
        )
        input_tokens = input_tokens.to(device)
        input_tokens_batch.append(input_tokens[0])        
    
    # x_t = tokenizer.batch_encode_plus(init_seq,
    #                                     add_special_tokens=False,
    #                                     padding="longest",
    #                                     return_tensors='pt')

    if config.get('use_seq_motif', False):
        # TODO 还没实现motif
        raise NotImplementedError()
        seq_cond = tokenizer.batch_encode_plus([sequence],
                                               add_special_tokens=True,
                                               padding="longest",
                                               return_tensors='pt')['input_ids']
        seq_cond = get_motif_gt_pos(seq_cond, model, torch.tensor(motif_start_end).unsqueeze(0))

    # print(f"input_tokens_batch: {input_tokens_batch}")
    out_batch = {
        'input_ids': torch.stack(input_tokens_batch),
    }

    # Annotation tags
    if config.get('use_go', False) and len(go_labels):
        out_batch['go_label'] = torch.tensor(go_labels)

    if config.get('use_ipr', False) and len(ipr_labels):
        out_batch['ipr_label'] = torch.tensor(ipr_labels)

    if config.get('use_ec', False) and len(ec_labels):
        out_batch['ec_label'] = torch.tensor(ec_labels)

    if config.get('use_seq_motif', False):
        out_batch['seq_cond'] = seq_cond

    return utils.recursive_to(out_batch, device)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def split_data_sequentially(data, num_splits):
    split_size = len(data) // num_splits
    remainder = len(data) % num_splits

    splits = []
    start_idx = 0
    for i in range(num_splits):
        end_idx = start_idx + split_size + (1 if i < remainder else 0)
        splits.append(data[start_idx:end_idx])
        start_idx = end_idx

    return splits


def process_on_gpu(gpu_idx, part_data, config, part_fasta_filename):
    try:
        print(f"Starting processing on GPU {gpu_idx} with {len(part_data)} sequences...")

        model = CondDiffusionProteinLanguageModel2.from_pretrained(config['ckpt_path'])
        model = model.eval().cuda(gpu_idx)
        tokenizer = model.tokenizer

        # print("model.net")

        set_seed(config.get('seed', 42) + gpu_idx)

        with open(part_fasta_filename, 'a') as fp_save:
            for index, row in enumerate(part_data):

                sequence = row['sequence']
                seq_id = row['uniprot_id']

                print(f"Generating for protein {seq_id} on GPU {gpu_idx}:")

                seq_len = random.randint(config['seq_lens'][0], config['seq_lens'][1])
                device = torch.device(f"cuda:{gpu_idx}")

                # print(row)
                batch = get_initial(config, model, row, seq_len, tokenizer, device, sequence)

                # TODO no motif
                # partial_mask = batch['input_ids'].ne(model.mask_id).type_as(batch['input_mask'])
                partial_mask = None

                # print(f"input_ids: {batch}")
                with autocast():
                    outputs = model.generate(batch=batch,
                                             max_iter=config['max_iter'],
                                             sampling_strategy=config['sampling_strategy'],
                                             partial_masks=partial_mask)

                output_tokens = outputs[0]
                struct_tokens, aatype_tokens = output_tokens.chunk(2, dim=-1)

                output_results = list(
                    map(
                        lambda s: "".join(s.split()),
                        tokenizer.batch_decode(
                            aatype_tokens, skip_special_tokens=True
                        ),
                    )
                )

                for _, seq in enumerate(output_results):
                    seq = seq.replace(" ", "")
                    fp_save.write(f">SEQUENCE_ID={seq_id}_L={seq_len}\n")
                    fp_save.write(f"{seq}\n")

        print(f"Finished processing on GPU {gpu_idx}.")

    except Exception as e:
        print(f"Error occurred on GPU {gpu_idx}: {e}")
        traceback.print_exc()


def main(config):

    # multi-process
    mp.set_start_method('spawn', force=True)

    with open(config['input_data'], 'rb') as f:
        input_data = pickle.load(f)

    if config['debug']:
        input_data = input_data[:10]
        config['saveto'] = config['saveto'] + '_debug'


    # 8309
    # {'uniprot_id': 'Q60888', 'ipr_numbers': ['IPR000276', 'IPR017452', 'IPR000725'], 'go_numbers': {'C': ['GO:0016020', 'GO:0005886'], 'F': ['GO:0004930', 'GO:0004984'], 'P': ['GO:0050911', 'GO:0007186', 'GO:0007608']}, 'pdb_ids': [], 'sequence': 'MKNLSVVTQFILLGIPHTEGVETMLFVLFFSFYIFTLVGNLLILLAIVSSSRLHTPMYFFLCQLSVCDIFFPSVSSPKMLFYLSGNTPAISYAGCVSQLFFYHFLGGTECFLYTVMAYDRFVAICYPLRYSVIMSHRICAFLAMGTAVFGCIHSTFLTTLTFQLPYCGPKDVNYYFCDIPVVMKLACADTSTLEMVGFISVGLMPLSCFFFILTSYSCIVRSILQIRSTEGRHRAFSTCSAHFTAILLFYMPVIFIYLRPTPSPWLDATVQILNNLVTPMLNPLIYSLRNKEVKSSLWTVLHLLCFLPKHL', 'short_name': '10D1B_MOUSE', 'afdb': 'Q60888', 'ipr_mapped': [0, 1, 2], 'go_f_mapped': [0, 1], 'domain_sites': [{'ipr_number': 'IPR000276', 'ipr_description': 'G protein-coupled receptor, rhodopsin-like', 'domain_id': 'PR00237', 'start_position': 24, 'end_position': 48}, {'ipr_number': 'IPR000276', 'ipr_description': 'G protein-coupled receptor, rhodopsin-like', 'domain_id': 'PR00237', 'start_position': 57, 'end_position': 78}, {'ipr_number': 'IPR000276', 'ipr_description': 'G protein-coupled receptor, rhodopsin-like', 'domain_id': 'PR00237', 'start_position': 102, 'end_position': 124}, {'ipr_number': 'IPR000276', 'ipr_description': 'G protein-coupled receptor, rhodopsin-like', 'domain_id': 'PR00237', 'start_position': 268, 'end_position': 294}, {'ipr_number': 'IPR017452', 'ipr_description': 'GPCR, rhodopsin-like, 7TM', 'domain_id': 'PS50262', 'start_position': 39, 'end_position': 286}, {'ipr_number': 'IPR000725', 'ipr_description': 'Olfactory receptor', 'domain_id': 'PF13853', 'start_position': 31, 'end_position': 301}, {'ipr_number': 'IPR000725', 'ipr_description': 'Olfactory receptor', 'domain_id': 'PR00245', 'start_position': 90, 'end_position': 101}, {'ipr_number': 'IPR000725', 'ipr_description': 'Olfactory receptor', 'domain_id': 'PR00245', 'start_position': 127, 'end_position': 139}, {'ipr_number': 'IPR000725', 'ipr_description': 'Olfactory receptor', 'domain_id': 'PR00245', 'start_position': 174, 'end_position': 190}, {'ipr_number': 'IPR000725', 'ipr_description': 'Olfactory receptor', 'domain_id': 'PR00245', 'start_position': 234, 'end_position': 243}, {'ipr_number': 'IPR000725', 'ipr_description': 'Olfactory receptor', 'domain_id': 'PR00245', 'start_position': 279, 'end_position': 290}], 'motif': [{'go_term': 'G protein-coupled receptor activity', 'motif_segment': 'FYIFTLVGNLLILLAIV', 'start': 31, 'end': 48}, {'go_term': 'olfactory receptor activity', 'motif_segment': 'SYAGCVSQLFF', 'start': 90, 'end': 101}]}
    # print(len(input_data))
    # print(input_data[0])
    # exit()

    # detect multi-gpu
    gpu_num = torch.cuda.device_count()
    if gpu_num == 0:
        raise RuntimeError("No GPU devices found!")

    print(f"Detected {gpu_num} GPUs.")

    os.makedirs(config['saveto'], exist_ok=True)
    basename = os.path.basename(os.path.dirname(os.path.dirname(config['ckpt_path'])))

    start_time = time.time()

    if gpu_num == 1:
        final_fasta_filename = os.path.join(config['saveto'], f"{basename}_{config['run_name']}.fasta")
        process_on_gpu(0, input_data, config, final_fasta_filename)
    else:
        part_filenames = [
            os.path.join(config['saveto'], f"{basename}_{config['run_name']}_part_{i}.fasta") for i in range(gpu_num)
        ]
        data_parts = split_data_sequentially(input_data, gpu_num)
        processes = []
        for gpu_idx, part_data in enumerate(data_parts):
            p = mp.Process(target=process_on_gpu, args=(gpu_idx, part_data, config, part_filenames[gpu_idx]))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total computation time: {hours} hours, {minutes} minutes, {seconds} seconds")

    if gpu_num > 1:
        final_fasta_filename = os.path.join(config['saveto'], f"{basename}_{config['run_name']}.fasta")
        with open(final_fasta_filename, 'w') as final_fp:
            for part_fasta_filename in part_filenames:
                with open(part_fasta_filename, 'r') as part_fp:
                    final_fp.write(part_fp.read())

        print(f"All parts have been merged into {final_fasta_filename}")


if __name__ == '__main__':
    config_path = 'configs/test_cfpgen_dplm2.yaml'
    config = load_config(config_path)
    main(config)
