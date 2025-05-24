
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
from pprint import pprint
import torch
import os
import pickle
from byprot import utils
from byprot.models.lm.dplm import DiffusionProteinLanguageModel

import random
from tqdm import tqdm


def format_check(args):
    seq_list = args.cond_seq
    cond_position = args.cond_position
    assert len(seq_list) == len(cond_position), \
        "The length of cond_seq and cond_position does not match."
    position_list = []
    for pos in cond_position:
        pos = pos.split('-')
        assert len(pos) == 2, \
            "The format of position is illegal, which is not correctly splited by \'-\'"
        start_pos, end_pos = int(pos[0]), int(pos[1])
        assert end_pos >= start_pos, "The end position is smaller than start position."
        position_list.append((start_pos, end_pos))
    # check if position segment has overlap
    temp_position_list = [pos for tup in position_list for pos in tup]
    for i in range(1, len(temp_position_list)-2, 2):
        assert temp_position_list[i+1] > temp_position_list[i], \
            "The position segment has overlap, which is not supported"
    # check if the length of each position segment and seq segment matches
    for i, (start_pos, end_pos) in enumerate(position_list):
        assert len(seq_list[i]) == (end_pos - start_pos + 1), \
            "The length of each position segment and seq segment does not match."
    return seq_list, position_list
        
        
def get_initial(args, length, tokenizer, device):
    seq = ['<mask>'] * length
    if args.cond_seq is not None:
        # Inpainting generation, conditioned on some sequence segments
        seq_segment_list, position_list = format_check(args)
        for i, (start_pos, end_pos) in enumerate(position_list):
            seq[start_pos:end_pos+1] = [char for char in seq_segment_list[i]]

    seq = [''.join(seq)]
    init_seq = seq * args.num_seqs
    batch = tokenizer.batch_encode_plus(init_seq,
                                add_special_tokens=True,
                                padding="longest",
                                return_tensors='pt')
    batch = {
        'input_ids':  batch['input_ids'],
        'input_mask': batch['attention_mask'].bool(),
    }
    # if cond_seq is None:
    #     batch['input_ids'], _ = _full_mask(batch['input_ids'].clone(), collater.alphabet)
    batch = utils.recursive_to(batch, device)
    # pprint(batch)
    return batch


def generate(args):
    model = DiffusionProteinLanguageModel.from_pretrained(args.model_name)
    tokenizer = model.tokenizer
    model = model.eval()
    model = model.cuda(); 
    device = next(model.parameters()).device

    with open(args.test_pkl, 'rb') as f:
        pkl_entries = pickle.load(f)
    print(f"Loaded {len(pkl_entries)} entries from {args.test_pkl}")

    os.makedirs(args.saveto, exist_ok=True)
    all_generated_sequences = []

    for entry_idx, entry in enumerate(pkl_entries):

        entry_generated_sequences = []
        for seq_len in args.seq_lens:
            max_iter = args.max_iter
            batch = get_initial(args, seq_len, tokenizer, device)
            partial_mask = batch['input_ids'].ne(model.mask_id).type_as(batch['input_mask'])
            with torch.cuda.amp.autocast():
                outputs = model.generate(batch=batch, tokenizer=tokenizer,
                                            max_iter=max_iter,
                                            sampling_strategy=args.sampling_strategy,
                                            partial_masks=partial_mask)
            output_tokens = outputs[0]

            output_results = [''.join(seq.split(' ')) for seq in tokenizer.batch_decode(output_tokens, skip_special_tokens=True)]

            for seq in output_results:
                entry_generated_sequences.append(
                    f">SEQUENCE_{entry['uniprot_id']}_L={seq_len}\n{seq}"
                )

        all_generated_sequences.extend(entry_generated_sequences)

    final_output_file = os.path.join(args.saveto, "dplm_t500_len200.fasta")
    with open(final_output_file, "w") as final_fp:
        final_fp.write("\n".join(all_generated_sequences))

    print(f"All generated sequences saved to {final_output_file}")
    
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='airkingbd/dplm_650m')
    parser.add_argument('--num_seqs', type=int, default=1)
    parser.add_argument('--seq_lens', nargs='*', type=int, default=[200])
    parser.add_argument('--saveto', type=str, default='generation-results/dplm_650m/')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sampling_strategy', type=str, default='gumbel_argmax')
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--test_pkl', type=str, default='data-bin/uniprotKB/cfpgen_general_dataset/test.pkl')
    # inpainting
    # Note: the format of --cond_position and --cond_seq should split by ','
    # the number and the length of segments should match.
    # Like this: 
    # --cond_position 1-4 8-10 (position starts from 0)
    # --cond_seq ALVE EME
    parser.add_argument('--cond_position', nargs='*', type=str)
    parser.add_argument('--cond_seq', nargs='*', type=str)
    args = parser.parse_args()
        
    generate(args)


if __name__ == '__main__':
    main()
