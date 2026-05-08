import sys
import os
from metrics.similarity import mmd
from metrics.conditional import mrr
import pickle
from Bio import SeqIO
import argparse

def load_generated_sequences(fasta_filename):
    generated = {}
    generated_list = []

    for record in SeqIO.parse(fasta_filename, "fasta"):
        generated[f"{record.id} {record.description}"] = str(record.seq)

    for key in generated:
        generated_list.append(generated[key])

    return generated, generated_list


def expand_gt_for_generated(generated_data, gt_data, excluded_ids):
    expanded_gt_sequences = []
    expanded_gt_labels = []
    expanded_gt_ids = []
    filtered_generated_list = []

    if '|' in list(generated_data.keys())[0] and 'name' in gt_data[0]:
        gt_mapping = {item['name']: item for item in gt_data}   # w/ bb
    else:
        gt_mapping = {item['uniprot_id']: item for item in gt_data}      # w/o bb

    # Loop through generated data
    for gen_seq_id, gen_seq in generated_data.items():
        # Extract the sequence ID from the generated data
        if 'unknown' in gen_seq_id:
            gen_id = gen_seq_id.split(" ")[0].split("_")[0]
        elif '_ID=' in gen_seq_id:
            gen_id = gen_seq_id.split("_ID=")[1].split("_")[0]
        elif '|' in gen_seq_id:
            if 'name=' in gen_seq_id:
                gen_id = gen_seq_id.split('|')[0].split('name=')[1][:-1]
            elif 'name' in gt_data[0]:
                gen_id = gen_seq_id.split(' ')[0]
                if '_sampled_seq' in gen_id:
                    gen_id = gen_id.split('_sampled_seq')[0]
                gt_mapping = {item['uniprot_id']: item for item in gt_data}
            else:
                gen_id = gen_seq_id.split('_')[-1].split(' ')[0]
        elif 'SEQUENCE' in gen_seq_id:
            gen_id = gen_seq_id.split('_')[1]
        elif 'go_prompt_longest_motif_seq30' in gen_seq_id:
            gen_id = gen_seq_id.split('_')[-1]
        else:
            gen_id = gen_seq_id.split(' ')[0]
            if 'L=200' in gen_id:
                gen_id = gen_id.split('_')[1]
        
        # Check if gen_id is in the exclusion list
        if gen_id in excluded_ids:
            continue
            
        # Find the matching GT entry based on the extracted ID
        if gen_id in gt_mapping:
            matching_gt = gt_mapping[gen_id]

            # Append the GT sequence and labels to match the generated sequence
            expanded_gt_sequences.append(matching_gt['sequence'])
            expanded_gt_labels.append({
                'go': matching_gt['go_numbers']['F'],
                'ipr': matching_gt['ipr_numbers'],
                # 'ec': matching_gt['EC_number'],
            })
            expanded_gt_ids.append(gen_id)
            filtered_generated_list.append(gen_seq)

    return expanded_gt_sequences, expanded_gt_labels, expanded_gt_ids, filtered_generated_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MMD with optional exclusion list.")
    parser.add_argument('key', choices=['go', 'ipr', 'ec'], help="Key to evaluate (go, ipr, or ec)")
    parser.add_argument('fasta_filename', help="Path to the generated FASTA file")
    parser.add_argument('gt_data_file', help="Path to the ground truth pickle file")
    parser.add_argument('--exclude-file', '-ef', default='/AIRvePFS/dair/chenxr-data/repo/dplm/overlapping_uniprot_ids.txt', help="File containing Uniprot IDs to exclude")

    args = parser.parse_args()

    key = args.key
    fasta_filename = args.fasta_filename
    gt_data_file = args.gt_data_file
    exclude_file = args.exclude_file

    print(f'Evaluating for {os.path.basename(fasta_filename)} with {key} annotation:')

    excluded_ids = set()
    if os.path.exists(exclude_file):
        with open(exclude_file, 'r') as f:
            for line in f:
                excluded_ids.add(line.strip())
        print(f"Loaded {len(excluded_ids)} IDs to exclude from {exclude_file}")
    else:
        print(f"Exclude file {exclude_file} not found. Skipping exclusion.")

    generated_dict, _ = load_generated_sequences(fasta_filename)

    with open(gt_data_file, 'rb') as f:
        gt_data = pickle.load(f)

    # robust to incomplete output and apply exclusion filter
    expanded_gt_sequences, expanded_gt_labels, expanded_gt_ids, filtered_generated_list = expand_gt_for_generated(generated_dict, gt_data, excluded_ids)
    expanded_labels = [ele[key] for ele in expanded_gt_labels]

    label_terms = set()
    for label_list in expanded_labels:
        label_terms.update(label_list)

    new_gt_data = dict(
        sequence=expanded_gt_sequences,
        labels=expanded_labels,
        terms=label_terms
    )
    
    print(f"Filtered pairs count: {len(filtered_generated_list)}")

    # print(new_gt_data['terms'])

    metrics = {
        'MRR': round(mrr(filtered_generated_list, new_gt_data['labels'], new_gt_data['sequence'], new_gt_data['labels'], terms=new_gt_data['terms']), 3),
        'MMD': round(mmd(filtered_generated_list, new_gt_data['sequence']), 3),
        'MMD-Gauss': round(mmd(filtered_generated_list, new_gt_data['sequence'], kernel='gaussian'), 3),
    }

    print(metrics)
