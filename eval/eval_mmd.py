import sys
import os
from metrics.similarity import mmd
from metrics.conditional import mrr
import pickle
from Bio import SeqIO

def load_generated_sequences(fasta_filename):
    generated = {}
    generated_list = []

    for record in SeqIO.parse(fasta_filename, "fasta"):
        generated[f"{record.id} {record.description}"] = str(record.seq)

    for key in generated:
        generated_list.append(generated[key])

    return generated, generated_list


def expand_gt_for_generated(generated_data, gt_data):
    expanded_gt_sequences = []
    expanded_gt_labels = []
    expanded_gt_ids = []
    # Create a mapping from uniprot_id to GT data

    if '|' in list(generated_data.keys())[0] and 'name' in gt_data[0]:
        gt_mapping = {item['name']: item for item in gt_data}   # w/ bb
    else:
        gt_mapping = {item['uniprot_id']: item for item in gt_data}      # w/o bb

    # Loop through generated data
    for gen_seq_id in generated_data:
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
            # Find the matching GT entry based on the extracted ID
        if gen_id in gt_mapping:
            matching_gt = gt_mapping[gen_id]

            # Append the GT sequence and labels to match the generated sequence
            expanded_gt_sequences.append(matching_gt['sequence'])
            expanded_gt_labels.append({
                'go': matching_gt['go_numbers']['F'],
                'ipr': matching_gt['ipr_numbers'],
                'ec': matching_gt['EC_number'],
            })
            expanded_gt_ids.append(gen_id)

    return expanded_gt_sequences, expanded_gt_labels, expanded_gt_ids



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python eval_mmd.py <key: go|ipr|ec> <fasta_filename> <gt_data>")
        sys.exit(1)

    key = sys.argv[1]
    fasta_filename = sys.argv[2]
    gt_data_file = sys.argv[3]

    if key not in ['go', 'ipr', 'ec']:
        print("Error: key must be one of ['go', 'ipr', 'ec']")
        sys.exit(1)

    print(f'Evaluating for {os.path.basename(fasta_filename)} with {key} annotation:')

    generated_dict, generated_list = load_generated_sequences(fasta_filename)

    with open(gt_data_file, 'rb') as f:
        gt_data = pickle.load(f)

    # robust to incomplete output
    expanded_gt_sequences, expanded_gt_labels, expanded_gt_ids = expand_gt_for_generated(generated_dict, gt_data)
    expanded_labels = [ele[key] for ele in expanded_gt_labels]

    label_terms = set()
    for label_list in expanded_labels:
        label_terms.update(label_list)

    new_gt_data = dict(
        sequence=expanded_gt_sequences,
        labels=expanded_labels,
        terms=label_terms
    )

    metrics = {
        'MRR': round(mrr(generated_list, new_gt_data['labels'], new_gt_data['sequence'], new_gt_data['labels'], terms=new_gt_data['terms']), 3),
        'MMD': round(mmd(generated_list, new_gt_data['sequence']), 3),
        'MMD-Gauss': round(mmd(generated_list, new_gt_data['sequence'], kernel='gaussian'), 3),
    }

    print(metrics)


