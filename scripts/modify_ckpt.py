import torch
import os
import argparse
from tqdm import tqdm


def make_ckpt_for_stage2(ckpt_path, save_path):

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    modified_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.net.esm.encoder.layer'):
            layer_num = k.split('.')[5]
            new_key = k.replace(f'model.net.esm.encoder.layer.{layer_num}', f'esm.encoder.seq_controlnet.{layer_num}.copied_block')
            modified_state_dict[new_key] = v

    combined_state_dict = modified_state_dict.copy()
    combined_state_dict.update(state_dict)

    new_state_dict = {}
    for key, value in combined_state_dict.items():
        if key.startswith('model.net.'):
            new_key = key.replace('model.net.', '', 1)
        else:
            new_key = key
        new_state_dict[new_key] = value

    checkpoint['state_dict'] = new_state_dict

    torch.save(checkpoint, save_path)
    print(f"Modified checkpoint saved to: {save_path}")


def make_ckpt_for_stage2_if(ckpt_path, save_path):

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    new_state_dict = {}
    for key, value in state_dict.items():

        if 'seq_controlnet' in key:
            continue
        if key.startswith('model.encoder.encoder'):
            new_key = key.replace('model.encoder.encoder', 'encoder', 1)
        elif key.startswith('model.encoder.out_proj'):
            new_key = key.replace('model.encoder.out_proj', 'out_proj', 1)
        elif key.startswith('model.decoder.net'):
            new_key = key.replace('model.decoder.net.', '', 1)
        elif key.startswith('model.net'):
            new_key = key.replace('model.net.', '', 1)
        else:
            new_key = key

        new_state_dict[new_key] = value

    checkpoint['state_dict'] = new_state_dict

    torch.save(checkpoint, save_path)
    print(f"Modified checkpoint saved to: {save_path}")


def make_ckpt_for_inverse_folding(ckpt1_path, ckpt2_path, save_path):
    """
    Combine the state_dict of two checkpoints by aligning keys that differ only by a 'decoder' prefix.
    """

    # Load checkpoints
    checkpoint1 = torch.load(ckpt1_path, map_location='cpu')
    checkpoint2 = torch.load(ckpt2_path, map_location='cpu')

    state_dict1 = checkpoint1['state_dict']
    state_dict2 = checkpoint2['state_dict']

    added_params = []

    # Loop through state_dict2 keys and align with state_dict1
    for key2 in tqdm(state_dict2.keys()):
        assert key2.startswith('model.net.')
        key1 = f"model.decoder.{key2[len('model.'):]}"

        # If the mapped key does not exist in state_dict1, add it
        if key1 in state_dict1:
            assert state_dict1[key1].sum() == state_dict2[key2].sum()
        else:
            state_dict1[key1] = state_dict2[key2]
            added_params.append(key1)

    # Save the updated checkpoint
    checkpoint1['state_dict'] = state_dict1
    torch.save(checkpoint1, save_path)

    # Print added parameters info
    print(f"Updated checkpoint saved to: {save_path}")
    print(f"Total new parameters added: {len(added_params)}")
    if added_params:
        print("List of added parameters:")
        for param_name in added_params:
            print(f"  - {param_name}")


def count_parameters_in_ckpt(ckpt_path):

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    total_params = sum(p.numel() for p in state_dict.values())

    if total_params >= 1e9:
        formatted_params = f"{total_params / 1e9:.2f}B"
    else:
        formatted_params = f"{total_params / 1e6:.2f}M"

    print(f"Total parameters in {ckpt_path}: {formatted_params}")
    return formatted_params



def main():
    parser = argparse.ArgumentParser(description="Select mode: RCFE、ZS or SFT")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['RCFE', 'ZS', 'SFT'],
        required=True,
        help="Please select mode: RCFE, ZS or SFT"
    )
    args = parser.parse_args()

    # For 2nd stage training with RCFE (to support motif condition).
    if args.mode == 'RCFE':
        run_name = 'cfpgen_general_dataset_stage1'
        src_path = os.path.join(f'byprot-checkpoints/{run_name}/checkpoints', 'last.ckpt')
        tgt_path = os.path.join('pretrained', f'{run_name}.ckpt')
        make_ckpt_for_stage2(src_path, tgt_path)

    # Combine the ckpts of CATH-pretrained model and CFP-Gen model. This is the zero-shot model that can be directly used for inference.
    elif args.mode == 'ZS':
        cath_name = 'cfpgen_650m_cath43_stage1'
        cfpgen_name = 'cfpgen_general_dataset_stage1'
        src_path1 = os.path.join(f'byprot-checkpoints/{cath_name}/checkpoints', 'best.ckpt')
        src_path2 = os.path.join(f'byprot-checkpoints/{cfpgen_name}/checkpoints', 'last.ckpt')
        tgt_path = os.path.join('pretrained', 'cfpgen_650m_cath43.ckpt')
        make_ckpt_for_inverse_folding(src_path1, src_path2, tgt_path)

    #This model is used for stage2 functional inverse folding training.
    elif args.mode == 'SFT':
        src_path = os.path.join('pretrained', 'cfpgen_650m_cath43.ckpt')
        tgt_path = os.path.join('pretrained', 'cfpgen_650m_cath43_for_SFT.ckpt')
        make_ckpt_for_stage2_if(src_path, tgt_path)

if __name__ == '__main__':
    main()



