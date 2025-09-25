import yaml
import torch
from byprot.models.lm.cfp_gen import CondDiffusionProteinLanguageModel2
import pickle
import random
from torch.cuda.amp import autocast
import os

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_initial(config, model, sample, length, tokenizer, device, sequence):
    go_labels = sample['go_f_mapped'] if 'go_f_mapped' in sample else sample['go_mapped']
    ipr_labels = sample['ipr_mapped']
    ec_labels = sample.get('EC_mapped', None)

    seq_struct = tokenizer.all_tokens[50] * length
    seq_aa = "A" * length
    seq_struct = (
        tokenizer.struct_cls_token
        + seq_struct
        + tokenizer.struct_eos_token
    )
    seq_aa = tokenizer.aa_cls_token + seq_aa + tokenizer.aa_eos_token
    
    batch_struct = tokenizer.batch_encode_plus(
        [seq_struct],
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )

    batch_aatype = tokenizer.batch_encode_plus(
        [seq_aa],
        add_special_tokens=False,
        padding="longest",
        return_tensors="pt",
    )

    input_tokens = torch.concat(
        [batch_struct["input_ids"], batch_aatype["input_ids"]], dim=1
    )
    input_tokens = input_tokens.to(device)

    out_batch = {
        'input_ids': input_tokens,
    }

    if config.get('use_go', False) and len(go_labels):
        out_batch['go_label'] = torch.tensor(go_labels).to(device)

    if config.get('use_ipr', False) and len(ipr_labels):
        out_batch['ipr_label'] = torch.tensor(ipr_labels).to(device)

    if config.get('use_ec', False) and len(ec_labels):
        out_batch['ec_label'] = torch.tensor(ec_labels).to(device)

    return out_batch

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_single_protein(config_path, uniprot_id, output_fasta_path):
    # 加载配置
    config = load_config(config_path)

    input_data_path = config['input_data']
    
    # 加载输入数据
    with open(input_data_path, 'rb') as f:
        input_data = pickle.load(f)
    
    # 查找指定的uniprot_id
    target_protein = None
    for protein in input_data:
        if protein['uniprot_id'] == uniprot_id:
            target_protein = protein
            break
    
    if target_protein is None:
        raise ValueError(f"Protein with uniprot_id {uniprot_id} not found in input data")
    
    # 加载模型
    model = CondDiffusionProteinLanguageModel2.from_pretrained(config['ckpt_path'])
    model = model.eval().cuda()
    tokenizer = model.tokenizer
    
    set_seed(config.get('seed', 42))
    
    # 获取序列信息
    sequence = target_protein['sequence']
    seq_id = target_protein['uniprot_id']
    seq_len = random.randint(config['seq_lens'][0], config['seq_lens'][1])
    # seq_len = 250
    device = torch.device("cuda")
    
    # 准备输入
    batch = get_initial(config, model, target_protein, seq_len, tokenizer, device, sequence)
    
    # 生成序列
    with autocast():
        outputs = model.generate(batch=batch,
                                 max_iter=config['max_iter'],
                                 sampling_strategy=config['sampling_strategy'],
                                 partial_masks=None)
    
    # 解码结果
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
    
    history_details = outputs[2]

    # 保存结果
    with open(output_fasta_path, 'w') as f:
        for i, seq in enumerate(output_results):
            seq = seq.replace(" ", "")
            f.write(f">SEQUENCE_ID={seq_id}_L={seq_len}_end\n")
            f.write(f"{seq}\n")

        for i, history_detail in enumerate(history_details):
            if i % 25 == 0:
                tokens = history_detail['tokens']
                struct_tokens, aatype_tokens = tokens.chunk(2, dim=-1)

                scores = history_detail['scores']
                struc_scores, aatype_scores = scores.chunk(2, dim=-1)


                aatype_flattened_scores = aatype_scores.flatten()
                aatype_bottom2_scores = aatype_flattened_scores.topk(min(2 * (i + 1), aatype_flattened_scores.numel()), largest=False)  # largest=False 获取最小值
                aatype_bottom2_values = aatype_bottom2_scores.values.tolist()
                aatype_bottom2_values = [f"{score:.1f}" for score in aatype_bottom2_values]

                struc_flattened_scores = struc_scores.flatten()
                struc_bottom2_scores = struc_flattened_scores.topk(min(2 * (i + 1), struc_flattened_scores.numel()), largest=False)  # largest=False 获取最小值
                struc_bottom2_values = struc_bottom2_scores.values.tolist()
                struc_bottom2_values = [f"{score:.1f}" for score in struc_bottom2_values]

                # print(tokens)
                
                output_results = list(
                    map(
                        lambda s: "".join(s.split()),
                        tokenizer.batch_decode(
                            aatype_tokens, skip_special_tokens=True
                        ),
                    )
                )

                seq = output_results[-1]
                f.write(f">SEQUENCE_ID={seq_id}_L={seq_len}_sample_{i}_history\n")
                f.write(f"{seq}\n")
                f.write(f"# Bottom2_aa_scores: {aatype_bottom2_values}\n")
                f.write(f"# Bottom2_struct_scores: {struc_bottom2_values}\n")
    
    print(f"Generated sequences for {uniprot_id} saved to {output_fasta_path}")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 配置参数
    config_path = 'configs/test_cfpgen_dplm2_single.yaml'
    # input_data_path = 'path/to/your/input_data.pkl'  # 替换为您的输入数据路径
    uniprot_id = 'A3PN82'  # 替换为您想要生成的蛋白的UniProt ID
    output_fasta_path = f'gen_single/generated_sequence_{uniprot_id}.fasta'   # 输出文件路径
    
    
    generate_single_protein(config_path, uniprot_id, output_fasta_path)