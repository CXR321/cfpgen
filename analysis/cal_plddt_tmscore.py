
# Copyright (c) 2023 Meta Platforms, Inc. and affiliates
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by Xinyou Wang on Jul 21, 2024
#
# Original file was released under MIT, with the full license text
# available at https://github.com/facebookresearch/esm/blob/main/LICENSE
#
# This modified file is released under the same license.


# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import re
import sys,os
from datetime import datetime
import argparse
import logging
import sys
import typing as T
from pathlib import Path
from timeit import default_timer as timer
from tqdm import tqdm
import pickle
import subprocess
import numpy as np
import multiprocessing as mp
from functools import partial

import torch
import esm


logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


PathLike = T.Union[str, Path]

def run_tmscore(query_pdb, reference_pdb, normalize_length=None, superpose_output=None):
    """
    运行 TMscore 计算 query PDB 与 reference PDB 之间的 TM-score。

    :param query_pdb: str, 预测的 PDB 文件路径
    :param reference_pdb: str, 参考的 PDB 文件路径
    :param normalize_length: int, 是否指定 TM-score 归一化长度（可选）
    :param superpose_output: str, 是否保存超位结构文件，例如 "TM_sup"（可选）
    :return: float, TM-score (normalized by reference) 或 np.nan（如果失败）
    """
    exec_path = "xxx/TMalign_cpp"
    cmd = [exec_path, query_pdb, reference_pdb]

    if normalize_length:
        cmd.extend(["-l", str(normalize_length)])  # 归一化长度
    if superpose_output:
        cmd.extend(["-o", superpose_output])  # 生成超位 PDB

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running TMscore: {e}")
        return np.nan

    # **提取 TM-score**
    matches = re.findall(r"TM-score\s*=\s*([\d\.]+)", output)
    if matches and len(matches) > 1:
        return float(matches[1])  # 选择第二个 TM-score（归一化到 GT）
    else:
        return np.nan

def compute_tm_score(sample_id, pred_pdb_path, gt_pdb_path):
    """
    计算单个 PDB 的 TM-score，并返回结果（不直接写入日志）。
    """
    tm_score = run_tmscore(pred_pdb_path, gt_pdb_path)
    return (sample_id, tm_score)  # 返回 (样本ID, TM-score)

def read_fasta(
        path,
        keep_gaps=True,
        keep_insertions=True,
        to_upper=False,
):
    with open(path, "r") as f:
        for result in read_alignment_lines(
                f, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
        ):
            yield result


def read_alignment_lines(
        lines,
        keep_gaps=True,
        keep_insertions=True,
        to_upper=False,
):
    seq = desc = None

    def parse(s):
        if not keep_gaps:
            s = re.sub("-", "", s)
        if not keep_insertions:
            s = re.sub("[a-z]", "", s)
        return s.upper() if to_upper else s

    for line in lines:
        # Line may be empty if seq % file_line_width == 0
        if len(line) > 0 and line[0] == ">":
            if seq is not None and 'X' not in seq:
                yield desc, parse(seq)
            desc = line.strip().lstrip(">")
            seq = ""
        else:
            assert isinstance(seq, str)
            seq += line.strip()
    assert isinstance(seq, str) and isinstance(desc, str)
    if 'X' not in seq:
        yield desc, parse(seq)


def enable_cpu_offloading(model):
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import enable_wrap, wrap

    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:9999", world_size=1, rank=0
    )

    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model

def read_fasta_all(fasta_path):
    """
    读取单个FASTA文件，并解析出所有蛋白的名字和序列，存入字典。

    :param fasta_path: str, FASTA文件路径
    :return: dict，键是蛋白的名字（header），值是蛋白的序列
    """
    protein_dict = {}  # 存储所有蛋白的名字和序列
    with open(fasta_path, 'r') as file:
        protein_name = None
        protein_seq = []

        for line in file:
            line = line.strip()
            if line.startswith(">"):  # 遇到新的蛋白名称
                if protein_name:  # 先存储上一个蛋白的序列
                    protein_dict[protein_name] = "".join(protein_seq)
                protein_name = line[1:]  # 去掉 ">"
                protein_seq = []  # 重新初始化序列列表
            else:
                protein_seq.append(line)  # 追加序列部分

        # 添加最后一个蛋白质
        if protein_name:
            protein_dict[protein_name] = "".join(protein_seq)

    return protein_dict


def init_model_on_gpu_with_cpu_offloading(model):
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model


def create_batched_sequence_datasest(
        sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:

    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--fasta",
        help="Path directory to input FASTA file",
        type=Path,
        required=True,
        default='ROOTDIR/' + \
                'generation-results/' + \
                'touchhigh_various_length_all',
    )
    parser.add_argument(
        "-gt", "--gt_path", help="Path directory to a pickle file with GT PDB info", type=Path, required=True,
        default=None
    )
    parser.add_argument(
        "-o", "--pdb", help="Path directory to output PDB directory", type=Path, default=None,
    )
    parser.add_argument(
        "-m", "--model-dir", help="Parent path to Pretrained ESM data directory. ", type=Path, default=None
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Number of recycles to run. Defaults to number used in training (4).",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help="Maximum number of tokens per gpu forward-pass. This will group shorter sequences together "
             "for batched prediction. Lowering this can help with out of memory issues, if these occur on "
             "short sequences.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunks axial attention computation to reduce memory usage from O(L^2) to O(L). "
             "Equivalent to running a for loop over chunks of of each dimension. Lower values will "
             "result in lower memory usage at the cost of speed. Recommended values: 128, 64, 32. "
             "Default: None.",
    )
    parser.add_argument("--cpu-only", help="CPU only", action="store_true")
    parser.add_argument("--cpu-offload", help="Enable CPU offloading", action="store_true")
    return parser


def run(args):

    if args.pdb is None:
        run_name = os.path.splitext(os.path.basename(args.fasta))[0]
        current_dir = os.getcwd()
        args.pdb = os.path.join(current_dir, 'cfp-gen_results', run_name+'_esmfold')
        args.pdb = Path(args.pdb)

    args.pdb.mkdir(exist_ok=True)

    logger.info("Loading model")

    # Use pre-downloaded ESM weights from model_pth.
    if args.model_dir is not None:
        # if pretrained model path is available
        torch.hub.set_dir(args.model_dir)

    assert os.path.exists(args.fasta)


    model = esm.pretrained.esmfold_v1()

    model = model.eval()
    model.set_chunk_size(args.chunk_size)

    if args.cpu_only:
        model.esm.float()  # convert to fp32 as ESM-2 in fp16 is not supported on CPU
        model.cpu()
    elif args.cpu_offload:
        model = init_model_on_gpu_with_cpu_offloading(model)
    else:
        model.cuda()

    all_sequences = sorted(read_fasta(args.fasta), key=lambda header_seq: len(header_seq[1]))
    logger.info(f"Loaded {len(all_sequences)} sequences from '{os.path.basename(args.fasta)}'")
    logger.info("Starting Predictions")
    batched_sequences = create_batched_sequence_datasest(all_sequences, args.max_tokens_per_batch)

    fasta_dir = os.path.dirname(args.fasta)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    log_file_path = os.path.join(fasta_dir, f"{os.path.basename(args.fasta).split('.')[0]}_esmfold_pLDDT_{timestamp}.log")  # 带时间戳的日志文件
    logger.info(f"Log file created: {log_file_path}")

    # 初始化计数
    num_completed = 0
    num_sequences = len(all_sequences)
    plddt_list = []
    ptm_list = []

    # 遍历批次
    for headers, sequences in batched_sequences:

        start = timer()
        try:
            output = model.infer(sequences, num_recycles=args.num_recycles)
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                if len(sequences) > 1:
                    logger.info(f"Failed (CUDA out of memory) to predict batch of size {len(sequences)}. "
                                "Try lowering `--max-tokens-per-batch`.")
                else:
                    logger.info(f"Failed (CUDA out of memory) on sequence {headers[0]} of length {len(sequences[0])}.")
                continue
            raise

        invalid_mask = (output["atom37_atom_exists"].sum(dim=(1, 2)) == 0)  # shape: (batch_size,)
        invalid_indices = torch.where(invalid_mask)[0].tolist()  # 获取无效样本的索引

        if invalid_indices:
            # 记录无效样本
            invalid_headers = [headers[i] for i in invalid_indices]
            for name in invalid_headers:
                logger.warning(f"Skipping {name} with empty `atom37_atom_exists`.")

            # 过滤 `headers` 和 `sequences`，去掉无效样本
            headers = [headers[i] for i in range(len(headers)) if i not in invalid_indices]
            sequences = [sequences[i] for i in range(len(sequences)) if i not in invalid_indices]

            if not headers:
                logger.error("All sequences in the batch are invalid. Skipping this batch.")
                continue

            try:
                output = model.infer(sequences, num_recycles=args.num_recycles)
            except RuntimeError as e:
                if e.args[0].startswith("CUDA out of memory"):
                    logger.info("CUDA out of memory again after filtering. Skipping batch.")
                    continue
                raise

        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)


        for header, pdb_str in zip(headers, pdbs):
            uniprot_id = header.split()[0]
            pdb_filename = args.pdb / f"{uniprot_id}.pdb"
            with open(pdb_filename, 'w') as pdb_file:
                pdb_file.write(pdb_str)
        logger.info(f"PDB saved to {args.pdb}")


        tottime = timer() - start
        time_string = f"{tottime / len(headers):0.1f}s"
        if len(sequences) > 1:
            time_string += f" (amortized, batch size {len(sequences)})"



    '''
    pLDDT
    '''
    # 追加写入 esmfold_eval_<timestamp>.log
    with open(log_file_path, 'a') as log_file:
        for header, seq, pdb_string, mean_plddt, ptm in zip(headers, sequences, pdbs, output["mean_plddt"], output["ptm"]):
            if 'name=' in header:
                header = header.split(' ')[0].split('name=')[1]

            output_file = os.path.join(args.pdb, f"{header}_plddt_{mean_plddt:.1f}.pdb")
            output_file = Path(output_file)
            output_file.write_text(pdb_string)

            plddt_list.append(mean_plddt.item())
            ptm_list.append(ptm.item())

            num_completed += 1
            log_entry = f"{header}, Length: {len(seq)}, pLDDT: {mean_plddt:.2f}, pTM: {ptm:.3f}"
            log_file.write(log_entry + "\n")

            logger.info(f"Predicted structure for {header} with length {len(seq)}, pLDDT {mean_plddt:.1f}, "
                        f"pTM {ptm:.3f} in {time_string}. "
                        f"{num_completed} / {num_sequences} completed.")

    # 计算并记录平均 pLDDT 和 pTM
    if plddt_list and ptm_list:
        avg_plddt = sum(plddt_list) / len(plddt_list)
        avg_ptm = sum(ptm_list) / len(ptm_list)
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"\nAverage pLDDT: {avg_plddt:.2f}\n")
            log_file.write(f"Average pTM: {avg_ptm:.3f}\n")

        logger.info(f"Prediction complete. Average pLDDT: {avg_plddt:.2f}, Average pTM: {avg_ptm:.3f}.")
        print(f"Prediction complete. Results saved to {log_file_path}")

    '''
    TM-score
    '''
    assert os.path.exists(args.gt_path)
    with open(args.gt_path, 'rb') as f:
        gt_data = pickle.load(f)  # gt_data 应该是 {sample_id: {"pdb_path": path}} 格式

    if 'name' in gt_data[0]:
        gt_dict = {ele['name']:ele['pdb_path'] for ele in gt_data}
    else:
        gt_dict = {ele['uniprot_id']:ele['pdb_path'] for ele in gt_data}

    fasta_dir = os.path.dirname(args.fasta)
    pdb_gt_dir = os.path.join(os.path.dirname(args.gt_path), 'pdb_afdb_files')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(fasta_dir, f"{os.path.basename(args.fasta).split('.')[0]}_esmfold_TM-score_{timestamp}.log")  # 带时间戳的日志文件
    # logger.info(f"Log file created: {log_file_path}")

    num_workers = min(32, mp.cpu_count())

    tasks = []
    for header in os.listdir(args.pdb):
        if header.endswith(".pdb"):
            if 'name' in gt_data[0]:
                sample_id = header.split("_plddt")[0]  # 提取样本 ID
            else:
                sample_id = header.split(' ')[0]   # todo debug: for uniprot_id

            pred_pdb_path = os.path.join(args.pdb, header)

            if sample_id in gt_dict:
                gt_pdb_name = os.path.basename(gt_dict[sample_id]).replace('.gz', '')
                gt_pdb_path = os.path.join(pdb_gt_dir, gt_pdb_name)

                # tasks.append((sample_id, pred_pdb_path, pred_pdb_path))
                tasks.append((sample_id, pred_pdb_path, gt_pdb_path))  # 任务参数

    # **使用 `multiprocessing.Pool` 进行并行计算**
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(compute_tm_score, tasks)  # 并行计算 TM-score

    # **统一写入日志**
    with open(log_file_path, 'a') as log_file:
        for sample_id, tm_score in results:
            log_file.write(f"TM-score for {sample_id}: {tm_score:.3f}\n")

    # **计算平均 TM-score 并写入日志**
    tm_scores = [tm_score for _, tm_score in results if not np.isnan(tm_score)]
    if tm_scores:
        avg_tm_score = sum(tm_scores) / len(tm_scores)
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"\nAverage TM-score: {avg_tm_score:.3f}\n")

        logger.info(f"Prediction complete. Average TM-score: {avg_tm_score:.3f}.")
        print(f"Prediction complete. TM-score results saved to {log_file_path}")


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
