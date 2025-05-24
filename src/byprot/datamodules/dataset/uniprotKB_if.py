
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import json
import pickle
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from byprot import utils
from collections import Counter
from torch.nn import functional as F
from torch.utils.data.datapipes.map import SequenceWrapper
from torch.utils.data.dataset import Subset

from .data_utils import Alphabet

import esm
from multiprocessing import Pool
from tqdm import tqdm
import logging

log = utils.get_logger(__name__)


def process_entry(entry, coord_dir, backbone_atoms, alphabet_set, max_length, filter_nan):
    seq = entry['seq']
    nan_detect = False
    entry_coord_path = os.path.join(coord_dir, f"{entry['uniprot_id']}.npy")

    try:
        backbone_coords_array = np.load(entry_coord_path)
    except FileNotFoundError:
        return None, 'missing_file'

    num_atoms = len(backbone_atoms)
    total_coords = backbone_coords_array.shape[0]

    if total_coords % num_atoms != 0:
        return None, 'invalid_coords'

    entry['coords'] = {}
    for i, atom in enumerate(backbone_atoms):
        # start_idx = i * seq_length
        # end_idx = (i + 1) * seq_length
        # atom_coords = backbone_coords_array[start_idx:end_idx]
        # entry['coords'][atom] = atom_coords

        entry['coords'][atom] = backbone_coords_array[i]

        if np.isnan(backbone_coords_array).any():
            nan_detect = True

    if filter_nan and nan_detect:
        return None, 'nan_coords'

    bad_chars = set([s for s in seq]).difference(alphabet_set)
    if len(bad_chars) > 0:
        return None, 'bad_chars'

    if len(seq) > max_length:
        return None, 'too_long'

    return entry, 'success'

def process_entry_wrapper(args):
    entry, coord_dir, backbone_atoms, alphabet_set, max_length, filter_nan = args
    return process_entry(entry, coord_dir, backbone_atoms, alphabet_set, max_length, filter_nan)

def UniProtKBInvFold(   # GO/IPR with bb training, load bb info form uniprot_bb_coords
        root=".data",
        split=("train", "valid", "test"),
        max_length=500,
        alphabet='ACDEFGHIKLMNPQRSTVWY',
        filter_nan=True,
):
    coord_dir = os.path.join(root, '../uniprot_bb_coords')
    split_paths = dict(
        train=os.path.join(root, 'train_bb.pkl'),
        valid=os.path.join(root, 'valid_bb.pkl'),
        test=os.path.join(root, 'test_bb.pkl'),
    )

    backbone_atoms = ['N', 'CA', 'C', 'O']
    alphabet_set = set([a for a in alphabet])
    discard_count = {
        'bad_chars': 0,
        'too_long': 0,
        'nan_coords': 0,
        'missing_file': 0,
        'invalid_coords': 0,
    }

    unified_dataset = []
    split_uniprot_ids = {'train': set(), 'valid': set(), 'test': set()}

    for split_name, split_path in split_paths.items():
        with open(split_path, 'rb') as f:
            split_dataset: List[Dict] = pickle.load(f)
            unified_dataset.extend(split_dataset)

            for entry in split_dataset:
                assert 'uniprot_id' in entry
                split_uniprot_ids[split_name].add(entry['uniprot_id'])

    # deduplicated
    unified_dataset = list({entry['uniprot_id']: entry for entry in reversed(unified_dataset)}.values())[::-1]

    # multi-process
    log.info(f"Processing {len(unified_dataset)} entries with 32 processes...")
    args = [
        (entry, coord_dir, backbone_atoms, alphabet_set, max_length, filter_nan)
        for entry in unified_dataset
    ]
    with Pool(processes=32) as pool:
        results = list(tqdm(pool.imap(process_entry_wrapper, args), total=len(unified_dataset)))

    # 统计结果
    processed_dataset = []
    for entry, status in results:
        if status == 'success':
            processed_dataset.append(entry)
        else:
            discard_count[status] += 1

    total_size = len(unified_dataset)

    log.info(f'Loaded data size: {len(processed_dataset)}/{total_size}. Discarded: {discard_count}.')

    # Split processed dataset based on uniprot_ids collected earlier
    dataset_splits = {'train': [], 'valid': [], 'test': []}

    for entry in processed_dataset:
        uniprot_id = entry.get('uniprot_id')
        if uniprot_id in split_uniprot_ids['train']:
            dataset_splits['train'].append(entry)
        elif uniprot_id in split_uniprot_ids['valid']:
            dataset_splits['valid'].append(entry)
        elif uniprot_id in split_uniprot_ids['test']:
            dataset_splits['test'].append(entry)

    # Convert splits into list format
    dataset_splits = [Subset(dataset_splits[key], [ele for ele in range(len(dataset_splits[key]))])
                      for key in split]

    # Log sizes of splits
    sizes = [f'{split[i]}: {len(dataset_splits[i])}' for i in range(len(dataset_splits))]
    msg_sizes = ', '.join(sizes)
    log.info(f'Size. {msg_sizes}')

    if len(dataset_splits) == 1:  # test case
        dataset_splits = dataset_splits[0]

    return dataset_splits, alphabet_set


def collate_batch(
    batch: List[Dict[str, Any]],
    batch_converter,
    transform=None,
    atoms=('N', 'CA', 'C', 'O')
):
    seqs, coords = [], []
    names = []
    for entry in batch:
        _seq, _coords = entry['seq'], entry['coords']
        seqs.append(_seq)
        # [L, 3] x 4 -> [L, 4, 3]
        coords.append(
            # np.stack([_coords[c] for c in ['N', 'CA', 'C', 'O']], 1)
            np.stack([_coords[c] for c in atoms], 1)
        )
        names.append(entry['name'])

    for atom in atoms:
        if np.isnan(_coords[atom]).any():
            log.warning(f"NaN detected in entry {i}, atom {atom}, name: {entry['name']}")

    coords, confidence, strs, tokens, lengths, coord_mask = batch_converter.from_lists(
        coords_list=coords, confidence_list=None, seq_list=seqs
    )

    # coords, tokens, coord_mask, lengths = featurize(batch, torch.device('cpu'), 0)
    # coord_mask = coord_mask > 0.5
    batch_data = {
        'coords': coords,
        'tokens': tokens,
        'confidence': confidence,
        'coord_mask': coord_mask,
        'lengths': lengths,
        'seqs': seqs,
        'names': names
    }

    if transform is not None:
        batch_data = transform(batch_data)

    return batch_data


class CoordBatchConverter(esm.data.BatchConverter):
    def __init__(self, alphabet, coord_pad_inf=False, coord_nan_to_zero=True, to_pifold_format=False):
        super().__init__(alphabet)
        self.coord_pad_inf = coord_pad_inf
        self.to_pifold_format = to_pifold_format
        self.coord_nan_to_zero = coord_nan_to_zero

    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x n_atoms x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x n_atoms x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        # self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))

        coords_and_confidence, strs, tokens = super().__call__(batch)

        if self.coord_pad_inf:
            # pad beginning and end of each protein due to legacy reasons
            coords = [
                F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.nan)
                for cd, _ in coords_and_confidence
            ]
            confidence = [
                F.pad(torch.tensor(cf), (1, 1), value=-1.)
                for _, cf in coords_and_confidence
            ]
        else:
            coords = [
                torch.tensor(cd) for cd, _ in coords_and_confidence
            ]
            confidence = [
                torch.tensor(cf) for _, cf in coords_and_confidence
            ]
        # coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)

        if self.to_pifold_format:
            coords, tokens, confidence = ToPiFoldFormat(X=coords, S=tokens, cfd=confidence)

        lengths = tokens.ne(self.alphabet.padding_idx).sum(1).long()
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
            lengths = lengths.to(device)

        coord_padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum([-2, -1]))
        confidence = confidence * coord_mask + (-1.) * coord_padding_mask

        if self.coord_nan_to_zero:
            coords[torch.isnan(coords)] = 0.

        return coords, confidence, strs, tokens, lengths, coord_mask

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


class ToSabdabDataFormat(object):
    def __init__(self, alphabet) -> None:
        self.alphabet_ori = alphabet

        from byprot.utils.protein import constants
        UNK = constants.ressymb_to_resindex['X']
        self.aa_map = {}
        for ind, tok in enumerate(alphabet.all_toks):
            if tok != '<pad>':
                self.aa_map[ind] = constants.ressymb_to_resindex.get(tok, UNK)
            else:
                self.aa_map[ind] = 21

    def _map_aatypes(self, tokens):
        sizes = tokens.size()
        mapped_aa_flat = tokens.new_tensor([self.aa_map[ind] for ind in tokens.flatten().tolist()])
        return mapped_aa_flat.reshape(*sizes)

    def __call__(self, batch_data) -> Any:
        """
            coords          -> `pos_heavyatom` [B, num_res, num_atom, 3]
            tokens          -> `aa` [B, num_res]
            coord_mask      -> `mask_heavyatom` [B, num_res, num_atom]
            all_zeros       -> `mask` [B, num_res]
            all_zeros       -> `chain_nb` [B, num_res]
            range           -> `res_nb` [B, num_res]
            coord_mask      -> `generate_flag` [B, num_res]
            all_ones        -> `fragment_type` [B, num_res]

            coord_padding_mask: coord_padding_mask
            confidence: confidence,
        """

        batch_data['pos_heavyatom'] = batch_data.pop('coords')
        batch_data['aa'] = self._map_aatypes(batch_data.pop('tokens'))
        batch_data['mask'] = batch_data.pop('coord_mask').bool()
        batch_data['mask_heavyatom'] = batch_data['mask'][:, :, None].repeat(1, 1, batch_data['pos_heavyatom'].shape[2])
        batch_data['chain_nb'] = torch.full_like(batch_data['aa'], fill_value=0, dtype=torch.int64)
        batch_data['res_nb'] = new_arange(batch_data['aa'])
        batch_data['generate_flag'] = batch_data['mask'].clone()
        batch_data['fragment_type'] = torch.full_like(batch_data['aa'], fill_value=1, dtype=torch.int64)

        return batch_data


def ToPiFoldFormat(X, S, cfd, pad_special_tokens=False):
    mask = torch.isfinite(torch.sum(X, [-2, -1]))  # atom mask
    numbers = torch.sum(mask, dim=1).long()

    S_new = torch.zeros_like(S)
    X_new = torch.zeros_like(X) + np.nan
    cfd_new = torch.zeros_like(cfd)

    for i, n in enumerate(numbers):
        X_new[i, :n] = X[i][mask[i] == 1]
        S_new[i, :n] = S[i][mask[i] == 1]
        cfd_new[i, :n] = cfd[i][mask[i] == 1]

    X = X_new
    S = S_new
    cfd = cfd_new

    return X, S, cfd_new


class Featurizer(object):
    def __init__(self, alphabet: Alphabet, 
                 to_pifold_format=False, 
                 coord_nan_to_zero=True,
                 atoms=('N', 'CA', 'C', 'O')):
        self.alphabet = alphabet
        self.batcher = CoordBatchConverter(
            alphabet=alphabet,
            coord_pad_inf=alphabet.add_special_tokens,
            to_pifold_format=to_pifold_format, 
            coord_nan_to_zero=coord_nan_to_zero
        )

        self.atoms = atoms

    def __call__(self, raw_batch: dict):
        seqs, coords, names, go_numbers, ipr_numbers, ec_numbers = [], [], [], [], [], []
        for entry in raw_batch:
            # [L, 3] x 4 -> [L, 4, 3]
            if isinstance(entry['coords'], dict):
                coords.append(np.stack([entry['coords'][atom] for atom in self.atoms], 1))
            else:
                coords.append(entry['coords'])
            seqs.append(entry['seq'])
            names.append(entry['name'])
            if 'go_numbers' in entry:
                if 'go_f_mapped' in entry:
                    go_numbers.append(entry['go_f_mapped'])
                elif 'go_mapped' in entry:
                    go_numbers.append(entry['go_mapped'])
            else:
                go_numbers.append([])
            if 'ipr_numbers' in entry:
                ipr_numbers.append(entry['ipr_mapped'])
            else:
                ipr_numbers.append([])
            if 'EC_number' in entry:
                ec_numbers.append(entry['EC_mapped'])
            else:
                ec_numbers.append([])

        # padding GO type
        max_go_len = max(len(go_list) for go_list in go_numbers)
        padded_go_numbers = [go_list + [-1] * (max_go_len - len(go_list)) for go_list in go_numbers]
        padded_go_numbers = torch.tensor(padded_go_numbers, dtype=torch.int64)  # 指定数据类型为int64

        # padding IPR type
        max_ipr_len = max(len(ipr_list) for ipr_list in ipr_numbers)
        padded_ipr_numbers = [ipr_list + [-1] * (max_ipr_len - len(ipr_list)) for ipr_list in ipr_numbers]
        padded_ipr_numbers = torch.tensor(padded_ipr_numbers, dtype=torch.int64)  # 指定数据类型为int64

        # padding EC type
        max_ec_len = max(len(ec_list) for ec_list in ec_numbers)
        padded_ec_numbers = [ec_list + [-1] * (max_ec_len - len(ec_list)) for ec_list in ec_numbers]
        padded_ec_numbers = torch.tensor(padded_ec_numbers, dtype=torch.int64)  # 指定数据类型为int64

        coords, confidence, strs, tokens, lengths, coord_mask = self.batcher.from_lists(
            coords_list=coords, confidence_list=None, seq_list=seqs
        )

        # coord_mask = coord_mask > 0.5
        batch = {
            'coords': coords,
            'tokens': tokens,
            'confidence': confidence,
            'coord_mask': coord_mask,
            'lengths': lengths,
            'seqs': seqs,
            'names': names,
            'go_type': padded_go_numbers,
            'ipr_type': padded_ipr_numbers,
            'ec_type': padded_ec_numbers,
        }
        return batch
