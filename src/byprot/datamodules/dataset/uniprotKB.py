
import json
import pickle
import os
from typing import Union, TypeVar, Sequence
import torch.distributed as dist
import numpy as np
from transformers import EsmTokenizer
import math
from typing import Iterable
import torch
import numpy as np
from torch.utils.data import BatchSampler, Dataset, Sampler, DataLoader
from byprot import utils

from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer

log = utils.get_logger(__name__)

T_co = TypeVar('T_co', covariant=True)

class SortishSampler(Sampler):
    """Returns indices such that inputs with similar lengths are close together."""

    def __init__(
        self, sequence_lengths: Iterable, bucket_size: int, num_replicas: int = 1, rank: int = 0
    ):
        if dist.is_available():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        self.data = np.argsort(sequence_lengths)
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.data) * 1.0 / self.num_replicas))
        self.bucket_size = bucket_size
        n_buckets = int(np.ceil(len(self.data) / self.bucket_size))
        self.data = [
            self.data[i * bucket_size : i * bucket_size + bucket_size] for i in range(n_buckets)
        ]
        self.rank = rank
        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        np.random.seed(self.epoch)
        for bucket in self.data:
            np.random.shuffle(bucket)
        np.random.shuffle(self.data)
        indices = [item for sublist in self.data for item in sublist]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ApproxBatchSampler(BatchSampler):
    """
    Parameters:
    -----------
    sampler : Pytorch Sampler
            Choose base sampler class to use for bucketing

    max_tokens : int
            Maximum number of tokens per batch

    max_batch: int
            Maximum batch size

    sample_lengths : array-like
            List of lengths of sequences in the order of the dataset
    """

    def __init__(
        self,
        sampler,
        max_tokens,
        max_batch,
        sample_lengths,
        max_square_tokens=np.inf,
        drop_last=False,
        batch_size=None,
        max_len=512
    ):
        super().__init__(sampler, max_batch, drop_last)
        self.longest_token = 0
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.sampler = sampler
        self.sample_lengths = sample_lengths
        self.max_square_tokens = max_square_tokens
        self.max_len = max_len
        self.batches = self._build_batches()
        
    def _build_batches(self):
        batches = []
        length = 0
        ell_sq = 0
        batch = []
        for i, idx in enumerate(self.sampler):
            this_length = min(self.max_len, self.sample_lengths[idx])
            linear = (len(batch) + 1) * max(length, this_length)
            quadratic = (len(batch) + 1) * max(ell_sq, this_length**2)
            if linear <= self.max_tokens and quadratic < self.max_square_tokens:
                batch.append(idx)
                length = max(length, this_length)
                ell_sq = max(ell_sq, this_length**2)
                if len(batch) == self.max_batch:
                    batches.append(batch)
                    batch = []
                    length = 0
            else:
                if len(batch) == 0:
                    print('Current batch is empty! idx is ', idx)
                    continue
                batches.append(batch)
                batch = [idx]
                length = this_length
                ell_sq = this_length**2
        if len(batch) > 0:
            batches.append(batch)
            
        if self.sampler.num_replicas > 1:
            num_samples = torch.tensor(len(batches)).cuda()
            print(f'==============Local Rank {self.sampler.rank} Num Samples {num_samples}==============')
            dist.all_reduce(num_samples, op=dist.ReduceOp.MAX)
            print(f'==============All Reduce Num Samples {num_samples}==============')

            # print(batches[0].shape)

            # if num_samples > 100:
            #     print(batches)
            #     exit()
            num_samples = num_samples.item()

            if len(batches) < num_samples:
                # padding_size = num_samples - len(batches)
                a = num_samples // len(batches)
                b = num_samples % len(batches)
                new_batches = batches * a
                new_batches += batches[:b]
                assert len(new_batches) == num_samples
                batches = new_batches
            print(f'==============After Reduce, Rank{self.sampler.rank}, Num Samples {num_samples}==============')
        return batches
            
    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch
            
class UniProtKBDataset(Dataset):
    """
    Dataset that pulls from UniRef/Uniclust downloads.

    The data folder should contain the following:
    - 'consensus.fasta': consensus sequences, no line breaks in sequences
    - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of indices
    - 'lengths_and_offsets.npz': byte offsets for the 'consensus.fasta' and sequence lengths
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_len=1022,
    ):
        self.data_dir = data_dir
        self.split = split
        file_path = os.path.join(self.data_dir.data_dir, self.split+'.pkl')
        assert os.path.isfile(file_path)
        with open(file_path, 'rb') as f:
            self.indices = pickle.load(f)

        log.info(f"Dataset size: {len(self.indices)}")
        self.metadata_lens = [len(ele['sequence']) for ele in self.indices]
        self.max_len = self.data_dir.max_len

        # print("data index 0: ", self.indices[0])
        # exit()

    def __len__(self):
        return len(self.indices)

    def get_metadata_lens(self):
        return self.metadata_lens


    def __getitem__(self, idx):
        idx = self.indices[idx]
        consensus = idx['sequence']
        if 'go_f_mapped' in idx:
            go_type = idx['go_f_mapped']
        elif 'go_mapped' in idx:
            go_type = idx['go_mapped']
        ipr_type = idx.get('ipr_mapped', [])
        ec_type = idx.get('EC_mapped', [])

        motif_info = idx.get('motif', [])
        if motif_info:
            motif_start_end = [[motif['start'], motif['end']] for motif in motif_info][0]
        else:
            motif_start_end = [0, 0]

        if len(consensus) - self.max_len > 0:
            start = np.random.choice(len(consensus) - self.max_len)
            stop = start + self.max_len
        else:
            start = 0
            stop = len(consensus)
        consensus = consensus[start:stop]

        return consensus, go_type, ipr_type, ec_type, motif_start_end

class UniProtKB_DPLM2_Dataset(Dataset):
    """
    Dataset that pulls from UniRef/Uniclust downloads.

    The data folder should contain the following:
    - 'consensus.fasta': consensus sequences, no line breaks in sequences
    - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of indices
    - 'lengths_and_offsets.npz': byte offsets for the 'consensus.fasta' and sequence lengths
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        max_len=1022,
    ):
        self.data_dir = data_dir
        self.split = split
        file_path = os.path.join(self.data_dir.data_dir, self.split+'_expanded.pkl')
        assert os.path.isfile(file_path)
        with open(file_path, 'rb') as f:
            self.indices = pickle.load(f)

        log.info(f"Dataset size: {len(self.indices)}")
        self.metadata_lens = [len(ele['aa_seq']) for ele in self.indices]
        self.max_len = self.data_dir.max_len

        self.tokenizer = DPLM2Tokenizer.from_pretrained("airkingbd/dplm2_650m")

        # print("data index 0: ", self.indices[0])
        # exit()

    def __len__(self):
        return len(self.indices)

    def get_metadata_lens(self):
        return self.metadata_lens


    def __getitem__(self, idx):
        idx = self.indices[idx]
        consensus = idx['sequence']

        aatype_tokens = idx["aa_seq"]
        struct_tokens = idx["struct_seq"]


        # all is string
        struct_tokens = struct_tokens.split(",")

        # print(f"name: {idx['uniprot_id']} aatype_tokens: {len(aatype_tokens)} struct_tokens: {len(struct_tokens)} consensus: {len(consensus)}")
        # assert len(aatype_tokens) == len(struct_tokens) == len(consensus)

        if 'go_f_mapped' in idx:    
            go_type = idx['go_f_mapped']
        elif 'go_mapped' in idx:
            go_type = idx['go_mapped']
        ipr_type = idx.get('ipr_mapped', [])
        ec_type = idx.get('EC_mapped', [])

        motif_info = idx.get('motif', [])
        if motif_info:
            motif_start_end = [[motif['start'], motif['end']] for motif in motif_info][0]
        else:
            motif_start_end = [0, 0]

        if len(consensus) - self.max_len > 0:
            start = np.random.choice(len(consensus) - self.max_len)
            stop = start + self.max_len
        else:
            start = 0
            stop = len(consensus)

        consensus = consensus[start:stop]
        aatype_tokens = aatype_tokens[start:stop]
        struct_tokens = struct_tokens[start:stop]
        struct_tokens = "".join(struct_tokens) # convert to str example: [1234,1234] -> "12341234"

        struct_tokens = (
            self.tokenizer.struct_cls_token
            + struct_tokens
            + self.tokenizer.struct_eos_token
        )
        aatype_tokens = (
            self.tokenizer.aa_cls_token
            + aatype_tokens
            + self.tokenizer.aa_eos_token
        )

        # Now consensus not add cls and eos but struct and aa type tokens add
        # assert (len(aatype_tokens)-2*len(self.tokenizer.aa_cls_token)) == ((len(struct_tokens)-2*len(self.tokenizer.struct_cls_token))/4) == (len(consensus))

        return consensus, go_type, ipr_type, ec_type, motif_start_end, struct_tokens, aatype_tokens


class UniProtKBDatasetForTesting(Dataset):
    def __init__(
        self,
        max_len=2048,
        num_seqs=40,
    ):
        # self.max_len = max_len
        self.indices = []
        self.metadata_lens = []
        for i in range(1, 11):
            self.indices += ['A' * 100 * i] * num_seqs
            self.metadata_lens += [100 * i] * num_seqs

    def __len__(self):
        return len(self.indices)

    def get_metadata_lens(self):
        return self.metadata_lens
    
    def __getitem__(self, idx):
        consensus = self.indices[idx]
        return (consensus,)

class UniProtKB_DPLM2_DatasetForTesting(Dataset):
    def __init__(
        self,
        max_len=2048,
        num_seqs=40,
    ):
        # self.max_len = max_len
        self.indices = []
        self.metadata_lens = []
        for i in range(1, 11):
            self.indices += ['A' * 100 * i] * num_seqs
            self.metadata_lens += [100 * i] * num_seqs

    def __len__(self):
        return len(self.indices)

    def get_metadata_lens(self):
        return self.metadata_lens
    
    def __getitem__(self, idx):
        consensus = self.indices[idx]
        return (consensus,)


class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class DPLMCollater(object):
    def __init__(self, tokenizer_path=None):
        # by default we use the EsmTokenizer and the esm vocab. 
        # if you want to use the different vocab, 
        # please set the vocab path to the tokenizer_path
        if tokenizer_path is None:
            self.alphabet = EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')
        else:
            self.alphabet = EsmTokenizer.from_pretrained(tokenizer_path)


    def __call__(self, input_data):
        if len(list(zip(*input_data))) == 0:
            print("list idx error!")
            print(input_data)

        sequences = [ele[0] for ele in input_data]
        go_type = [ele[1] for ele in input_data]
        ipr_type = [ele[2] for ele in input_data]

        motif_start_end = [ele[4] for ele in input_data]

        # pdding for labels
        if len(input_data[0][3]):
            ec_type = [ele[3] for ele in input_data]
            max_length = max(len(ec_list) for ec_list in ec_type)
            if max_length>1:
                padded_ec_type = torch.tensor([item + [-1]*(max_length - len(item)) for item in ec_type])
            else:
                padded_ec_type = torch.tensor(ec_type)
        else:
            padded_ec_type = [[] for _ in input_data]

        # padding for seqs
        batch = self.alphabet.batch_encode_plus(sequences,
                                                add_special_tokens=True,
                                                padding="longest",
                                                return_tensors='pt')

        # pdding for labels
        max_go_len = max(len(go_list) for go_list in go_type)
        max_ipr_len = max(len(ipr_list) for ipr_list in ipr_type)
        padded_go_type = [go_list + [-1] * (max_go_len - len(go_list)) for go_list in go_type]
        padded_ipr_type = [ipr_list + [-1] * (max_ipr_len - len(ipr_list)) for ipr_list in ipr_type]

        padded_go_type = torch.tensor(padded_go_type)
        padded_ipr_type = torch.tensor(padded_ipr_type)

        batch = {
            'input_ids':  batch['input_ids'],
            'input_mask': batch['attention_mask'].bool(),
            'targets':    batch['input_ids'].clone(),
            'go_type':    padded_go_type,
            'ipr_type':   padded_ipr_type,
            'ec_type':    padded_ec_type,
            'motif_start_end': torch.tensor(motif_start_end)
        }

        return batch

class DPLM2Collater(object):
    def __init__(self):
        self.tokenizer = DPLM2Tokenizer.from_pretrained("airkingbd/dplm2_650m")
        self.alphabet = EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')

    def __call__(self, raw_batch):
        
        input_data = raw_batch
        # cfpgen batch start
        
        if len(list(zip(*input_data))) == 0:
            print("list idx error!")
            print(input_data)

        sequences = [ele[0] for ele in input_data]
        go_type = [ele[1] for ele in input_data]
        ipr_type = [ele[2] for ele in input_data]

        motif_start_end = [ele[4] for ele in input_data]

        # pdding for labels
        if len(input_data[0][3]):
            ec_type = [ele[3] for ele in input_data]
            max_length = max(len(ec_list) for ec_list in ec_type)
            if max_length>1:
                padded_ec_type = torch.tensor([item + [-1]*(max_length - len(item)) for item in ec_type])
            else:
                padded_ec_type = torch.tensor(ec_type)
        else:
            padded_ec_type = [[] for _ in input_data]

        # padding for seqs
        batch = self.alphabet.batch_encode_plus(sequences,
                                                add_special_tokens=True,
                                                padding="longest",
                                                return_tensors='pt')

        # pdding for labels
        max_go_len = max(len(go_list) for go_list in go_type)
        max_ipr_len = max(len(ipr_list) for ipr_list in ipr_type)
        padded_go_type = [go_list + [-1] * (max_go_len - len(go_list)) for go_list in go_type]
        padded_ipr_type = [ipr_list + [-1] * (max_ipr_len - len(ipr_list)) for ipr_list in ipr_type]

        padded_go_type = torch.tensor(padded_go_type)
        padded_ipr_type = torch.tensor(padded_ipr_type)

        cfpgen_batch = {
            'input_ids':  batch['input_ids'],
            'input_mask': batch['attention_mask'].bool(),
            'targets':    batch['input_ids'].clone(),
            'go_type':    padded_go_type,
            'ipr_type':   padded_ipr_type,
            'ec_type':    padded_ec_type,
            'motif_start_end': torch.tensor(motif_start_end)
        }        
        
        
        # DPLM-2 batch start
        if len(list(zip(*raw_batch))) == 0:
            print("list idx error!")
            print(raw_batch)

        # print(raw_batch)
        # exit()
        struct_tokens_list = [ele[5] for ele in input_data]
        # struct_tokens_list = [sample["struct_tokens"] for sample in raw_batch]

        batch_struct = self.tokenizer.batch_encode_plus(
            struct_tokens_list,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        batch_struct = {
            "targets": batch_struct["input_ids"],
            "attention_mask": batch_struct["attention_mask"].bool(),
        }

        aatype_list = [ele[6] for ele in input_data]
        # aatype_list = [sample["aatype_tokens"] for sample in raw_batch]
        batch_aatype = self.tokenizer.batch_encode_plus(
            aatype_list,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )
        batch_aatype = {
            "targets": batch_aatype["input_ids"],
            "attention_mask": batch_aatype["attention_mask"].bool(),
        }

        batch = {
            "struct_tokens": batch_struct,
            "aatype_tokens": batch_aatype,
        }

        if "pdb_name" in raw_batch[0]:
            pdb_name_list = [sample["pdb_name"] for sample in raw_batch]
            batch["pdb_name"] = pdb_name_list

        batch.update(cfpgen_batch)

        # assert len(batch["struct_tokens"]["targets"][0]) == len(batch["aatype_tokens"]["targets"][0]) == len(batch["input_ids"]) == len(batch["input_mask"][0]) == len(batch["targets"][0])

        return batch


def setup_dataloader(
        ds: UniProtKBDataset,
        max_tokens=6000, bucket_size=1000,
        max_batch_size=800,  num_workers=8,
        rank=0, world_size=1,
        mini_run=False,
        max_len=512,
    ) -> DataLoader:
    collater = DPLMCollater()
    if mini_run:
        dl = DataLoader(dataset=ds, shuffle=True, batch_size=1, num_workers=0, collate_fn=collater)
    else:
        lens = ds.get_metadata_lens()
        train_sortish_sampler = SortishSampler(lens, bucket_size, num_replicas=world_size, rank=rank)
        train_sampler = ApproxBatchSampler(train_sortish_sampler, max_tokens, max_batch_size, lens,
                                           max_len=max_len)
        dl = DataLoader(
            dataset=ds, batch_sampler=train_sampler, num_workers=num_workers, collate_fn=collater
        )
    return dl
    
def setup_dataloader_dplm2(
        ds: UniProtKB_DPLM2_Dataset,
        max_tokens=6000, bucket_size=1000,
        max_batch_size=800,  
        # max_batch_size=600,
        num_workers=8,
        rank=0, world_size=1,
        mini_run=False,
        max_len=512,
    ) -> DataLoader:
    collater = DPLM2Collater()
    if mini_run:
        dl = DataLoader(dataset=ds, shuffle=True, batch_size=1, num_workers=0, collate_fn=collater)
    else:
        lens = ds.get_metadata_lens()
        train_sortish_sampler = SortishSampler(lens, bucket_size, num_replicas=world_size, rank=rank)
        train_sampler = ApproxBatchSampler(train_sortish_sampler, max_tokens, max_batch_size, lens,
                                           max_len=max_len)
        dl = DataLoader(
            dataset=ds, batch_sampler=train_sampler, num_workers=num_workers, collate_fn=collater
        )

    # 检查dl的第一个元素
    # num = 0
    # for batch in dl:
    #     print(f"batch: {batch['struct_tokens']['targets'].shape}")
    #     num += 1
    # print(f"num: {num}")
    
    # exit()
    return dl