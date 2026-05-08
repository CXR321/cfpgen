import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
from src.byprot.utils.ontology import Ontology

# ================= 配置 =================
TSV_PATH = './generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv'
TSV_PATH = './generation-results-cfpgen_650m/cfpgen_650m_go_preds_mf.tsv'

TRAIN_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
TEST_PATH = 'data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl'
GO_MAPPING_PATH = 'go_mapping.pkl'
GO_OBO_PATH = 'data/go.obo'
CONFIDENCE_THRESHOLD = 0.0

# ================= 1. 数据准备 =================
ontology = Ontology(GO_OBO_PATH)
with open(GO_MAPPING_PATH, 'rb') as f:
    index_to_go = {v: k for k, v in pickle.load(f).items()}

with open(TRAIN_PATH, 'rb') as f:
    train_data = pickle.load(f)

# 构建训练集索引用于 Unseen 检测
go_to_train_indices = defaultdict(set)
train_combos_set = set()
valid_train_gos = set()

for idx, entry in enumerate(train_data):
    go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
    combo = tuple(sorted(go_ids))
    train_combos_set.add(combo)
    for go_id in go_ids:
        go_to_train_indices[go_id].add(idx)
        valid_train_gos.add(go_id)

def is_strict_unseen_combo(go_list):
    if not go_list: return False
    if tuple(sorted(go_list)) in train_combos_set: return False
    sets = [go_to_train_indices[go_id] for go_id in go_list if go_id in go_to_train_indices]
    if len(sets) != len(go_list): return True
    return len(set.intersection(*sets)) == 0

# ================= 2. 加载预测 =================
df = pd.read_csv(TSV_PATH, sep='\t', header=None, names=['raw_id', 'go_id', 'score'])
df['uniprot_id'] = df['raw_id'].apply(lambda x: x.split('_L=')[0].replace('SEQUENCE_ID=', ''))
df = df[df['score'] >= CONFIDENCE_THRESHOLD]

# 预构建预测映射：pid -> set(pred_gos)
pid_to_preds = df.groupby('uniprot_id')['go_id'].apply(set).to_dict()

# ================= 3. 筛选目标 =================
with open(TEST_PATH, 'rb') as f:
    test_data = pickle.load(f)

strict_unseen_pids = set()
seen_gt_combos = set()

for entry in test_data:
    pid = entry['uniprot_id']
    gt_go_ids = [index_to_go[i] for i in entry['go_f_mapped']]
    if is_strict_unseen_combo(gt_go_ids) and all(g in valid_train_gos for g in gt_go_ids):
        gt_key = frozenset(gt_go_ids)
        if gt_key not in seen_gt_combos:
            strict_unseen_pids.add(pid)
            seen_gt_combos.add(gt_key)

# ================= 4. 评估逻辑 =================
def evaluate_dataset(data_entries, filter_pids=None):
    correct_em = 0
    total = 0
    
    for entry in data_entries:
        pid = entry['uniprot_id']
        if filter_pids is not None and pid not in filter_pids:
            continue
        
        gt_set = set([index_to_go[i] for i in entry['go_f_mapped']])
        pred_set = pid_to_preds.get(pid, set())
        
        # Exact Match: GT ⊆ Pred (基于当前预测集合)
        if gt_set and gt_set.issubset(pred_set):
            correct_em += 1
        total += 1
        
    return correct_em / total if total > 0 else 0.0

print(f"Overall EMR: {evaluate_dataset(test_data):.4f}")
print(f"Strict Unseen EMR: {evaluate_dataset(test_data, filter_pids=strict_unseen_pids):.4f}")