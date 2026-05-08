#!/usr/bin/env python

import math
import os
import pickle
import re

import click as ck
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from src.byprot.utils.ontology import Ontology


def load_pkl_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def parse_protein_id(raw_id):
    if "prompt_first_seq30" in raw_id:
        return re.match(r"prompt_first_seq30_([\w\d]+)", raw_id).groups()[0], False
    if "name=" in raw_id:
        return re.match(r"name=([\w\d\.]+)", raw_id).groups()[0], True
    if "recovery" in raw_id:
        return re.match(r"([\w\d]+)", raw_id).groups()[0], False
    if "SEQUENCE_ID=" in raw_id:
        return re.match(r"SEQUENCE_ID=([\w\d\.]+)_L", raw_id).groups()[0], False
    if "SEQUENCE_" in raw_id:
        return re.match(r"SEQUENCE_([\w\d\.]+)_L=", raw_id).groups()[0], False
    if "_seq30_" in raw_id:
        return re.match(r"go_prompt_longest_motif_seq30_([\w\d\.]+)", raw_id).groups()[0], False
    return raw_id, False


def load_predictions(prediction_file):
    predictions = {}
    use_name = False
    with open(prediction_file) as f:
        for line in f:
            items = line.strip().split("\t")
            if len(items) < 3:
                continue
            prot_id, parsed_use_name = parse_protein_id(items[0])
            use_name = use_name or parsed_use_name
            go_id = items[1]
            score = float(items[2])
            if prot_id not in predictions:
                predictions[prot_id] = {}
            predictions[prot_id][go_id] = score
    preds = {k: list(v.keys()) for k, v in predictions.items()}
    return preds, use_name


def safe_metric(metric_func, y_true, y_pred, average):
    try:
        return metric_func(y_true, y_pred, average=average)
    except ValueError:
        return np.nan


def evaluate_on_ids(preds, gts, eval_ids, ontology):
    if not eval_ids:
        raise ValueError("No common proteins found between guided/unguided predictions and ground truth.")

    gt_list = [set(gts[uid]) for uid in eval_ids]
    pred_list = [set(preds[uid]) for uid in eval_ids]

    for i, this_gt_go in enumerate(gt_list):
        expanded_go = []
        for go in this_gt_go:
            expanded_go.extend(ontology.get_ancestors(go))
        gt_list[i] = set(expanded_go)

    unique_go_gt = set()
    for go_set in gt_list:
        unique_go_gt.update(go_set)

    unique_go_pred = set()
    for go_set in pred_list:
        unique_go_pred.update(go_set)

    unique_go = unique_go_gt & unique_go_pred
    if not unique_go:
        raise ValueError("No overlapping GO terms between prediction and ground truth after ancestor expansion.")
    for i, go_set in enumerate(pred_list):
        pred_list[i] = {go for go in go_set if go in unique_go}
    for i, go_set in enumerate(gt_list):
        gt_list[i] = {go for go in go_set if go in unique_go}

    mlb = MultiLabelBinarizer()
    mlb.fit(gt_list + pred_list)
    y_true_binary = mlb.transform(gt_list)
    y_pred_binary = mlb.transform(pred_list)

    metrics = {
        "precision_macro": precision_score(y_true_binary, y_pred_binary, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_binary, y_pred_binary, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true_binary, y_pred_binary, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true_binary, y_pred_binary, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true_binary, y_pred_binary, average="micro", zero_division=0),
        "auc_roc_macro": safe_metric(roc_auc_score, y_true_binary, y_pred_binary, "macro"),
        "auc_roc_micro": safe_metric(roc_auc_score, y_true_binary, y_pred_binary, "micro"),
        "aupr_macro": safe_metric(average_precision_score, y_true_binary, y_pred_binary, "macro"),
        "aupr_micro": safe_metric(average_precision_score, y_true_binary, y_pred_binary, "micro"),
        "num_go_terms": len(mlb.classes_),
        "num_samples": len(eval_ids),
    }
    return metrics


def fmt(value):
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return "nan"
    return f"{value:.4f}"


@ck.command()
@ck.option("--data-root", "-dr", required=True, help="Path to test.pkl", default="data-bin/uniprotKB/cfpgen_general_dataset/test.pkl")
# @ck.option("--guided-predictions", "-gp", required=True, help="Classifier guidance predictions TSV", default="/AIRvePFS/dair/chenxr-data/repo/cfpgen/generation-codefp-clf/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv")
# @ck.option("--guided-predictions", "-gp", required=True, help="Classifier guidance predictions TSV", default="/AIRvePFS/dair/chenxr-data/repo/cfpgen/generation-codefp-lora/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pn-lora-wandb_go-ipr-500iter-repeat_cut_preds_mf.tsv")
# @ck.option("--guided-predictions", "-gp", required=True, help="Classifier guidance predictions TSV", default="/AIRvePFS/dair/chenxr-data/repo/dplm/generation-dplm2-motif-scaffold/test_dplm2-motif-scaffold_preds_mf.tsv")
# @ck.option("--guided-predictions", "-gp", required=True, help="Classifier guidance predictions TSV", default="/AIRvePFS/dair/chenxr-data/repo/dplm/generation-dplm2-motif-scaffold-30_50_no_aa/test_dplm2-motif-scaffold_preds_mf.tsv")
# @ck.option("--guided-predictions", "-gp", required=True, help="Classifier guidance predictions TSV", default="/AIRvePFS/dair/chenxr-data/repo/dplm/generation-dplm2-motif-scaffold-10_30_no_aa/test_dplm2-motif-scaffold_preds_mf.tsv")
@ck.option("--guided-predictions", "-gp", required=True, help="Classifier guidance predictions TSV", default="/AIRvePFS/dair/chenxr-data/repo/dplm/generation-dplm2-motif-scaffold-30_no_aa/test_dplm2-motif-scaffold_preds_mf.tsv")
@ck.option("--unguided-predictions", "-up", required=True, help="No-guidance predictions TSV", default="/AIRvePFS/dair/chenxr-data/repo/cfpgen/generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv")
@ck.option("--ont", "-ont", default="mf", type=ck.Choice(["mf", "bp", "cc"]), help="GO subontology")
@ck.option("--output-log", "-ol", default="", help="Optional output log path")
def main(data_root, guided_predictions, unguided_predictions, ont, output_log):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    obo_path = os.path.join(base_dir, "data", "go.obo")
    ontology = Ontology(obo_path, with_rels=True)

    test_data = load_pkl_file(data_root)
    guided_preds, guided_use_name = load_predictions(guided_predictions)
    unguided_preds, unguided_use_name = load_predictions(unguided_predictions)

    use_name = guided_use_name or unguided_use_name
    ont_map = {"mf": "F", "bp": "P", "cc": "C"}
    ont_key = ont_map[ont]
    if use_name:
        gts = {ele["name"]: set(ele["go_numbers"][ont_key]) for ele in test_data}
    else:
        gts = {ele["uniprot_id"]: set(ele["go_numbers"][ont_key]) for ele in test_data}

    guided_ids = set(guided_preds.keys())
    unguided_ids = set(unguided_preds.keys())
    common_pred_ids = guided_ids & unguided_ids
    eval_ids = sorted(uid for uid in common_pred_ids if uid in gts)

    guided_metrics = evaluate_on_ids(guided_preds, gts, eval_ids, ontology)
    unguided_metrics = evaluate_on_ids(unguided_preds, gts, eval_ids, ontology)

    metric_order = [
        "f1_micro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "precision_macro",
        "recall_macro",
        "auc_roc_micro",
        "auc_roc_macro",
        "aupr_micro",
        "aupr_macro",
    ]

    print(f"Guided proteins: {len(guided_ids)}")
    print(f"Unguided proteins: {len(unguided_ids)}")
    print(f"Common predicted proteins: {len(common_pred_ids)}")
    print(f"Evaluated proteins (common & in GT): {len(eval_ids)}")
    print(f"GO terms (guided): {guided_metrics['num_go_terms']}")
    print(f"GO terms (unguided): {unguided_metrics['num_go_terms']}")
    print("")
    print("metric\tguided\tunguided\tdelta(guided-unguided)")
    for key in metric_order:
        guided_val = guided_metrics[key]
        unguided_val = unguided_metrics[key]
        if (
            isinstance(guided_val, (float, np.floating))
            and isinstance(unguided_val, (float, np.floating))
            and (math.isnan(guided_val) or math.isnan(unguided_val))
        ):
            delta_str = "nan"
        else:
            delta_str = f"{guided_val - unguided_val:.4f}"
        print(f"{key}\t{fmt(guided_val)}\t{fmt(unguided_val)}\t{delta_str}")

    if not output_log:
        guided_base = os.path.splitext(guided_predictions)[0]
        output_log = f"{guided_base}_vs_unguided_common-go-eval.log"

    with open(output_log, "w") as f:
        f.write(f"guided_predictions: {guided_predictions}\n")
        f.write(f"unguided_predictions: {unguided_predictions}\n")
        f.write(f"data_root: {data_root}\n")
        f.write(f"ontology: {ont}\n\n")
        f.write(f"guided_proteins: {len(guided_ids)}\n")
        f.write(f"unguided_proteins: {len(unguided_ids)}\n")
        f.write(f"common_predicted_proteins: {len(common_pred_ids)}\n")
        f.write(f"evaluated_proteins: {len(eval_ids)}\n")
        f.write(f"go_terms_guided: {guided_metrics['num_go_terms']}\n")
        f.write(f"go_terms_unguided: {unguided_metrics['num_go_terms']}\n\n")
        f.write("metric\tguided\tunguided\tdelta(guided-unguided)\n")
        for key in metric_order:
            guided_val = guided_metrics[key]
            unguided_val = unguided_metrics[key]
            if (
                isinstance(guided_val, (float, np.floating))
                and isinstance(unguided_val, (float, np.floating))
                and (math.isnan(guided_val) or math.isnan(unguided_val))
            ):
                delta_str = "nan"
            else:
                delta_str = f"{guided_val - unguided_val:.4f}"
            f.write(f"{key}\t{fmt(guided_val)}\t{fmt(unguided_val)}\t{delta_str}\n")
    print("")
    print(f"Saved comparison log to: {output_log}")


if __name__ == "__main__":
    main()
