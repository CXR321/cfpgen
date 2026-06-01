import argparse
import math
import pickle
import sys
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.byprot.utils.ontology import Ontology  # noqa: E402


DEFAULT_TRAIN_PATH = REPO_ROOT / "data-bin/uniprotKB/cfpgen_general_dataset/train_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl"
DEFAULT_TEST_PATH = REPO_ROOT / "data-bin/uniprotKB/cfpgen_general_dataset/test_all_old_motif_added_pfamMotif_esmfold_pfamEmb.pkl"
DEFAULT_STRICT_TEST_PATH = REPO_ROOT / "test_strict_unseen_repeated_10x.pkl"
DEFAULT_GO_MAPPING_PATH = REPO_ROOT / "go_mapping.pkl"
DEFAULT_GO_OBO_PATH = REPO_ROOT / "data/go.obo"
DEFAULT_PRED_TSV_PATH = REPO_ROOT / "generation-results-dplm2-goonly-unseen-all/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_nondup_preds_mf.tsv"
DEFAULT_FULL_TEST_PRED_TSV_PATH = REPO_ROOT / "generation-results-dplm2-goonly-alldata-dm-ca-me-scale-0.2-weight-headclloss-2.0_sn-pn-11wstep/cfpgen_general_dataset_stage1_dplm2_goonly_alldata_dm_ca_me-scale-0.2_weight-headclloss-2.0_sn-pnwandb_go-ipr-500iter-repeat_cut_preds_mf.tsv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis_outputs/unseen_go_pair_replacability"
MF_ROOT = "GO:0003674"


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def combo_from_entry(entry, index_to_go):
    return tuple(sorted(index_to_go[i] for i in entry["go_f_mapped"]))


def get_children(ontology, term_id):
    term = ontology.get_term(term_id)
    if term is None:
        return set()
    return set(term.get("children", set()))


def get_descendants(ontology, term_id):
    if not ontology.has_term(term_id):
        return set()
    descendants = set()
    queue = deque([term_id])
    while queue:
        current = queue.popleft()
        for child in get_children(ontology, current):
            if child not in descendants:
                descendants.add(child)
                queue.append(child)
    descendants.discard(term_id)
    return descendants


def get_siblings(ontology, term_id):
    siblings = set()
    for parent in ontology.get_parents(term_id):
        siblings.update(get_children(ontology, parent))
    siblings.discard(term_id)
    return siblings


def relation_type(ontology, source, target):
    if source == target:
        return "same"
    source_anc = ontology.get_ancestors(source)
    target_anc = ontology.get_ancestors(target)
    if source in target_anc - {target}:
        return "ancestor"
    if target in source_anc - {source}:
        return "descendant"
    if get_siblings(ontology, source) and target in get_siblings(ontology, source):
        return "sibling"
    if ontology.get_namespace(source) != ontology.get_namespace(target):
        return "cross_namespace"
    return "shared_ancestor_or_other"


def semantic_distance(ontology, left, right):
    return float(ontology.get_semantic_distance(left, right))


def lca_depth(ontology, left, right):
    inter = ontology.get_ancestors(left) & ontology.get_ancestors(right)
    if not inter:
        return -1.0
    return float(max(ontology.get_depth(term) for term in inter))


def normalized_pair_weight(value, max_value):
    if max_value <= 0:
        return 0.0
    return math.log1p(value) / math.log1p(max_value)


def relation_strength_from_type(rel_type):
    if rel_type == "same":
        return 1.0
    if rel_type == "sibling":
        return 0.9
    if rel_type in {"ancestor", "descendant"}:
        return 0.8
    return 0.0


def build_training_statistics(train_data, index_to_go, ontology):
    go_freq = Counter()
    combo_freq = Counter()
    pair_freq = Counter()
    neighbors = defaultdict(set)
    pair_to_proteins = defaultdict(set)

    for idx, entry in enumerate(train_data):
        combo = combo_from_entry(entry, index_to_go)
        combo_freq[combo] += 1
        for go_id in combo:
            go_freq[go_id] += 1
        for pair in combinations(combo, 2):
            pair_freq[pair] += 1
            left, right = pair
            neighbors[left].add(right)
            neighbors[right].add(left)
            pair_to_proteins[pair].add(idx)

    partner_degree = {go_id: len(nb) for go_id, nb in neighbors.items()}
    ontology._ensure_graph_loaded()

    return {
        "go_freq": go_freq,
        "combo_freq": combo_freq,
        "pair_freq": pair_freq,
        "neighbors": neighbors,
        "partner_degree": partner_degree,
        "pair_to_proteins": pair_to_proteins,
    }


def build_strict_unseen_unique_combos(strict_test_data, index_to_go):
    combo_to_entries = defaultdict(list)
    for entry in strict_test_data:
        combo = combo_from_entry(entry, index_to_go)
        combo_to_entries[combo].append(entry)
    return combo_to_entries


def parse_prediction_tsv(tsv_path):
    if not tsv_path or not Path(tsv_path).exists():
        return defaultdict(list)

    try:
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["raw_id", "go_id", "score"])
    except Exception:
        df = pd.read_csv(tsv_path, sep=r"\s+", header=None, names=["raw_id", "go_id", "score"])

    df = df[df["score"] >= 0.0].copy()
    by_clean = defaultdict(list)
    for raw_id, group in df.groupby("raw_id"):
        clean_id = str(raw_id).replace("SEQUENCE_ID=", "").split("_L=")[0]
        by_clean[clean_id].append(set(group["go_id"].tolist()))
    return by_clean


def compute_combo_performance(strict_test_data, index_to_go, pred_tsv_path):
    pred_dict = parse_prediction_tsv(pred_tsv_path)
    combo_perf = defaultdict(list)

    for entry in strict_test_data:
        combo = combo_from_entry(entry, index_to_go)
        pid = entry["uniprot_id"]
        preds = pred_dict.get(pid, [])
        if not preds:
            combo_perf[combo].append(0)
            continue
        best_raw_cover = max(int(set(combo).issubset(pred)) for pred in preds)
        combo_perf[combo].append(best_raw_cover)

    combo_exact_any = {combo: int(any(vals)) for combo, vals in combo_perf.items()}
    combo_exact_rate = {combo: float(np.mean(vals)) for combo, vals in combo_perf.items()}
    return combo_exact_any, combo_exact_rate


def safe_product(values):
    clean_values = [float(value) for value in values if pd.notna(value)]
    if not clean_values:
        return np.nan
    result = 1.0
    for value in clean_values:
        result *= value
    return result


def safe_min(values):
    clean_values = [float(value) for value in values if pd.notna(value)]
    if not clean_values:
        return np.nan
    return min(clean_values)


def build_go_difficulty_baseline(full_test_data, index_to_go, full_test_pred_tsv_path):
    pred_dict = parse_prediction_tsv(full_test_pred_tsv_path)
    go_presence_stats = defaultdict(lambda: {"count": 0, "hit": 0})
    go_single_stats = defaultdict(lambda: {"count": 0, "hit": 0})
    protein_rows = []

    for entry in full_test_data:
        pid = entry["uniprot_id"]
        gt_combo = combo_from_entry(entry, index_to_go)
        gt_set = set(gt_combo)
        preds = pred_dict.get(pid, [])
        pred_union = set().union(*preds) if preds else set()
        best_raw_cover = max((int(gt_set.issubset(pred)) for pred in preds), default=0)

        for go_id in gt_combo:
            hit = int(go_id in pred_union)
            go_presence_stats[go_id]["count"] += 1
            go_presence_stats[go_id]["hit"] += hit
            if len(gt_combo) == 1:
                go_single_stats[go_id]["count"] += 1
                go_single_stats[go_id]["hit"] += hit

        protein_rows.append(
            {
                "uniprot_id": pid,
                "combo": str(gt_combo),
                "combo_size": len(gt_combo),
                "raw_cover": best_raw_cover,
                "pred_union_size": len(pred_union),
            }
        )

    rows = []
    for go_id in sorted(set(go_presence_stats) | set(go_single_stats)):
        presence_count = go_presence_stats[go_id]["count"]
        presence_hit = go_presence_stats[go_id]["hit"]
        single_count = go_single_stats[go_id]["count"]
        single_hit = go_single_stats[go_id]["hit"]
        rows.append(
            {
                "go_id": go_id,
                "full_test_presence_count": presence_count,
                "full_test_presence_hit_rate": presence_hit / presence_count if presence_count else np.nan,
                "single_label_count": single_count,
                "single_label_hit_rate": single_hit / single_count if single_count else np.nan,
            }
        )

    go_baseline_df = pd.DataFrame(rows)
    protein_df = pd.DataFrame(protein_rows)
    baseline_lookup = go_baseline_df.set_index("go_id").to_dict(orient="index")
    return go_baseline_df, protein_df, baseline_lookup


def safe_mean(values):
    if not values:
        return np.nan
    return float(np.mean(values))


def compute_pair_link_prediction(neighbors, left, right):
    left_neighbors = neighbors.get(left, set())
    right_neighbors = neighbors.get(right, set())
    common = left_neighbors & right_neighbors
    union = left_neighbors | right_neighbors
    common_count = len(common)
    jaccard = common_count / len(union) if union else 0.0

    adamic_adar = 0.0
    resource_allocation = 0.0
    for mid in common:
        deg = max(1, len(neighbors.get(mid, set())))
        adamic_adar += 1.0 / math.log(max(2, deg))
        resource_allocation += 1.0 / deg

    preferential_attachment = float(len(left_neighbors) * len(right_neighbors))
    return {
        "common_neighbors": common_count,
        "neighbor_jaccard": jaccard,
        "adamic_adar": adamic_adar,
        "resource_allocation": resource_allocation,
        "preferential_attachment": preferential_attachment,
    }


def count_seen_edges_between_sets(pair_freq, left_terms, right_terms):
    if not left_terms or not right_terms:
        return 0, 0.0
    total = len(left_terms) * len(right_terms)
    seen = 0
    for left in left_terms:
        for right in right_terms:
            if left == right:
                continue
            if pair_freq[tuple(sorted((left, right)))] > 0:
                seen += 1
    ratio = seen / total if total else 0.0
    return seen, ratio


def compute_sibling_edge_support(anchor, unseen_partner, ontology, pair_freq):
    anchor_siblings = get_siblings(ontology, anchor)
    unseen_siblings = get_siblings(ontology, unseen_partner)

    anchor_sibling_to_unseen, anchor_sibling_to_unseen_ratio = count_seen_edges_between_sets(
        pair_freq, anchor_siblings, {unseen_partner}
    )
    anchor_to_unseen_sibling, anchor_to_unseen_sibling_ratio = count_seen_edges_between_sets(
        pair_freq, {anchor}, unseen_siblings
    )
    sibling_cross_edge_count, sibling_cross_edge_ratio = count_seen_edges_between_sets(
        pair_freq, anchor_siblings, unseen_siblings
    )

    return {
        "anchor_sibling_count": len(anchor_siblings),
        "unseen_partner_sibling_count": len(unseen_siblings),
        "anchor_sibling_to_unseen_edge_count": anchor_sibling_to_unseen,
        "anchor_sibling_to_unseen_edge_ratio": anchor_sibling_to_unseen_ratio,
        "anchor_to_unseen_sibling_edge_count": anchor_to_unseen_sibling,
        "anchor_to_unseen_sibling_edge_ratio": anchor_to_unseen_sibling_ratio,
        "sibling_cross_edge_count": sibling_cross_edge_count,
        "sibling_cross_edge_ratio": sibling_cross_edge_ratio,
    }


def collect_replacement_candidates(anchor, unseen_partner, known_neighbors, neighbors, pair_freq, ontology):
    max_pair_weight = max((pair_freq[tuple(sorted((anchor, nb)))] for nb in known_neighbors), default=0)
    unseen_partner_neighbors = neighbors.get(unseen_partner, set())
    rows = []

    for known_neighbor in known_neighbors:
        rel = relation_type(ontology, unseen_partner, known_neighbor)
        dist = semantic_distance(ontology, unseen_partner, known_neighbor)
        lca = lca_depth(ontology, unseen_partner, known_neighbor)
        known_neighbor_neighbors = neighbors.get(known_neighbor, set())
        common_neighbors = len(unseen_partner_neighbors & known_neighbor_neighbors)
        union_neighbors = len(unseen_partner_neighbors | known_neighbor_neighbors)
        neighbor_jaccard = common_neighbors / union_neighbors if union_neighbors else 0.0
        common_neighbors_norm = common_neighbors / max(1, min(len(unseen_partner_neighbors), len(known_neighbor_neighbors)))
        anchor_pair_freq = pair_freq[tuple(sorted((anchor, known_neighbor)))]
        pair_weight = normalized_pair_weight(anchor_pair_freq, max_pair_weight)
        tree_similarity = max(relation_strength_from_type(rel), 1.0 / (1.0 + dist))
        graph_similarity = 0.6 * neighbor_jaccard + 0.4 * common_neighbors_norm
        replaceability_score = 0.45 * tree_similarity + 0.35 * graph_similarity + 0.20 * pair_weight

        rows.append(
            {
                "anchor_go": anchor,
                "unseen_partner_go": unseen_partner,
                "known_partner_go": known_neighbor,
                "anchor_known_pair_freq": anchor_pair_freq,
                "relation_type": rel,
                "semantic_distance": dist,
                "lca_depth": lca,
                "known_partner_degree": len(known_neighbor_neighbors),
                "unknown_partner_degree": len(unseen_partner_neighbors),
                "common_neighbors": common_neighbors,
                "neighbor_jaccard": neighbor_jaccard,
                "common_neighbors_norm": common_neighbors_norm,
                "pair_weight_norm": pair_weight,
                "tree_similarity": tree_similarity,
                "graph_similarity": graph_similarity,
                "replaceability_score": replaceability_score,
            }
        )

    return rows


def aggregate_unseen_pair_statistics(
    unseen_pairs,
    combo_to_entries,
    training_stats,
    ontology,
    combo_exact_any,
    combo_exact_rate,
    go_baseline_lookup,
):
    neighbors = training_stats["neighbors"]
    pair_freq = training_stats["pair_freq"]
    go_freq = training_stats["go_freq"]
    partner_degree = training_stats["partner_degree"]
    detailed_rows = []
    summary_rows = []
    anchor_summary = defaultdict(lambda: {"unknown_targets": set(), "known_neighbors": set(), "replaceability_scores": []})

    for left, right in sorted(unseen_pairs):
        combos_with_pair = [combo for combo in combo_to_entries if left in combo and right in combo]
        unordered_pair_key = tuple(sorted((left, right)))
        pair_level_perf = [combo_exact_rate.get(combo, 0.0) for combo in combos_with_pair]
        pair_level_any = [combo_exact_any.get(combo, 0) for combo in combos_with_pair]
        link_stats = compute_pair_link_prediction(neighbors, left, right)

        for anchor, unseen_partner in [(left, right), (right, left)]:
            known_neighbors = sorted(neighbors.get(anchor, set()))
            candidate_rows = collect_replacement_candidates(
                anchor=anchor,
                unseen_partner=unseen_partner,
                known_neighbors=known_neighbors,
                neighbors=neighbors,
                pair_freq=pair_freq,
                ontology=ontology,
            )
            detailed_rows.extend(candidate_rows)

            parent_support = len(ontology.get_parents(unseen_partner) & set(known_neighbors))
            sibling_support = len(get_siblings(ontology, unseen_partner) & set(known_neighbors))
            child_support = len(get_children(ontology, unseen_partner) & set(known_neighbors))
            ancestor_support = len((ontology.get_ancestors(unseen_partner) - {unseen_partner}) & set(known_neighbors))
            descendant_support = len(get_descendants(ontology, unseen_partner) & set(known_neighbors))
            sibling_edge_support = compute_sibling_edge_support(
                anchor=anchor,
                unseen_partner=unseen_partner,
                ontology=ontology,
                pair_freq=pair_freq,
            )

            best_candidate = max(candidate_rows, key=lambda row: row["replaceability_score"], default=None)
            anchor_baseline = go_baseline_lookup.get(anchor, {})
            unseen_baseline = go_baseline_lookup.get(unseen_partner, {})
            anchor_full_rate = anchor_baseline.get("full_test_presence_hit_rate", np.nan)
            unseen_full_rate = unseen_baseline.get("full_test_presence_hit_rate", np.nan)
            anchor_single_rate = anchor_baseline.get("single_label_hit_rate", np.nan)
            unseen_single_rate = unseen_baseline.get("single_label_hit_rate", np.nan)
            expected_pair_rate_fulltest = safe_min([anchor_full_rate, unseen_full_rate])
            expected_pair_rate_single = safe_min([anchor_single_rate, unseen_single_rate])
            observed_pair_rate = safe_mean(pair_level_perf)
            any_tree_support = int(
                any(
                    value > 0
                    for value in [
                        parent_support,
                        sibling_support,
                        child_support,
                        ancestor_support,
                        descendant_support,
                    ]
                )
            )
            any_graph_support = int(
                any(
                    value > 0
                    for value in [
                        link_stats["common_neighbors"],
                        sibling_edge_support["anchor_sibling_to_unseen_edge_count"],
                        sibling_edge_support["anchor_to_unseen_sibling_edge_count"],
                        sibling_edge_support["sibling_cross_edge_count"],
                    ]
                )
            )
            theoretically_generalizable = int(any_tree_support or any_graph_support)
            summary_rows.append(
                {
                    "pair_unordered": str(unordered_pair_key),
                    "anchor_go": anchor,
                    "unseen_partner_go": unseen_partner,
                    "anchor_freq": go_freq[anchor],
                    "unseen_partner_freq": go_freq[unseen_partner],
                    "anchor_partner_degree": partner_degree.get(anchor, 0),
                    "unseen_partner_degree": partner_degree.get(unseen_partner, 0),
                    "anchor_known_neighbor_count": len(known_neighbors),
                    "anchor_parent_support_count": parent_support,
                    "anchor_sibling_support_count": sibling_support,
                    "anchor_child_support_count": child_support,
                    "anchor_ancestor_support_count": ancestor_support,
                    "anchor_descendant_support_count": descendant_support,
                    "anchor_sibling_count": sibling_edge_support["anchor_sibling_count"],
                    "unseen_partner_sibling_count": sibling_edge_support["unseen_partner_sibling_count"],
                    "anchor_sibling_to_unseen_edge_count": sibling_edge_support["anchor_sibling_to_unseen_edge_count"],
                    "anchor_sibling_to_unseen_edge_ratio": sibling_edge_support["anchor_sibling_to_unseen_edge_ratio"],
                    "anchor_to_unseen_sibling_edge_count": sibling_edge_support["anchor_to_unseen_sibling_edge_count"],
                    "anchor_to_unseen_sibling_edge_ratio": sibling_edge_support["anchor_to_unseen_sibling_edge_ratio"],
                    "sibling_cross_edge_count": sibling_edge_support["sibling_cross_edge_count"],
                    "sibling_cross_edge_ratio": sibling_edge_support["sibling_cross_edge_ratio"],
                    "link_common_neighbors": link_stats["common_neighbors"],
                    "link_neighbor_jaccard": link_stats["neighbor_jaccard"],
                    "link_adamic_adar": link_stats["adamic_adar"],
                    "link_resource_allocation": link_stats["resource_allocation"],
                    "link_preferential_attachment": link_stats["preferential_attachment"],
                    "best_known_partner_go": best_candidate["known_partner_go"] if best_candidate else "",
                    "best_relation_type": best_candidate["relation_type"] if best_candidate else "",
                    "best_semantic_distance": best_candidate["semantic_distance"] if best_candidate else np.nan,
                    "best_replaceability_score": best_candidate["replaceability_score"] if best_candidate else np.nan,
                    "min_semantic_distance_to_known": min((row["semantic_distance"] for row in candidate_rows), default=np.nan),
                    "mean_semantic_distance_to_known": safe_mean([row["semantic_distance"] for row in candidate_rows]),
                    "max_neighbor_jaccard_to_known": max((row["neighbor_jaccard"] for row in candidate_rows), default=np.nan),
                    "max_common_neighbors_to_known": max((row["common_neighbors"] for row in candidate_rows), default=0),
                    "strict_combo_count_containing_pair": len(combos_with_pair),
                    "strict_combo_exact_any_fraction": safe_mean(pair_level_any),
                    "strict_combo_exact_rate_mean": observed_pair_rate,
                    "anchor_full_test_hit_rate": anchor_full_rate,
                    "unseen_partner_full_test_hit_rate": unseen_full_rate,
                    "mean_full_test_hit_rate": safe_mean([anchor_full_rate, unseen_full_rate]),
                    "min_full_test_hit_rate": np.nanmin([anchor_full_rate, unseen_full_rate])
                    if pd.notna(anchor_full_rate) or pd.notna(unseen_full_rate)
                    else np.nan,
                    "anchor_single_label_hit_rate": anchor_single_rate,
                    "unseen_partner_single_label_hit_rate": unseen_single_rate,
                    "mean_single_label_hit_rate": safe_mean([anchor_single_rate, unseen_single_rate]),
                    "expected_pair_rate_fulltest": expected_pair_rate_fulltest,
                    "expected_pair_rate_single": expected_pair_rate_single,
                    "residual_vs_fulltest_pair_expectation": observed_pair_rate - expected_pair_rate_fulltest
                    if pd.notna(expected_pair_rate_fulltest)
                    else np.nan,
                    "residual_vs_single_pair_expectation": observed_pair_rate - expected_pair_rate_single
                    if pd.notna(expected_pair_rate_single)
                    else np.nan,
                    "any_tree_support": any_tree_support,
                    "any_graph_support": any_graph_support,
                    "theoretically_generalizable": theoretically_generalizable,
                }
            )

            anchor_summary[anchor]["unknown_targets"].add(unseen_partner)
            anchor_summary[anchor]["known_neighbors"].update(known_neighbors)
            if best_candidate:
                anchor_summary[anchor]["replaceability_scores"].append(best_candidate["replaceability_score"])

    anchor_rows = []
    for anchor, payload in anchor_summary.items():
        anchor_rows.append(
            {
                "anchor_go": anchor,
                "known_neighbor_count": len(payload["known_neighbors"]),
                "unknown_target_count": len(payload["unknown_targets"]),
                "mean_best_replaceability_score": safe_mean(payload["replaceability_scores"]),
            }
        )

    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = pd.DataFrame(summary_rows)
    anchor_df = pd.DataFrame(anchor_rows).sort_values(
        ["unknown_target_count", "mean_best_replaceability_score"], ascending=[False, False]
    )
    return detailed_df, summary_df, anchor_df


def build_performance_relation_table(summary_df):
    if summary_df.empty:
        return pd.DataFrame(), {}

    relation_rows = []
    feature_cols = [
        "anchor_known_neighbor_count",
        "anchor_sibling_support_count",
        "anchor_parent_support_count",
        "anchor_child_support_count",
        "anchor_ancestor_support_count",
        "anchor_descendant_support_count",
        "anchor_sibling_to_unseen_edge_count",
        "anchor_to_unseen_sibling_edge_count",
        "sibling_cross_edge_count",
        "link_common_neighbors",
        "link_neighbor_jaccard",
        "best_replaceability_score",
        "mean_full_test_hit_rate",
        "min_full_test_hit_rate",
        "expected_pair_rate_fulltest",
        "expected_pair_rate_single",
        "residual_vs_fulltest_pair_expectation",
        "residual_vs_single_pair_expectation",
        "anchor_freq",
        "unseen_partner_freq",
        "anchor_partner_degree",
        "unseen_partner_degree",
    ]
    for feature in feature_cols:
        x = pd.to_numeric(summary_df[feature], errors="coerce")
        y = pd.to_numeric(summary_df["strict_combo_exact_rate_mean"], errors="coerce")
        mask = ~(x.isna() | y.isna())
        corr = np.corrcoef(x[mask], y[mask])[0, 1] if mask.sum() >= 3 else np.nan
        relation_rows.append({"feature": feature, "corr_with_exact_rate": corr})

    binary_conditions = {
        "tree_support": summary_df["any_tree_support"] > 0,
        "graph_support": summary_df["any_graph_support"] > 0,
        "theoretical_generalizable": summary_df["theoretically_generalizable"] > 0,
        "common_neighbors": summary_df["link_common_neighbors"] > 0,
        "sibling_cross_edges": summary_df["sibling_cross_edge_count"] > 0,
        "sibling_support": summary_df["anchor_sibling_support_count"] > 0,
        "parent_or_child_support": (summary_df["anchor_parent_support_count"] > 0)
        | (summary_df["anchor_child_support_count"] > 0),
    }
    for name, mask in binary_conditions.items():
        supported = summary_df[mask]
        unsupported = summary_df[~mask]
        relation_rows.append(
            {
                "feature": f"{name}_supported_group",
                "corr_with_exact_rate": np.nan,
                "supported_count": len(supported),
                "unsupported_count": len(unsupported),
                "supported_exact_rate_mean": safe_mean(supported["strict_combo_exact_rate_mean"].tolist()),
                "unsupported_exact_rate_mean": safe_mean(unsupported["strict_combo_exact_rate_mean"].tolist()),
            }
        )

    quartile_labels = ["Q1", "Q2", "Q3", "Q4"]
    qcut_source = summary_df["best_replaceability_score"].rank(method="first")
    replaceability_bucket = pd.qcut(qcut_source, 4, labels=quartile_labels)
    for bucket in quartile_labels:
        subset = summary_df[replaceability_bucket == bucket]
        relation_rows.append(
            {
                "feature": f"replaceability_{bucket}",
                "corr_with_exact_rate": np.nan,
                "supported_count": len(subset),
                "supported_exact_rate_mean": safe_mean(subset["strict_combo_exact_rate_mean"].tolist()),
                "supported_exact_any_mean": safe_mean(subset["strict_combo_exact_any_fraction"].tolist()),
            }
        )

    stats = {
        "theoretical_generalizable_ratio": float((summary_df["theoretically_generalizable"] > 0).mean()),
        "tree_support_ratio": float((summary_df["any_tree_support"] > 0).mean()),
        "graph_support_ratio": float((summary_df["any_graph_support"] > 0).mean()),
        "common_neighbor_ratio": float((summary_df["link_common_neighbors"] > 0).mean()),
        "sibling_cross_edge_ratio": float((summary_df["sibling_cross_edge_count"] > 0).mean()),
    }
    return pd.DataFrame(relation_rows), stats


def build_go_bias_summary(summary_df, training_stats):
    if summary_df.empty:
        return pd.DataFrame()

    rows = []
    go_freq = training_stats["go_freq"]
    partner_degree = training_stats["partner_degree"]
    all_unique_pairs = summary_df["pair_unordered"].nunique()

    for go_id in sorted(set(summary_df["anchor_go"]) | set(summary_df["unseen_partner_go"])):
        as_anchor = summary_df[summary_df["anchor_go"] == go_id]
        as_partner = summary_df[summary_df["unseen_partner_go"] == go_id]
        involved = summary_df[(summary_df["anchor_go"] == go_id) | (summary_df["unseen_partner_go"] == go_id)]
        hard = involved[involved["strict_combo_exact_rate_mean"] <= 0.0]
        easy = involved[involved["strict_combo_exact_rate_mean"] > 0.0]
        rows.append(
            {
                "go_id": go_id,
                "train_freq": go_freq[go_id],
                "partner_degree": partner_degree.get(go_id, 0),
                "as_anchor_count": len(as_anchor),
                "as_unseen_partner_count": len(as_partner),
                "total_involved_count": len(involved),
                "unique_pair_coverage_ratio": involved["pair_unordered"].nunique() / max(1, all_unique_pairs),
                "mean_exact_rate_when_involved": safe_mean(involved["strict_combo_exact_rate_mean"].tolist()),
                "hard_count": len(hard),
                "hard_ratio": len(hard) / max(1, len(involved)),
                "easy_count": len(easy),
                "theoretical_generalizable_ratio": safe_mean(involved["theoretically_generalizable"].tolist()),
                "mean_replaceability_score": safe_mean(involved["best_replaceability_score"].tolist()),
                "mean_common_neighbors": safe_mean(involved["link_common_neighbors"].tolist()),
                "mean_full_test_hit_rate_when_involved": safe_mean(involved["mean_full_test_hit_rate"].tolist()),
                "mean_single_label_hit_rate_when_involved": safe_mean(involved["mean_single_label_hit_rate"].tolist()),
            }
        )

    go_bias_df = pd.DataFrame(rows).sort_values(
        ["hard_count", "hard_ratio", "total_involved_count", "train_freq"], ascending=[False, False, False, False]
    )
    return go_bias_df


def build_combo_bias_summary(combo_to_entries, combo_exact_rate, training_stats, go_baseline_lookup):
    combo_freq = training_stats["combo_freq"]
    rows = []
    for combo, entries in combo_to_entries.items():
        size = len(combo)
        go_freqs = [training_stats["go_freq"][go_id] for go_id in combo]
        full_rates = [go_baseline_lookup.get(go_id, {}).get("full_test_presence_hit_rate", np.nan) for go_id in combo]
        single_rates = [go_baseline_lookup.get(go_id, {}).get("single_label_hit_rate", np.nan) for go_id in combo]
        pair_support = []
        unseen_pair_count = 0
        for pair in combinations(combo, 2):
            freq = training_stats["pair_freq"][tuple(sorted(pair))]
            pair_support.append(freq)
            if freq == 0:
                unseen_pair_count += 1
        observed_rate = combo_exact_rate.get(combo, 0.0)
        expected_full = safe_min(full_rates)
        expected_single = safe_min(single_rates)
        rows.append(
            {
                "combo": str(combo),
                "combo_size": size,
                "strict_test_count": len(entries),
                "train_combo_freq": combo_freq.get(combo, 0),
                "mean_go_freq": safe_mean(go_freqs),
                "min_go_freq": min(go_freqs) if go_freqs else np.nan,
                "mean_pair_support": safe_mean(pair_support),
                "unseen_pair_count": unseen_pair_count,
                "pair_support_ratio": (len(pair_support) - unseen_pair_count) / max(1, len(pair_support)),
                "mean_full_test_hit_rate": safe_mean(full_rates),
                "mean_single_label_hit_rate": safe_mean(single_rates),
                "expected_combo_rate_fulltest": expected_full,
                "expected_combo_rate_single": expected_single,
                "strict_combo_exact_rate_mean": observed_rate,
                "residual_vs_fulltest_combo_expectation": observed_rate - expected_full if pd.notna(expected_full) else np.nan,
                "residual_vs_single_combo_expectation": observed_rate - expected_single if pd.notna(expected_single) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strict_combo_exact_rate_mean", "combo_size", "unseen_pair_count"], ascending=[True, False, False]
    )


def build_unseen_pairs(combo_to_entries, training_stats):
    pair_freq = training_stats["pair_freq"]
    unseen_pairs = set()
    for combo in combo_to_entries:
        for pair in combinations(combo, 2):
            ordered_pair = tuple(sorted(pair))
            if pair_freq[ordered_pair] == 0:
                unseen_pairs.add(ordered_pair)
    return unseen_pairs


def summarize_report(
    training_stats,
    combo_to_entries,
    unseen_pairs,
    summary_df,
    anchor_df,
    performance_df,
    performance_stats,
    go_bias_df,
    combo_bias_df,
):
    all_go_terms = len(training_stats["go_freq"])
    train_pair_count = len(training_stats["pair_freq"])

    lines = []
    lines.append("# 未见 GO Pair 边可替换性分析报告")
    lines.append("")
    lines.append("## 数据概览")
    lines.append(f"- 训练集 GO 原子数: `{all_go_terms}`")
    lines.append(f"- 训练集已知 GO pair 边数: `{train_pair_count}`")
    lines.append(f"- strict OOD unique 组合数: `{len(combo_to_entries)}`")
    lines.append(f"- strict OOD 中未见 pair 边数: `{len(unseen_pairs)}`")
    lines.append("")

    if not summary_df.empty:
        lines.append("## 全局统计")
        lines.append(
            f"- 每个锚点-未知边的平均已知邻居数: `{summary_df['anchor_known_neighbor_count'].mean():.2f}`"
        )
        lines.append(
            f"- 至少存在一个 sibling 支持的比例: `{(summary_df['anchor_sibling_support_count'] > 0).mean():.3f}`"
        )
        lines.append(
            f"- 至少存在一个 parent 支持的比例: `{(summary_df['anchor_parent_support_count'] > 0).mean():.3f}`"
        )
        lines.append(
            f"- 至少存在一个 child 支持的比例: `{(summary_df['anchor_child_support_count'] > 0).mean():.3f}`"
        )
        lines.append(
            f"- 平均最小语义距离 `min dist(e, N(a))`: `{summary_df['min_semantic_distance_to_known'].mean():.2f}`"
        )
        lines.append(
            f"- 至少存在 1 个共同邻居的比例: `{(summary_df['link_common_neighbors'] > 0).mean():.3f}`"
        )
        lines.append(
            f"- 平均 link-prediction Jaccard: `{summary_df['link_neighbor_jaccard'].mean():.3f}`"
        )
        lines.append(
            f"- sibling-set 之间至少存在一条训练边的比例: `{(summary_df['sibling_cross_edge_count'] > 0).mean():.3f}`"
        )
        lines.append(
            f"- 理论可泛化 pair 比例（只要任一树/图条件满足）: `{performance_stats.get('theoretical_generalizable_ratio', np.nan):.3f}`"
        )
        lines.append("")

        lines.append("## 共现支持与模型正确率关系")
        lines.append("")
        relation_cols = performance_df[
            performance_df["feature"].isin(
                [
                    "best_replaceability_score",
                    "link_common_neighbors",
                    "mean_full_test_hit_rate",
                    "expected_pair_rate_fulltest",
                    "residual_vs_fulltest_pair_expectation",
                    "anchor_known_neighbor_count",
                    "anchor_sibling_support_count",
                    "sibling_cross_edge_count",
                    "anchor_freq",
                    "unseen_partner_freq",
                    "anchor_partner_degree",
                    "unseen_partner_degree",
                ]
            )
        ]
        for row in relation_cols.itertuples():
            lines.append(f"- `{row.feature}` 与 `strict_combo_exact_rate_mean` 的相关系数: `{row.corr_with_exact_rate:.3f}`")
        lines.append("")
        for name in ["tree_support_supported_group", "graph_support_supported_group", "theoretical_generalizable_supported_group"]:
            subset = performance_df[performance_df["feature"] == name]
            if subset.empty:
                continue
            row = subset.iloc[0]
            lines.append(
                f"- `{name}`: support 组平均正确率=`{row['supported_exact_rate_mean']:.3f}`，"
                f"无 support 组=`{row['unsupported_exact_rate_mean']:.3f}`，"
                f"样本数=`{int(row['supported_count'])}/{int(row['unsupported_count'])}`"
            )
        lines.append("")

        lines.append("## 排除单 GO 难度影响后的观察")
        lines.append("")
        lines.append("- `mean_full_test_hit_rate` 表示组成该未见 pair 的两个 GO，在完整测试集上的平均边际命中率。")
        lines.append("- `expected_pair_rate_fulltest` 现在改为两个 GO 的完整测试集边际命中率的 `min`，表示“如果瓶颈只由更难的那个 GO 决定”时的 pair 期望成功率。")
        lines.append("- `residual_vs_fulltest_pair_expectation` 为 strict OOD 观察成功率减去这个期望，越负说明越像是组合泛化额外失败。")
        lines.append("")

        lines.append("## Top 20 最可替换的未见边（按 best replaceability score）")
        lines.append("")
        top_replaceable = summary_df.sort_values(
            ["best_replaceability_score", "link_common_neighbors", "anchor_sibling_support_count"],
            ascending=[False, False, False],
        ).head(20)
        for row in top_replaceable.itertuples():
            lines.append(
                f"- `({row.anchor_go}, {row.unseen_partner_go})`: "
                f"best substitute=`{row.best_known_partner_go}`，"
                f"relation=`{row.best_relation_type}`，"
                f"score=`{row.best_replaceability_score:.3f}`，"
                f"common-neighbors=`{int(row.link_common_neighbors)}`，"
                f"min-dist=`{row.min_semantic_distance_to_known:.1f}`"
            )
        lines.append("")

        lines.append("## Top 20 最难替换的未见边")
        lines.append("")
        hardest = summary_df.sort_values(
            ["best_replaceability_score", "link_common_neighbors", "min_semantic_distance_to_known"],
            ascending=[True, True, False],
        ).head(20)
        for row in hardest.itertuples():
            lines.append(
                f"- `({row.anchor_go}, {row.unseen_partner_go})`: "
                f"score=`{row.best_replaceability_score:.3f}`，"
                f"parent/sibling/child support=`{int(row.anchor_parent_support_count)}/{int(row.anchor_sibling_support_count)}/{int(row.anchor_child_support_count)}`，"
                f"common-neighbors=`{int(row.link_common_neighbors)}`，"
                f"min-dist=`{row.min_semantic_distance_to_known:.1f}`"
            )
        lines.append("")

    if not anchor_df.empty:
        lines.append("## Top 10 锚点 GO（未知目标最多）")
        lines.append("")
        for row in anchor_df.head(10).itertuples():
            lines.append(
                f"- `{row.anchor_go}`: known-neighbors=`{int(row.known_neighbor_count)}`，"
                f"unknown-targets=`{int(row.unknown_target_count)}`，"
                f"mean-best-score=`{row.mean_best_replaceability_score:.3f}`"
            )
        lines.append("")

    if not go_bias_df.empty:
        lines.append("## GO 标签偏好与困难样本")
        lines.append("")
        for row in go_bias_df.head(10).itertuples():
            lines.append(
                f"- `{row.go_id}`: train-freq=`{int(row.train_freq)}`，partner-degree=`{int(row.partner_degree)}`，"
                f"涉及未见边=`{int(row.total_involved_count)}`，hard-ratio=`{row.hard_ratio:.3f}`，"
                f"mean-exact-rate=`{row.mean_exact_rate_when_involved:.3f}`，"
                f"fulltest-hit=`{row.mean_full_test_hit_rate_when_involved:.3f}`"
            )
        lines.append("")

    if not combo_bias_df.empty:
        lines.append("## 困难组合偏好")
        lines.append("")
        for row in combo_bias_df.head(10).itertuples():
            lines.append(
                f"- `{row.combo}`: size=`{int(row.combo_size)}`，unseen-pairs=`{int(row.unseen_pair_count)}`，"
                f"mean-go-freq=`{row.mean_go_freq:.1f}`，pair-support-ratio=`{row.pair_support_ratio:.3f}`，"
                f"expected-fulltest=`{row.expected_combo_rate_fulltest:.3f}`，"
                f"exact-rate=`{row.strict_combo_exact_rate_mean:.3f}`，"
                f"residual=`{row.residual_vs_fulltest_combo_expectation:.3f}`"
            )
        lines.append("")

    lines.append("## 输出文件")
    lines.append("- `unseen_pair_ordered_summary.csv`: 每条有向未见边 `(a, e)` 的聚合统计")
    lines.append("- `unseen_pair_replacement_candidates.csv`: `(a, e)` 相对每个已知 `(a, b)` 的细粒度可替换性比较")
    lines.append("- `anchor_go_summary.csv`: 以锚点 GO 为中心的统计摘要")
    lines.append("- `performance_vs_cooccurrence.csv`: 共现/树支持特征与模型正确率关系")
    lines.append("- `go_bias_summary.csv`: GO 标签本身的频次、hub 程度与困难度偏好")
    lines.append("- `combo_bias_summary.csv`: strict OOD 组合层面的频次/共现偏好与困难度")
    lines.append("- `go_difficulty_baseline.csv`: 完整测试集上的 per-GO 命中率与单标签命中率")
    lines.append("- `full_test_protein_baseline.csv`: 完整测试集逐蛋白的 raw cover 基线摘要")
    lines.append("")
    lines.append("## 指标说明")
    lines.append("- `parent/sibling/child support`: 对锚点 `a` 而言，`e` 的父/兄弟/子节点里有多少个已经作为 `a` 的已知配对邻居出现过")
    lines.append("- `link_common_neighbors`: 训练共现 pair 图里，`a` 与 `e` 这条未见边两端的共同邻居数量")
    lines.append("- `sibling_cross_edge_count`: `a` 的 sibling 集合与 `e` 的 sibling 集合之间，训练集中已见 pair 边的数量")
    lines.append("- `theoretically_generalizable`: 只要树支持或图支持任一条件满足，就视为这条未见边理论上可由训练信号泛化")
    lines.append("- `replaceability score`: 综合树关系、pair 图邻域相似度和已知 `(a,b)` 边频率得到的启发式替换性分数")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析 strict OOD 中未见 GO pair 边在 GO 树和训练共现 pair 图中的可替换性。"
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--strict-test-path", type=Path, default=DEFAULT_STRICT_TEST_PATH)
    parser.add_argument("--go-mapping-path", type=Path, default=DEFAULT_GO_MAPPING_PATH)
    parser.add_argument("--go-obo-path", type=Path, default=DEFAULT_GO_OBO_PATH)
    parser.add_argument("--pred-tsv-path", type=Path, default=DEFAULT_PRED_TSV_PATH)
    parser.add_argument("--full-test-pred-tsv-path", type=Path, default=DEFAULT_FULL_TEST_PRED_TSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    go_mapping = load_pickle(args.go_mapping_path)
    index_to_go = {v: k for k, v in go_mapping.items()}

    train_data = load_pickle(args.train_path)
    full_test_data = load_pickle(args.test_path)
    strict_test_data = load_pickle(args.strict_test_path)
    ontology = Ontology(str(args.go_obo_path), with_rels=True)

    training_stats = build_training_statistics(train_data, index_to_go, ontology)
    combo_to_entries = build_strict_unseen_unique_combos(strict_test_data, index_to_go)
    unseen_pairs = build_unseen_pairs(combo_to_entries, training_stats)
    go_baseline_df, full_test_protein_df, go_baseline_lookup = build_go_difficulty_baseline(
        full_test_data=full_test_data,
        index_to_go=index_to_go,
        full_test_pred_tsv_path=args.full_test_pred_tsv_path,
    )

    combo_exact_any, combo_exact_rate = compute_combo_performance(
        strict_test_data=strict_test_data,
        index_to_go=index_to_go,
        pred_tsv_path=args.pred_tsv_path,
    )

    detailed_df, summary_df, anchor_df = aggregate_unseen_pair_statistics(
        unseen_pairs=unseen_pairs,
        combo_to_entries=combo_to_entries,
        training_stats=training_stats,
        ontology=ontology,
        combo_exact_any=combo_exact_any,
        combo_exact_rate=combo_exact_rate,
        go_baseline_lookup=go_baseline_lookup,
    )
    performance_df, performance_stats = build_performance_relation_table(summary_df)
    go_bias_df = build_go_bias_summary(summary_df, training_stats)
    combo_bias_df = build_combo_bias_summary(combo_to_entries, combo_exact_rate, training_stats, go_baseline_lookup)

    report = summarize_report(
        training_stats=training_stats,
        combo_to_entries=combo_to_entries,
        unseen_pairs=unseen_pairs,
        summary_df=summary_df,
        anchor_df=anchor_df,
        performance_df=performance_df,
        performance_stats=performance_stats,
        go_bias_df=go_bias_df,
        combo_bias_df=combo_bias_df,
    )

    summary_path = args.output_dir / "unseen_pair_ordered_summary.csv"
    detailed_path = args.output_dir / "unseen_pair_replacement_candidates.csv"
    anchor_path = args.output_dir / "anchor_go_summary.csv"
    performance_path = args.output_dir / "performance_vs_cooccurrence.csv"
    go_bias_path = args.output_dir / "go_bias_summary.csv"
    combo_bias_path = args.output_dir / "combo_bias_summary.csv"
    go_baseline_path = args.output_dir / "go_difficulty_baseline.csv"
    fulltest_protein_path = args.output_dir / "full_test_protein_baseline.csv"
    report_path = args.output_dir / "REPORT.md"

    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)
    anchor_df.to_csv(anchor_path, index=False)
    performance_df.to_csv(performance_path, index=False)
    go_bias_df.to_csv(go_bias_path, index=False)
    combo_bias_df.to_csv(combo_bias_path, index=False)
    go_baseline_df.to_csv(go_baseline_path, index=False)
    full_test_protein_df.to_csv(fulltest_protein_path, index=False)
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved summary to: {summary_path}")
    print(f"Saved detailed replacement candidates to: {detailed_path}")
    print(f"Saved anchor summary to: {anchor_path}")
    print(f"Saved performance analysis to: {performance_path}")
    print(f"Saved GO bias summary to: {go_bias_path}")
    print(f"Saved combo bias summary to: {combo_bias_path}")
    print(f"Saved GO difficulty baseline to: {go_baseline_path}")
    print(f"Saved full test protein baseline to: {fulltest_protein_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
