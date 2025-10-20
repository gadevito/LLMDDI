import argparse
import pickle
from typing import List, Tuple, Dict, Any, Set
import csv
import sys

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def norm_drug(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()

def parse_label(y: Any) -> int:
    if y is None:
        raise ValueError("Found None label")
    s = str(y).strip().lower()
    if s in {"interaction", "interazione"}:
        return 1
    if s in {"no interaction", "no_interaction", "no-interaction", "nessuna interazione"}:
        return 0
    try:
        v = int(s)
        if v in (0, 1):
            return v
    except Exception:
        pass
    raise ValueError(f"Unrecognized label: {y}")

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

def metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"n": 0, "accuracy": 0.0, "precision": 0.0, "sensitivity": 0.0, "f1": 0.0,
                "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
    tn = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==0)
    fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
    fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
    acc = safe_div(tp+tn, len(y_true))
    prec = safe_div(tp, tp+fp)
    sens = safe_div(tp, tp+fn)
    f1 = safe_div(2*prec*sens, prec+sens) if (prec+sens)>0 else 0.0
    return {"n": len(y_true), "accuracy": acc, "precision": prec, "sensitivity": sens, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def build_train_sets(train_data: List[Tuple[Any, ...]]):
    train_drugs: Set[str] = set()
    train_pairs_ordered: Set[Tuple[str, str]] = set()
    train_pairs_unordered: Set[frozenset] = set()
    for row in train_data:
        if len(row) < 2:
            continue
        d1 = norm_drug(row[0])
        d2 = norm_drug(row[1])
        train_drugs.add(d1)
        train_drugs.add(d2)
        train_pairs_ordered.add((d1, d2))
        train_pairs_unordered.add(frozenset((d1, d2)))
    return train_drugs, train_pairs_ordered, train_pairs_unordered

def annotate_predictions(pred_data: List[Dict[str, Any]],
                         train_drugs: Set[str],
                         train_pairs_ord: Set[Tuple[str,str]],
                         train_pairs_unord: Set[frozenset]) -> List[Dict[str, Any]]:
    annotated = []
    for i, row in enumerate(pred_data):
        d1 = norm_drug(row.get("drug1"))
        d2 = norm_drug(row.get("drug2"))
        try:
            y = parse_label(row.get("target"))
            yhat = parse_label(row.get("new_target"))
        except ValueError as e:
            print(f"[WARN] Row {i}: {e}", file=sys.stderr)
            continue

        both_seen = (d1 in train_drugs) and (d2 in train_drugs)
        one_seen = ((d1 in train_drugs) ^ (d2 in train_drugs))
        none_seen = (d1 not in train_drugs) and (d2 not in train_drugs)

        ord_pair_seen = (d1, d2) in train_pairs_ord
        unord_pair_seen = frozenset((d1, d2)) in train_pairs_unord

        annotated.append({
            "drug1": d1,
            "drug2": d2,
            "target": y,
            "new_target": yhat,
            "both_seen_in_train": both_seen,
            "one_seen_in_train": one_seen,
            "none_seen_in_train": none_seen,
            "ordered_pair_in_train": ord_pair_seen,
            "unordered_pair_in_train": unord_pair_seen
        })
    return annotated

def summarize_groups(annotated: List[Dict[str, Any]]):
    # Diagnostic groups
    groups = {
        "ALL": lambda r: True,
        "NO_ENTITY_OVERLAP": lambda r: r["none_seen_in_train"],
        "ONE_ENTITY_OVERLAP": lambda r: r["one_seen_in_train"],
        "TWO_ENTITY_OVERLAP": lambda r: r["both_seen_in_train"],
        "NO_UNORDERED_PAIR_OVERLAP": lambda r: not r["unordered_pair_in_train"],
        "NO_ORDERED_PAIR_OVERLAP": lambda r: not r["ordered_pair_in_train"],
        "EXACT_ORDERED_PAIR_OVERLAP": lambda r: r["ordered_pair_in_train"],
        "EXACT_UNORDERED_PAIR_OVERLAP": lambda r: r["unordered_pair_in_train"],
        # Additional regimes to separate memorization vs generalization
        "BOTH_KNOWN_NEW_PAIR": lambda r: r["both_seen_in_train"] and not r["unordered_pair_in_train"],
        "UNORDERED_SEEN_NEW_DIRECTION": lambda r: r["unordered_pair_in_train"] and not r["ordered_pair_in_train"],
    }
    out = {}
    for name, pred in groups.items():
        subset = [r for r in annotated if pred(r)]
        y_true = [r["target"] for r in subset]
        y_pred = [r["new_target"] for r in subset]
        out[name] = metrics(y_true, y_pred)
    return out

def fmt_pct(x: int, total: int) -> str:
    return f"{(100.0 * x / total):.1f}%" if total else "0.0%"

def build_interpretation(counts: Dict[str, int], gm: Dict[str, Dict[str, float]]) -> str:
    lines = []
    total = counts.get("TOTAL_PRED_ROWS", 0)

    def m(group, key="f1"):
        return gm.get(group, {}).get(key, 0.0)

    # Prevalence of overlaps
    n_ent0 = counts.get("NO_ENTITY_OVERLAP", 0)
    n_ent1 = counts.get("ONE_ENTITY_OVERLAP", 0)
    n_ent2 = counts.get("TWO_ENTITY_OVERLAP", 0)
    n_ord = counts.get("EXACT_ORDERED_PAIR_OVERLAP", 0)
    n_unord = counts.get("EXACT_UNORDERED_PAIR_OVERLAP", 0)

    lines.append("Interpretation")
    lines.append("- This section interprets the overlap and subgroup metrics to assess potential leakage and true generalization.")

    # 1) Do predictions include pairs also present in training?
    if n_ord > 0 or n_unord > 0:
        lines.append(f"- Pair overlap detected: ordered={n_ord} ({fmt_pct(n_ord,total)}), unordered={n_unord} ({fmt_pct(n_unord,total)}).")
        if n_ord > 0:
            lines.append("  The validation/test set contains exact ordered pairs seen in training; this can inflate performance via pair memorization.")
        if n_unord > 0 and n_ord == 0:
            lines.append("  Unordered pair overlap without ordered overlap indicates exposure to the same pair but potentially different direction; still a form of leakage for directional tasks.")
    else:
        lines.append("- No pair overlap detected (ordered or unordered). Pair-level leakage is unlikely.")

    # 2) Entity-level exposure
    lines.append(f"- Entity overlap distribution: none={n_ent0} ({fmt_pct(n_ent0,total)}), one={n_ent1} ({fmt_pct(n_ent1,total)}), both={n_ent2} ({fmt_pct(n_ent2,total)}).")
    if n_ent0 == total:
        lines.append("  All evaluation pairs are entity-disjoint from training (strongest leakage control).")
    elif n_ent2 / total > 0.5 if total else False:
        lines.append("  Most evaluation pairs contain two drugs seen in training; performance may rely on entity familiarity rather than compositional generalization.")

    # 3) Performance comparisons to diagnose memorization vs generalization
    f1_all = m("ALL")
    f1_no_ent = m("NO_ENTITY_OVERLAP")
    f1_two_ent = m("TWO_ENTITY_OVERLAP")
    f1_no_unord = m("NO_UNORDERED_PAIR_OVERLAP")
    f1_ord = m("EXACT_ORDERED_PAIR_OVERLAP")
    f1_unord = m("EXACT_UNORDERED_PAIR_OVERLAP")
    f1_both_known_new_pair = m("BOTH_KNOWN_NEW_PAIR")
    f1_unord_seen_new_dir = m("UNORDERED_SEEN_NEW_DIRECTION")

    def cmp_line(a_name, a_val, b_name, b_val, label):
        diff = a_val - b_val
        sign = "higher" if diff >= 0 else "lower"
        return f"- {label}: {a_name} F1={a_val:.3f} vs {b_name} F1={b_val:.3f} ({abs(diff):.3f} {sign})."

    lines.append(cmp_line("NO_ENTITY_OVERLAP", f1_no_ent, "TWO_ENTITY_OVERLAP", f1_two_ent, "Entity generalization"))
    lines.append(cmp_line("NO_UNORDERED_PAIR_OVERLAP", f1_no_unord, "EXACT_UNORDERED_PAIR_OVERLAP", f1_unord, "Pair memorization (unordered)"))
    lines.append(cmp_line("NO_ORDERED_PAIR_OVERLAP", gm["NO_ORDERED_PAIR_OVERLAP"]["f1"], "EXACT_ORDERED_PAIR_OVERLAP", f1_ord, "Pair memorization (ordered)"))

    # Heuristic thresholds for flags
    LARGE_GAP = 0.10

    # Entity-level generalization
    if f1_two_ent - f1_no_ent > LARGE_GAP and n_ent0 > 0:
        lines.append("  Interpretation: Much higher F1 when both entities are known than when both are new suggests reliance on entity familiarity; consider an entity-disjoint evaluation.")
    elif f1_no_ent >= f1_all - 0.02 and n_ent0 > 0:
        lines.append("  Interpretation: F1 on completely new entities is close to overall F1; model shows true entity-level generalization.")

    # Pair-level memorization (unordered)
    if f1_unord - f1_no_unord > LARGE_GAP and n_unord > 0:
        lines.append("  Interpretation: Markedly higher F1 on unordered-pair overlaps indicates pair-level memorization; filter out any unordered-pair overlaps for a stricter test.")
    elif n_unord == 0:
        lines.append("  Interpretation: No unordered-pair overlaps; pair-level leakage (as a set) is unlikely.")

    # Directional effect
    if f1_ord - f1_unord > LARGE_GAP and n_ord > 0 and n_unord > 0:
        lines.append("  Interpretation: Exact ordered-pair overlap scores exceed unordered overlap by a large margin; strong evidence of direction-specific memorization.")
    if f1_unord_seen_new_dir > 0 and n_unord > 0 and counts.get("UNORDERED_SEEN_NEW_DIRECTION", 0) > 0:
        lines.append(f"- Directional generalization on previously seen unordered pairs (new direction): F1={f1_unord_seen_new_dir:.3f}.")
        if f1_unord_seen_new_dir < f1_unord - LARGE_GAP:
            lines.append("  Interpretation: Performance drops when the pair appears in the unseen direction; directionality remains challenging.")

    # Both-known but new pair
    if counts.get("BOTH_KNOWN_NEW_PAIR", 0) > 0:
        lines.append(f"- Both-known-but-new-pair F1={f1_both_known_new_pair:.3f}.")
        if f1_both_known_new_pair < f1_two_ent - LARGE_GAP:
            lines.append("  Interpretation: Even with both entities known, performance drops for unseen combinations, indicating limited compositional generalization.")
        else:
            lines.append("  Interpretation: Good compositional generalization: when both drugs are known but combined in a new pair, performance remains strong.")

    # Overall recommendation
    recs = []
    if n_ord > 0 or n_unord > 0:
        recs.append("re-run evaluation after removing any pairs (ordered and unordered) present in training")
    if n_ent0 == 0:
        recs.append("add an entity-disjoint evaluation split to quantify generalization to entirely new drugs")
    if f1_no_ent and f1_two_ent - f1_no_ent > LARGE_GAP:
        recs.append("report subgroup metrics (entity-disjoint vs entity-overlap) in the paper to transparently address memorization concerns")

    if recs:
        lines.append("Recommended actions:")
        for r in recs:
            lines.append(f"- {r}.")

    return "\n".join(lines)

def print_report(train_stats: Dict[str, Any],
                 counts: Dict[str, int],
                 group_metrics: Dict[str, Dict[str, float]],
                 interpretation: str,
                 save_txt: str = None):
    lines = []
    lines.append("=== Training/Prediction Overlap Analysis (DDI) ===")
    lines.append("")
    lines.append("Training summary:")
    lines.append(f"- Unique drugs in training: {train_stats['n_drugs']}")
    lines.append(f"- Ordered pairs in training: {train_stats['n_pairs_ordered']}")
    lines.append(f"- Unordered pairs in training: {train_stats['n_pairs_unordered']}")
    lines.append("")
    lines.append("Prediction overlap counts:")
    for k, v in counts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Group metrics (n, accuracy, precision, sensitivity, f1, tp, tn, fp, fn):")
    for g, m in group_metrics.items():
        lines.append(f"{g}: n={m['n']} acc={m['accuracy']:.3f} prec={m['precision']:.3f} sens={m['sensitivity']:.3f} f1={m['f1']:.3f} tp={m.get('tp',0)} tn={m.get('tn',0)} fp={m.get('fp',0)} fn={m.get('fn',0)}")
    lines.append("")
    lines.append(interpretation)

    report = "\n".join(lines)
    print(report)
    if save_txt:
        with open(save_txt, "w", encoding="utf-8") as f:
            f.write(report)

def main():
    ap = argparse.ArgumentParser(description="Analyze entity/pair overlap between training and DDI predictions. Computes subgroup metrics and prints interpretation.")
    ap.add_argument("--training", required=True, help="Training pickle: list of tuples; first two entries are drugs")
    ap.add_argument("--validation", required=True, help="Prediction pickle: list of dicts with drug1, drug2, target, new_target")
    ap.add_argument("--out_csv", default=None, help="Optional: CSV with row-level annotations")
    ap.add_argument("--output", default=None, help="Optional: TXT to save the aggregated report")
    ap.add_argument("--clean_csv", default=None, help="Optional: CSV of a 'clean' evaluation subset (no entity overlap and no unordered-pair overlap)")
    args = ap.parse_args()

    train_data = load_pickle(args.training)
    pred_data = load_pickle(args.validation)

    train_drugs, train_pairs_ord, train_pairs_unord = build_train_sets(train_data)
    train_stats = {
        "n_drugs": len(train_drugs),
        "n_pairs_ordered": len(train_pairs_ord),
        "n_pairs_unordered": len(train_pairs_unord),
    }

    annotated = annotate_predictions(pred_data, train_drugs, train_pairs_ord, train_pairs_unord)

    counts = {
        "TOTAL_PRED_ROWS": len(annotated),
        "NO_ENTITY_OVERLAP": sum(1 for r in annotated if r["none_seen_in_train"]),
        "ONE_ENTITY_OVERLAP": sum(1 for r in annotated if r["one_seen_in_train"]),
        "TWO_ENTITY_OVERLAP": sum(1 for r in annotated if r["both_seen_in_train"]),
        "EXACT_ORDERED_PAIR_OVERLAP": sum(1 for r in annotated if r["ordered_pair_in_train"]),
        "EXACT_UNORDERED_PAIR_OVERLAP": sum(1 for r in annotated if r["unordered_pair_in_train"]),
        "BOTH_KNOWN_NEW_PAIR": sum(1 for r in annotated if r["both_seen_in_train"] and not r["unordered_pair_in_train"]),
        "UNORDERED_SEEN_NEW_DIRECTION": sum(1 for r in annotated if r["unordered_pair_in_train"] and not r["ordered_pair_in_train"]),
    }

    group_metrics = summarize_groups(annotated)
    interpretation = build_interpretation(counts, group_metrics)

    # Optional row-level CSV
    if args.out_csv:
        fieldnames = ["drug1","drug2","target","new_target",
                      "both_seen_in_train","one_seen_in_train","none_seen_in_train",
                      "ordered_pair_in_train","unordered_pair_in_train"]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in annotated:
                w.writerow(r)

    # Optional 'clean' subset: no entity overlap and no unordered-pair overlap
    if args.clean_csv:
        clean_rows = [
            r for r in annotated
            if r["none_seen_in_train"] and (not r["unordered_pair_in_train"])
        ]
        fieldnames = ["drug1","drug2","target","new_target"]
        with open(args.clean_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in clean_rows:
                w.writerow({k: r[k] for k in fieldnames})

    print_report(train_stats, counts, group_metrics, interpretation, save_txt=args.output)

if __name__ == "__main__":
    main()

