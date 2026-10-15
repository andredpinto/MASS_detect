"""Runs every comparison of notebook 06 from the terminal, stage by stage, and saves the results that the notebook reads.

    Stage 0  decision rule
    Stage 1  ways of using the two datasets together
    Stage 2  models
    Stage 3  training label
    Stage 4  the 1 to 3 scale
    Stage 5  confirmation on new seeds, and permutation importance

Each stage keeps its winner with a rule fixed in advance: the option with the best mean QWK, unless the 95% interval of
its difference to the simplest option includes zero, in which case the simplest one is kept. Every configuration is saved
as soon as it is done, so a stopped run resumes where it stopped.

    uv run python -m src.mass_detect.scripts.run_models
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from .config import MODEL_FEATURES_FILE, MODEL_RESULTS_DIR, READING_FEATURES_FILE
from .modelling import EDGE_LIMIT, RULES, edge_share, nested_cv, paired_difference, permutation_importance_cv, summary

SEEDS = (0, 1, 2)
CONFIRMATION_SEEDS = (10, 11, 12)

# Columns of the tables of notebook 05 that are not model inputs
NOT_FEATURES = {
    "patch_id", "nodule_uid", "dataset", "patient_id", "scan_id", "rating", "rating_3", "mal_mean", "n_rated",
    "manufacturer", "kernel", "slice_thickness_mm", "pixel_spacing_mm", "roi_diameter_mm", "roi_axial_diameter_mm",
    "reader", "reader_rating", "weight",
}


def load_tables(model_file=MODEL_FEATURES_FILE, reading_file=READING_FEATURES_FILE):
    """The nodule table and the reading table of notebook 05, and the list of features."""
    nodules = pd.read_csv(model_file)
    readings = pd.read_csv(reading_file)
    features = [column for column in nodules.columns if column not in NOT_FEATURES]
    return nodules, readings, features


def run_config(key, frame, features, results_dir, processes, **options):
    """Runs one configuration with nested_cv and saves it, or loads it from disk when it was already done."""
    path = results_dir / f"{key}.npz"
    choices_path = results_dir / f"{key}_choices.csv"
    inner_path = results_dir / f"{key}_inner_scores.csv"

    if path.exists():
        saved = np.load(path, allow_pickle=True)
        return {"proba": saved["proba"], "decoded": saved["decoded"], "classes": saved["classes"],
                "choices": pd.read_csv(choices_path), "frame": frame, "name": options["name"]}

    started = time.time()
    result = nested_cv(frame=frame, features=features, processes=processes, **options)
    np.savez(path, proba=result["proba"], decoded=result["decoded"], classes=result["classes"])
    result["choices"].to_csv(choices_path, index=False)
    result["inner_scores"].to_csv(inner_path, index=False)
    print(f"{key}: {time.time() - started:.0f} s", flush=True)

    # The grid check: a value chosen at an end of the grid in too many outer folds means the grid is too narrow
    edges = edge_share(options["name"], result["choices"])
    if (edges > EDGE_LIMIT).any():
        print(f"  WARNING {key}: chosen at an end of the grid in {edges.round(2).to_dict()}, widen the grid", flush=True)
    result["frame"] = frame
    result["name"] = options["name"]
    return result


def describe(key, stage, result, label="rating", reference=None):
    """One row of the summary table: metrics per dataset, mean QWK with its interval, and the difference to a reference."""
    frame = result["frame"]
    y = frame[label].to_numpy()
    dataset = frame["dataset"].to_numpy()
    patients = frame["patient_id"].to_numpy()

    table, primary = summary(y, result["decoded"], dataset, patients)
    row = {"config": key, "stage": stage, "mean_qwk": primary["mean_qwk"], "low": primary["low"], "high": primary["high"],
           "rules_chosen": result["choices"]["rule"].value_counts().to_dict()}
    edges = edge_share(result["name"], result["choices"])
    row["share_at_grid_edge"] = float(edges.max()) if len(edges) else np.nan   # the grid check, see modelling.edge_share
    for d in table.index:
        for metric in table.columns:
            row[f"{metric}_{d}"] = table.loc[d, metric]

    if reference is not None:
        difference = paired_difference(y, reference["decoded"], result["decoded"], dataset, patients)
        row["difference"] = difference["difference"]
        row["difference_low"] = difference["low"]
        row["difference_high"] = difference["high"]
    return row


def choose(rows, keys):
    """The rule fixed in advance. 'keys' go from the simplest option to the most complex one."""
    simplest = keys[0]
    best = max(keys, key=lambda key: rows[key]["mean_qwk"])
    if best == simplest or rows[best]["difference_low"] <= 0:
        return simplest
    return best


def run(processes=15, model_file=MODEL_FEATURES_FILE, reading_file=READING_FEATURES_FILE, results_dir=MODEL_RESULTS_DIR,
        models=("ordinal", "random_forest", "gradient_boosting", "svm")):
    results_dir.mkdir(parents=True, exist_ok=True)
    nodules, readings, features = load_tables(model_file, reading_file)
    common = dict(results_dir=results_dir, processes=processes)
    rows = {}      # one summary row per configuration
    chosen = {}    # the winner of each stage

    # Baseline: the diameter of the consensus region alone (Baltatzis et al., 2021)
    diameter = run_config("baseline_diameter", nodules, ["roi_diameter_mm"],
                          name="ordinal", strategy="pooled", seeds=SEEDS, **common)

    # Stage 0: the decision rule, with the ordinal model on both datasets pooled
    stage0 = {}
    for rule in RULES + ("tuned",):
        rules = RULES if rule == "tuned" else (rule,)
        stage0[f"rule_{rule}"] = run_config(f"rule_{rule}", nodules, features,
                                            name="ordinal", strategy="pooled", seeds=SEEDS, rules=rules, **common)
    rows["baseline_diameter"] = describe("baseline_diameter", "baseline", diameter, reference=stage0["rule_tuned"])
    for key, result in stage0.items():
        rows[key] = describe(key, "0 decision rule", result, reference=stage0["rule_tuned"])

    # Stage 1: ways of using the datasets together (from here on the inner folds choose the rule)
    for dataset in ("LNDb", "LIDC"):
        alone = nodules[nodules["dataset"] == dataset].reset_index(drop=True)
        key = f"pooling_{dataset}_only"
        result = run_config(key, alone, features, name="ordinal", strategy="pooled", seeds=SEEDS, **common)
        rows[key] = describe(key, "1 pooling", result)

    stage1 = {"pooled": stage0["rule_tuned"]}
    for strategy in ("indicator", "combat", "per_dataset"):
        stage1[strategy] = run_config(f"pooling_{strategy}", nodules, features,
                                      name="ordinal", strategy=strategy, seeds=SEEDS, **common)
    for strategy, result in stage1.items():
        rows[f"pooling_{strategy}"] = describe(f"pooling_{strategy}", "1 pooling", result, reference=stage1["pooled"])
    chosen["strategy"] = choose({k: rows[f"pooling_{k}"] for k in stage1}, list(stage1))

    # Stage 2: models, with the chosen way of pooling
    stage2 = {"ordinal": stage1[chosen["strategy"]]}
    for model in models[1:]:
        stage2[model] = run_config(f"model_{model}", nodules, features,
                                   name=model, strategy=chosen["strategy"], seeds=SEEDS, **common)
    for model, result in stage2.items():
        rows[f"model_{model}"] = describe(f"model_{model}", "2 model", result, reference=stage2["ordinal"])
    chosen["model"] = choose({k: rows[f"model_{k}"] for k in stage2}, list(stage2))

    # Stage 3: training label. A is the mean rating of each nodule; B is every reading with a weight of 1/n.
    # The evaluation always uses A.
    best = dict(name=chosen["model"], strategy=chosen["strategy"])
    key_b = f"label_B_{chosen['model']}_{chosen['strategy']}"
    stage3 = {
        "A": stage2[chosen["model"]],
        "B": run_config(key_b, nodules, features, seeds=SEEDS, training=readings, weight="weight", **best, **common),
    }
    for label, result in stage3.items():
        rows[f"label_{label}"] = describe(f"label_{label}", "3 label", result, reference=stage3["A"])
    chosen["label"] = choose({k: rows[f"label_{k}"] for k in stage3}, list(stage3))

    final = dict(best)
    if chosen["label"] == "B":
        final.update(training=readings, weight="weight")
    final_result = stage3[chosen["label"]]

    # The file names of stages 4 and 5 carry the chosen configuration, so a new winner is computed again
    tag = f"{chosen['model']}_{chosen['strategy']}_{chosen['label']}"

    # Stage 4: the 1 to 3 scale, by grouping the 1 to 5 predictions (1-2, 3, 4-5) or by training on it directly
    collapsed = dict(final_result)
    collapsed["decoded"] = np.digitize(final_result["decoded"], [2.5, 3.5]) + 1
    rows["scale3_collapsed"] = describe("scale3_collapsed", "4 scale 1 to 3", collapsed, label="rating_3")
    if chosen["label"] == "A":
        direct = run_config(f"scale3_direct_{tag}", nodules, features, label="rating_3", seeds=SEEDS, **best, **common)
        rows["scale3_direct"] = describe("scale3_direct", "4 scale 1 to 3", direct, label="rating_3", reference=collapsed)

    # Stage 5: the chosen configuration on new seeds, then permutation importance
    confirmation = run_config(f"confirmation_{tag}", nodules, features, seeds=CONFIRMATION_SEEDS, **final, **common)
    rows["confirmation"] = describe("confirmation", "5 confirmation", confirmation)

    importance_file = results_dir / f"permutation_importance_{tag}.csv"
    if not importance_file.exists():
        importance = permutation_importance_cv(final["name"], final["strategy"], nodules, features,
                                               final_result["choices"], training=final.get("training"),
                                               weight=final.get("weight"))
        importance.to_csv(importance_file)

    table = pd.DataFrame(rows.values())
    table.to_csv(results_dir / "summary.csv", index=False)
    (results_dir / "chosen.json").write_text(json.dumps(chosen, indent=1))
    print("chosen:", chosen, flush=True)
    return table, chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # 15 processes: one per outer fold (5 folds x 3 seeds), so every configuration runs in a single round
    parser.add_argument("--processes", type=int, default=15)
    parser.add_argument("--skip-svm", action="store_true", help="leave the slowest model out")
    args = parser.parse_args()

    models = ("ordinal", "random_forest", "gradient_boosting")
    if not args.skip_svm:
        models += ("svm",)
    run(args.processes, models=models)


if __name__ == "__main__":
    main()
