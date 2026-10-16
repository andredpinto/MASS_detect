"""Runs every comparison of notebook 06 from the terminal, stage by stage, and saves the results that the notebook reads.

    Baseline  the diameter of the consensus region alone
    Stage 0   features: with or without the wavelet images
    Stage 1   ways of using the two datasets together
    Stage 2   models
    Stage 3   the binary scale: the probability of a rating of 4 or 5 given by the 1 to 5 model, or a model trained on
              the binary scale
    Stage 4   extensions of the chosen model: rare ratings weighted up, elastic net, average with the random forest
    Then      the final configuration on new seeds and its permutation importance; a new hospital (trained on one
              dataset, validated on the other); and the final configuration trained on the mean and on the mode of the
              ratings instead of the median

The label is always the median rating of the radiologists, rounded half up, and the predicted rating is the most
probable one. Each stage keeps its winner with a rule fixed in advance: an option replaces the simplest one only if the
95% interval of its difference to the simplest option lies above zero, and among such options the one with the best
primary metric (the mean QWK, or the mean AUC on the binary scale) is kept; otherwise the simplest one stays. Every configuration is saved as soon as it is done, so a stopped
run resumes where it stopped.

    OMP_NUM_THREADS=1 python -m src.mass_detect.scripts.run_models
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from .config import FEATURE_SETS_FILE, MODEL_FEATURES_FILE, MODEL_RESULTS_DIR, READING_FEATURES_FILE
from .modelling import (ENSEMBLES, EDGE_LIMIT, binary_summary, edge_share, external_validation, is_suspicious,
                        nested_cv, paired_auc_difference, paired_difference, permutation_importance_cv,
                        reading_agreement, summary)

SEEDS = (0, 1, 2)
CONFIRMATION_SEEDS = (10, 11, 12)
DATASETS = ("LNDb", "LIDC")


def load_tables(model_file=MODEL_FEATURES_FILE, reading_file=READING_FEATURES_FILE, sets_file=FEATURE_SETS_FILE):
    """The nodule table and the reading table of notebook 05, and the two sets of features (with and without wavelet)."""
    nodules = pd.read_csv(model_file)
    readings = pd.read_csv(reading_file)
    feature_sets = json.loads(sets_file.read_text())
    return nodules, readings, feature_sets


def _save(path, result, extra=()):
    keys = ["proba", "decoded", "suspicious_proba", "suspicious_pred", "classes", *extra]
    np.savez(path, **{key: result[key] for key in keys})


def run_config(key, frame, features, results_dir, processes, **options):
    """Runs one configuration with nested_cv and saves it, or loads it from disk when it was already done."""
    path = results_dir / f"{key}.npz"
    choices_path = results_dir / f"{key}_choices.csv"
    inner_path = results_dir / f"{key}_inner_scores.csv"

    if path.exists():
        saved = np.load(path, allow_pickle=True)
        result = {name: saved[name] for name in saved.files}
        result.update(choices=pd.read_csv(choices_path), frame=frame, name=options["name"])
        return result

    started = time.time()
    result = nested_cv(frame=frame, features=features, processes=processes, **options)
    _save(path, result)
    result["choices"].to_csv(choices_path, index=False)
    result["inner_scores"].to_csv(inner_path, index=False)
    print(f"{key}: {time.time() - started:.0f} s", flush=True)

    # The grid check: a value chosen at an end of the grid in too many outer folds means the grid is too narrow
    edges = edge_share(options["name"], result["choices"])
    if (edges > EDGE_LIMIT).any():
        print(f"  WARNING {key}: chosen at an end of the grid in {edges.round(2).to_dict()}, widen the grid", flush=True)
    result.update(frame=frame, name=options["name"])
    return result


def run_external(key, frame, features, source, results_dir, processes, name, harmonise=False):
    """Trains on one dataset and predicts the other (modelling.external_validation), saved like run_config."""
    path = results_dir / f"{key}.npz"
    choices_path = results_dir / f"{key}_choices.csv"
    if path.exists():
        saved = np.load(path, allow_pickle=True)
        result = {name_: saved[name_] for name_ in saved.files}
        result["choices"] = pd.read_csv(choices_path)
    else:
        started = time.time()
        result = external_validation(name, frame, features, source, harmonise=harmonise, processes=processes)
        _save(path, result, extra=("rows",))
        result["choices"].to_csv(choices_path, index=False)
        print(f"{key}: {time.time() - started:.0f} s", flush=True)
    result.update(frame=frame.reset_index(drop=True).iloc[result["rows"]].reset_index(drop=True), name=name)
    return result


def describe(key, stage, result, label="rating", reference=None, readings=None):
    """One row of the summary table.

    On the 1 to 5 scale: every metric per dataset, the mean QWK with its interval and its difference to a reference.
    On the binary scale: AUC, sensitivity, specificity and balanced accuracy per dataset, the mean AUC with its interval
    and its difference to the reference. With 'readings', also the agreement with every single radiologist.
    """
    frame = result["frame"]
    dataset = frame["dataset"].to_numpy()
    patients = frame["patient_id"].to_numpy()
    classes = result["classes"]
    row = {"config": key, "stage": stage}

    if len(classes) > 2:
        y = frame[label].to_numpy()
        table, primary = summary(y, result["decoded"], dataset, patients)
        row.update(primary)
        for d in table.index:
            for metric in table.columns:
                row[f"{metric}_{d}"] = table.loc[d, metric]
        if reference is not None and len(reference["classes"]) > 2:
            difference = paired_difference(y, reference["decoded"], result["decoded"], dataset, patients)
            row.update(difference=difference["difference"], difference_low=difference["low"],
                       difference_high=difference["high"])

    y_binary = is_suspicious(frame[label], classes)
    table, primary = binary_summary(y_binary, result["suspicious_proba"], result["suspicious_pred"], dataset, patients)
    row.update(primary)
    for d in table.index:
        for metric in table.columns:
            row[f"{metric}_{d}"] = table.loc[d, metric]
    if reference is not None:
        difference = paired_auc_difference(y_binary, reference["suspicious_proba"], result["suspicious_proba"], dataset,
                                           patients)
        row.update(auc_difference=difference["difference"], auc_difference_low=difference["low"],
                   auc_difference_high=difference["high"])

    if readings is not None:
        row.update(reading_agreement(frame, result["decoded"], result["suspicious_proba"], readings))
    if result.get("name") and "outer_fold" in result["choices"]:
        edges = edge_share(result["name"], result["choices"])
        row["share_at_grid_edge"] = float(edges.max()) if len(edges) else np.nan   # the grid check, see modelling.edge_share
    return row


def base_params_from(results):
    """{(seed, outer fold): {dataset: {model: hyperparameters}}}, from the choices of the models of an ensemble."""
    base = {}
    for model_name, result in results.items():
        for row in result["choices"].itertuples(index=False):
            for dataset, own in json.loads(row.params).items():
                base.setdefault((int(row.seed), int(row.outer_fold)), {}).setdefault(dataset, {})[model_name] = own
    return base


def choose(rows, keys, metric="mean_qwk", low="difference_low"):
    """The rule fixed in advance. 'keys' go from the simplest option to the most complex one.

    An option can replace the simplest one only if the 95% interval of its difference to it lies above zero; among
    those options the best one wins, and if there is none the simplest one stays.
    """
    simplest = keys[0]
    better = [key for key in keys[1:] if rows[key][low] > 0]
    if not better:
        return simplest
    return max(better, key=lambda key: rows[key][metric])


def run(processes=15, model_file=MODEL_FEATURES_FILE, reading_file=READING_FEATURES_FILE, results_dir=MODEL_RESULTS_DIR,
        models=("ordinal", "random_forest", "gradient_boosting", "svm"), seeds=SEEDS, confirmation_seeds=CONFIRMATION_SEEDS):
    results_dir.mkdir(parents=True, exist_ok=True)
    nodules, readings, feature_sets = load_tables(model_file, reading_file)
    common = dict(results_dir=results_dir, processes=processes)
    rows = {}      # one summary row per configuration
    chosen = {}    # the winner of each stage

    # Baseline: the diameter of the consensus region alone (Baltatzis et al., 2021)
    diameter = run_config("baseline_diameter", nodules, ["roi_diameter_mm"],
                          name="ordinal", strategy="pooled", seeds=seeds, **common)

    # Stage 0: features, the robust and non redundant ones of the original and LoG images, with or without the
    # robust and non redundant ones of the wavelet images (notebook 05)
    stage0 = {}
    for set_name in ("without_wavelet", "with_wavelet"):
        stage0[set_name] = run_config(f"features_{set_name}", nodules, feature_sets[set_name],
                                      name="ordinal", strategy="pooled", seeds=seeds, **common)
    rows["baseline_diameter"] = describe("baseline_diameter", "baseline", diameter, reference=stage0["without_wavelet"])
    for set_name, result in stage0.items():
        rows[f"features_{set_name}"] = describe(f"features_{set_name}", "0 features", result,
                                                reference=stage0["without_wavelet"])
    chosen["features"] = choose({k: rows[f"features_{k}"] for k in stage0}, list(stage0))
    features = feature_sets[chosen["features"]]

    # Stage 1: ways of using the datasets together, with each dataset alone as a reference
    for dataset in DATASETS:
        alone = nodules[nodules["dataset"] == dataset].reset_index(drop=True)
        key = f"pooling_{dataset}_only"
        rows[key] = describe(key, "1 pooling", run_config(key, alone, features, name="ordinal", strategy="pooled",
                                                          seeds=seeds, **common))
    stage1 = {"pooled": stage0[chosen["features"]]}
    for strategy in ("indicator", "combat", "per_dataset"):
        stage1[strategy] = run_config(f"pooling_{strategy}", nodules, features,
                                      name="ordinal", strategy=strategy, seeds=seeds, **common)
    for strategy, result in stage1.items():
        rows[f"pooling_{strategy}"] = describe(f"pooling_{strategy}", "1 pooling", result, reference=stage1["pooled"])
    chosen["strategy"] = choose({k: rows[f"pooling_{k}"] for k in stage1}, list(stage1))

    # Stage 2: models, with the chosen way of pooling
    stage2 = {"ordinal": stage1[chosen["strategy"]]}
    for model in models[1:]:
        stage2[model] = run_config(f"model_{model}", nodules, features,
                                   name=model, strategy=chosen["strategy"], seeds=seeds, **common)
    for model, result in stage2.items():
        rows[f"model_{model}"] = describe(f"model_{model}", "2 model", result, reference=stage2["ordinal"])
    chosen["model"] = choose({k: rows[f"model_{k}"] for k in stage2}, list(stage2))
    best = dict(name=chosen["model"], strategy=chosen["strategy"])
    final, final_result = dict(best), stage2[chosen["model"]]

    # Stage 3: the binary scale. The 1 to 5 model already gives the probability of a rating of 4 or 5 ("from 1 to 5");
    # a model trained directly on the binary scale is the alternative. The primary metric is the mean AUC.
    stage3 = {"from_1_to_5": final_result,
              "direct": run_config(f"binary_direct_{chosen['model']}_{chosen['strategy']}", nodules, features,
                                   label="suspicious", seeds=seeds, **best, **common)}
    rows["binary_from_1_to_5"] = describe("binary_from_1_to_5", "3 binary", stage3["from_1_to_5"])
    rows["binary_direct"] = describe("binary_direct", "3 binary", stage3["direct"], label="suspicious",
                                     reference=stage3["from_1_to_5"])
    rows["binary_from_1_to_5"].update(auc_difference=0.0, auc_difference_low=0.0, auc_difference_high=0.0)
    chosen["binary"] = choose({k: rows[f"binary_{k}"] for k in stage3}, list(stage3), metric="mean_auc",
                              low="auc_difference_low")

    # Stage 4: extensions of the chosen model. Each one is compared with the chosen configuration by the same rule as
    # every stage, the chosen configuration being the simplest option.
    chosen["final_model"] = chosen["model"]
    stage4 = {"chosen": final_result}
    on_strategy = dict(strategy=chosen["strategy"], seeds=seeds, **common)
    stage4["ordinal_balanced"] = run_config("extension_ordinal_balanced", nodules, features, name="ordinal_balanced",
                                            **on_strategy)
    stage4["elastic_net"] = run_config("extension_elastic_net", nodules, features, name="elastic_net", **on_strategy)
    if chosen["strategy"] == "per_dataset" and "random_forest" in stage2:
        # the ensemble reuses the hyperparameters that the ordinal model and the forest chose in each outer fold
        base = base_params_from({"ordinal": stage2["ordinal"], "random_forest": stage2["random_forest"]})
        stage4["ordinal_rf"] = run_config("extension_ordinal_rf", nodules, features, name="ordinal_rf",
                                          base_params=base, **on_strategy)
    for key, result in stage4.items():
        rows[f"extension_{key}"] = describe(f"extension_{key}", "4 extension", result, reference=stage4["chosen"])
    winner = choose({k: rows[f"extension_{k}"] for k in stage4}, list(stage4))
    if winner != "chosen":
        chosen["final_model"] = winner
        final = dict(name=winner, strategy=chosen["strategy"])
        final_result = stage4[winner]
    final_tag = f"{chosen['final_model']}_{chosen['strategy']}_{chosen['features']}"

    # The final configuration on new seeds. An ensemble first needs the choices of its models on the new seeds.
    extra = {}
    if final["name"] in ENSEMBLES:
        bases = {m: run_config(f"confirmation_base_{m}_{chosen['strategy']}", nodules, features, name=m,
                               strategy=chosen["strategy"], seeds=confirmation_seeds, **common)
                 for m in ENSEMBLES[final["name"]]}
        extra["base_params"] = base_params_from(bases)
    confirmation = run_config(f"confirmation_{final_tag}", nodules, features, seeds=confirmation_seeds, **final,
                              **extra, **common)
    rows["confirmation"] = describe("confirmation", "confirmation", confirmation, readings=readings)
    if chosen["binary"] == "direct":
        direct = run_config(f"confirmation_binary_direct_{chosen['model']}_{chosen['strategy']}", nodules, features,
                            label="suspicious", seeds=confirmation_seeds, **best, **common)
        rows["confirmation_binary"] = describe("confirmation_binary", "confirmation", direct, label="suspicious")

    importance_file = results_dir / f"permutation_importance_{final_tag}.csv"
    if not importance_file.exists():
        importance = permutation_importance_cv(final["name"], final["strategy"], nodules, features,
                                               final_result["choices"], training=final.get("training"),
                                               weight=final.get("weight"))
        importance.to_csv(importance_file)

    # A new hospital: trained on every nodule of one dataset, validated on the other. The model is the final one (or,
    # for an ensemble, the model of stage 2, because an ensemble needs choices from outer folds that do not exist here),
    # with three sets of features: the diameter alone, the features without and with the wavelet images. Each is
    # compared, on the same nodules, with the same model in cross validation inside the dataset that is predicted.
    external_model = final["name"] if final["name"] not in ENSEMBLES else chosen["model"]
    sets = {"diameter": ["roi_diameter_mm"], **feature_sets}
    for source in DATASETS:
        target = "LIDC" if source == "LNDb" else "LNDb"
        alone = nodules[nodules["dataset"] == target].reset_index(drop=True)
        for set_name, set_features in sets.items():
            within = run_config(f"within_{target}_{set_name}_{external_model}", alone, set_features,
                                name=external_model, strategy="pooled", seeds=seeds, **common)
            rows[f"within_{target}_{set_name}"] = describe(f"within_{target}_{set_name}", "new hospital", within)
            external = run_external(f"external_{source}_to_{target}_{set_name}_{external_model}", nodules,
                                    set_features, source, name=external_model, **common)
            rows[f"external_{source}_to_{target}_{set_name}"] = describe(
                f"external_{source}_to_{target}_{set_name}", "new hospital", external, reference=within)
        harmonised = run_external(f"external_{source}_to_{target}_{chosen['features']}_combat_{external_model}", nodules,
                                  features, source, name=external_model, harmonise=True, **common)
        rows[f"external_{source}_to_{target}_combat"] = describe(
            f"external_{source}_to_{target}_combat", "new hospital", harmonised,
            reference=run_external(f"external_{source}_to_{target}_{chosen['features']}_{external_model}", nodules,
                                   features, source, name=external_model, **common))

    # The final configuration trained on the mean and on the mode of the ratings instead of the median. The evaluation
    # still uses the median, and the agreement with every single radiologist does not depend on any of the three rules.
    for rule in ("mean", "mode"):
        training = nodules.assign(rating=nodules[f"rating_{rule}"])
        result = run_config(f"label_{rule}_{final_tag}", nodules, features, seeds=seeds, training=training, **final,
                            **extra_for_seeds(final, stage2), **common)
        rows[f"label_{rule}"] = describe(f"label_{rule}", "label check", result, reference=final_result,
                                         readings=readings)
    rows["label_median"] = describe("label_median", "label check", final_result, readings=readings)

    table = pd.DataFrame(rows.values())
    table.to_csv(results_dir / "summary.csv", index=False)
    (results_dir / "chosen.json").write_text(json.dumps(chosen, indent=1))
    print("chosen:", chosen, flush=True)
    return table, chosen


def extra_for_seeds(final, stage2):
    """The choices that an ensemble reuses on the seeds used for choosing (empty for any other model)."""
    if final["name"] not in ENSEMBLES:
        return {}
    return {"base_params": base_params_from({m: stage2[m] for m in ENSEMBLES[final["name"]]})}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # 15 processes: one per outer fold (5 folds x 3 seeds), so every configuration runs in a single round
    parser.add_argument("--processes", type=int, default=15)
    args = parser.parse_args()
    run(args.processes)


if __name__ == "__main__":
    main()
