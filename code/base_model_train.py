import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

from pathlib import Path
import gc
import shutil
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from utils import (
    MODELS,
    MODEL_SPECS,
    build_model,
    build_training_progress,
    evaluate_predictions,
    find_fold_result,
    fold_item_key,
    get_class_weight,
    load_csv_records,
    load_partial_metrics,
    load_paths_and_labels,
    make_dataset,
    persist_partial_metrics,
    plot_and_save_confusion_matrix,
    set_global_seeds,
    train_two_stage,
    upsert_fold_result,
    upsert_model_metric,
    write_json,
)

# =========================
# Config
# =========================
DATA_DIR = Path("../data/public_dataset/public_data")

HEAD_EPOCHS = 100
FINETUNE_EPOCHS = 120
HEAD_LR = 1e-3
FINETUNE_LR = 1e-5
UNFREEZE_RATIO = 0.3

N_SPLITS = 5
N_REPEATS = 3
BATCH_SIZE = 16
BASE_SEED = 42
TEST_SIZE = 0.2
IMAGE_SIZES = [192, 224]
USE_AUGMENTATION = True
USE_MIXED_PRECISION = True

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
RESULTS_DIR = ARTIFACTS_DIR / "results"
FINAL_MODEL_DIR = ARTIFACTS_DIR / "final_model"
CONFUSION_MATRIX_DIR = ARTIFACTS_DIR / "confusionmatrix"
PROGRESS_STATE_PATH = RESULTS_DIR / "training_progress_state.json"
HOLDOUT_ALL_MODELS_PATH = RESULTS_DIR / "holdout_test_metrics_all_models.csv"

PERSISTENT_ARTIFACTS_DIR = Path(os.environ.get("PERSISTENT_ARTIFACTS_DIR", str(ARTIFACTS_DIR)))
PERSISTENT_RESULTS_DIR = PERSISTENT_ARTIFACTS_DIR / "results"
PERSISTENT_FINAL_MODEL_DIR = PERSISTENT_ARTIFACTS_DIR / "final_model"
PERSISTENT_CONFUSION_MATRIX_DIR = PERSISTENT_ARTIFACTS_DIR / "confusionmatrix"


def _sync_dir_to_persistent(src: Path, dst: Path):
    if src.resolve() == dst.resolve():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)


def _compute_fold_progress_state(fold_results):
    all_items = [
        fold_item_key(model_name, img_size, fold)
        for model_name in MODELS
        for img_size in IMAGE_SIZES
        for fold in range(1, (N_SPLITS * N_REPEATS) + 1)
    ]
    done_items = sorted(
        {
            fold_item_key(row["model"], int(row["img_size"]), int(row["fold"]))
            for row in fold_results
        }
    )
    done_set = set(done_items)
    pending_items = [item for item in all_items if item not in done_set]
    return all_items, done_items, pending_items


def _persist_progress_state(fold_results, current_item=None):
    all_items, done_items, pending_items = _compute_fold_progress_state(fold_results)
    state = build_training_progress(
        total_steps=len(all_items),
        done_items=done_items,
        pending_items=pending_items,
        current_item=current_item,
    )
    write_json(PROGRESS_STATE_PATH, state)
    print(f"Progress: {state['progress']} | pending: {state['pending_steps']}")


def _ensure_final_model_and_holdout(
    model_name,
    img_size,
    X_train,
    y_train,
    X_test,
    y_test,
    num_classes,
    class_weight,
    class_names=None,
    holdout_all=None,
):
    _, preprocess_fn = MODEL_SPECS[model_name]
    final_model_path = FINAL_MODEL_DIR / f"{model_name}_img{img_size}_final.keras"
    existing_holdout = next(
        (
            item for item in holdout_all
            if item["model"] == model_name and int(item["img_size"]) == int(img_size)
        ),
        None,
    )

    print(f"\n=== Final full-train: {model_name} | img_size={img_size} ===")

    if final_model_path.exists() and existing_holdout is not None:
        print("Final model and holdout metrics already exist. Skipping.")
        return holdout_all

    tf.keras.backend.clear_session()
    set_global_seeds(BASE_SEED)

    train_ds = make_dataset(
        X_train, y_train, img_size, preprocess_fn, BATCH_SIZE,
        training=True, use_augmentation=USE_AUGMENTATION, seed=BASE_SEED,
    )
    test_ds = make_dataset(
        X_test, y_test, img_size, preprocess_fn, BATCH_SIZE,
        training=False, seed=BASE_SEED,
    )

    if final_model_path.exists():
        print("Loading existing final model to compute missing holdout metrics.")
        final_model = tf.keras.models.load_model(final_model_path)
    else:
        final_model, final_base = build_model(model_name, num_classes, img_size, MODEL_SPECS)
        train_two_stage(
            model=final_model,
            base_model=final_base,
            train_ds=train_ds,
            class_weight=class_weight,
            head_epochs=HEAD_EPOCHS,
            finetune_epochs=FINETUNE_EPOCHS,
            head_lr=HEAD_LR,
            finetune_lr=FINETUNE_LR,
            unfreeze_ratio=UNFREEZE_RATIO,
            val_ds=None,
            best_checkpoint_path=None,
        )
        del final_base
        final_model.save(final_model_path)
        print(f"Saved final model: {final_model_path}")
        if PERSISTENT_FINAL_MODEL_DIR.resolve() != FINAL_MODEL_DIR.resolve():
            PERSISTENT_FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_model_path, PERSISTENT_FINAL_MODEL_DIR / final_model_path.name)

    test_metrics = evaluate_predictions(final_model, test_ds, y_test, class_names=class_names)
    print(
        "Holdout TEST -> "
        f"F1-macro: {test_metrics['f1_macro']:.4f} | "
        f"F1-weighted: {test_metrics['f1_weighted']:.4f} | "
        f"Acc: {test_metrics['accuracy']:.4f} | "
        f"Prec-macro: {test_metrics['precision_macro']:.4f} | "
        f"Recall-macro: {test_metrics['recall_macro']:.4f}"
    )
    if "f1_per_class" in test_metrics and class_names is not None:
        per_lines = [
            f"{cn}: F1={test_metrics['f1_per_class'][i]:.4f} | "
            f"P={test_metrics['precision_per_class'][i]:.4f} | "
            f"R={test_metrics['recall_per_class'][i]:.4f}"
            for i, cn in enumerate(class_names)
        ]
        print("  Holdout Per-class -> " + " | ".join(per_lines))
    if class_names is not None:
        y_prob = final_model.predict(test_ds, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        del y_prob
        gc.collect()
        backbone_name = f"{model_name}_img{img_size}"
        plot_and_save_confusion_matrix(y_test, y_pred, class_names, backbone_name, CONFUSION_MATRIX_DIR)
        print(f"Saved confusion matrix: {CONFUSION_MATRIX_DIR}/confusion_matrix_{backbone_name}.png")

    holdout_all = upsert_model_metric(
        holdout_all,
        {"model": model_name, "img_size": int(img_size), **test_metrics},
    )
    pd.DataFrame(holdout_all).to_csv(HOLDOUT_ALL_MODELS_PATH, index=False)
    _sync_dir_to_persistent(RESULTS_DIR, PERSISTENT_RESULTS_DIR)
    _sync_dir_to_persistent(CONFUSION_MATRIX_DIR, PERSISTENT_CONFUSION_MATRIX_DIR)
    del final_model, train_ds, test_ds
    gc.collect()
    tf.keras.backend.clear_session()
    return holdout_all


def main():
    sys.stdout.reconfigure(line_buffering=True)
    set_global_seeds(BASE_SEED)
    if USE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    X, y, class_names = load_paths_and_labels(DATA_DIR)
    num_classes = len(class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=BASE_SEED,
    )

    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    PERSISTENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PERSISTENT_FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    fold_results, results = load_partial_metrics(RESULTS_DIR)
    full_train_class_weight = get_class_weight(y_train, num_classes)
    holdout_all = load_csv_records(HOLDOUT_ALL_MODELS_PATH)
    _persist_progress_state(fold_results, current_item=None)

    for model_name in MODELS:
        _, preprocess_fn = MODEL_SPECS[model_name]
        for img_size in IMAGE_SIZES:
            print(f"\n===== {model_name} | img_size={img_size} =====")

            rkf = RepeatedStratifiedKFold(
                n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=BASE_SEED,
            )

            fold_scores = []
            for fold, (tr_idx, va_idx) in enumerate(rkf.split(X_train, y_train), start=1):
                fold_item = fold_item_key(model_name, img_size, fold)
                existing = find_fold_result(fold_results, model_name, img_size, fold)
                if existing is not None:
                    print(f"Skipping completed fold: {model_name}|img{img_size}|fold{fold:02d}")
                    fold_scores.append(
                        {
                            "f1_macro": float(existing["f1_macro"]),
                            "f1_weighted": float(existing["f1_weighted"]),
                            "accuracy": float(existing["accuracy"]),
                            "precision_macro": float(existing["precision_macro"]),
                            "recall_macro": float(existing["recall_macro"]),
                        }
                    )
                    _persist_progress_state(fold_results, current_item=fold_item)
                    continue

                tf.keras.backend.clear_session()
                set_global_seeds(BASE_SEED + fold)

                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train[va_idx], y_train[va_idx]

                train_ds = make_dataset(
                    X_tr, y_tr, img_size, preprocess_fn, BATCH_SIZE,
                    training=True, use_augmentation=USE_AUGMENTATION, seed=BASE_SEED + fold,
                )
                val_ds = make_dataset(
                    X_va, y_va, img_size, preprocess_fn, BATCH_SIZE,
                    training=False, seed=BASE_SEED + fold,
                )

                fold_class_weight = get_class_weight(y_tr, num_classes)
                model, base = build_model(model_name, num_classes, img_size, MODEL_SPECS)

                try:
                    train_two_stage(
                        model=model,
                        base_model=base,
                        train_ds=train_ds,
                        class_weight=fold_class_weight,
                        head_epochs=HEAD_EPOCHS,
                        finetune_epochs=FINETUNE_EPOCHS,
                        head_lr=HEAD_LR,
                        finetune_lr=FINETUNE_LR,
                        unfreeze_ratio=UNFREEZE_RATIO,
                        val_ds=val_ds,
                    )
                    fold_metrics = evaluate_predictions(model, val_ds, y_va, class_names=class_names)
                except tf.errors.ResourceExhaustedError as oom_err:
                    print(f"[OOM] {model_name}|img{img_size}|fold{fold:02d}: {oom_err}. Skipping fold.")
                    del model, base, train_ds, val_ds
                    gc.collect()
                    tf.keras.backend.clear_session()
                    continue

                fold_scores.append(fold_metrics)
                fold_results = upsert_fold_result(
                    fold_results,
                    {"model": model_name, "img_size": img_size, "fold": fold, **fold_metrics},
                )
                print(
                    f"Fold {fold:02d} -> "
                    f"F1-macro: {fold_metrics['f1_macro']:.4f} | "
                    f"F1-weighted: {fold_metrics['f1_weighted']:.4f} | "
                    f"Acc: {fold_metrics['accuracy']:.4f} | "
                    f"Prec-macro: {fold_metrics['precision_macro']:.4f} | "
                    f"Recall-macro: {fold_metrics['recall_macro']:.4f}"
                )
                if "f1_per_class" in fold_metrics and class_names is not None:
                    per_lines = [
                        f"{cn}: F1={fold_metrics['f1_per_class'][i]:.4f} | "
                        f"P={fold_metrics['precision_per_class'][i]:.4f} | "
                        f"R={fold_metrics['recall_per_class'][i]:.4f}"
                        for i, cn in enumerate(class_names)
                    ]
                    print("  Per-class -> " + " | ".join(per_lines))

                persist_partial_metrics(fold_results, results, RESULTS_DIR)
                _persist_progress_state(fold_results, current_item=fold_item)
                _sync_dir_to_persistent(RESULTS_DIR, PERSISTENT_RESULTS_DIR)
                del model, base, train_ds, val_ds
                gc.collect()

            f1_macros = [s["f1_macro"] for s in fold_scores]
            f1_weighted = [s["f1_weighted"] for s in fold_scores]
            accs = [s["accuracy"] for s in fold_scores]
            precision_macros = [s["precision_macro"] for s in fold_scores]
            recall_macros = [s["recall_macro"] for s in fold_scores]

            print(f"{model_name} (img={img_size}) | F1-macro mean±std: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
            print(f"{model_name} (img={img_size}) | F1-weighted mean±std: {np.mean(f1_weighted):.4f} ± {np.std(f1_weighted):.4f}")
            print(f"{model_name} (img={img_size}) | Acc mean±std: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            print(f"{model_name} (img={img_size}) | Prec-macro mean±std: {np.mean(precision_macros):.4f} ± {np.std(precision_macros):.4f}")
            print(f"{model_name} (img={img_size}) | Recall-macro mean±std: {np.mean(recall_macros):.4f} ± {np.std(recall_macros):.4f}")

            summary = {
                "model": model_name,
                "img_size": img_size,
                "f1_macro_mean": np.mean(f1_macros),
                "f1_macro_std": np.std(f1_macros),
                "f1_weighted_mean": np.mean(f1_weighted),
                "f1_weighted_std": np.std(f1_weighted),
                "acc_mean": np.mean(accs),
                "acc_std": np.std(accs),
                "precision_macro_mean": np.mean(precision_macros),
                "precision_macro_std": np.std(precision_macros),
                "recall_macro_mean": np.mean(recall_macros),
                "recall_macro_std": np.std(recall_macros),
                "runs": len(fold_scores),
            }
            results = [
                row for row in results
                if not (row["model"] == model_name and int(row["img_size"]) == img_size)
            ]
            results.append(summary)
            persist_partial_metrics(fold_results, results, RESULTS_DIR)
            _persist_progress_state(fold_results, current_item=None)
            holdout_all = _ensure_final_model_and_holdout(
                model_name=model_name,
                img_size=img_size,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                num_classes=num_classes,
                class_weight=full_train_class_weight,
                class_names=class_names,
                holdout_all=holdout_all,
            )

    df_folds = pd.DataFrame(fold_results)
    df_results = pd.DataFrame(results).sort_values("f1_macro_mean", ascending=False)
    print("\n=== Final ranking ===")
    print(df_results.to_string(index=False))
    df_folds.to_csv(RESULTS_DIR / "cv_fold_metrics.csv", index=False)
    df_results.to_csv(RESULTS_DIR / "cv_ranking.csv", index=False)

    for row in df_results.to_dict(orient="records"):
        holdout_all = _ensure_final_model_and_holdout(
            model_name=row["model"],
            img_size=int(row["img_size"]),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
            class_weight=full_train_class_weight,
            holdout_all=holdout_all,
        )

    if holdout_all:
        best_holdout = next(
            (
                item for item in holdout_all
                if item["model"] == df_results.iloc[0]["model"]
                and int(item["img_size"]) == int(df_results.iloc[0]["img_size"])
            ),
            None,
        )
        if best_holdout is not None:
            pd.DataFrame([best_holdout]).to_csv(
                RESULTS_DIR / "holdout_test_metrics.csv", index=False
            )

    best_per_arch_rows = []
    for row in (
        df_results.groupby("model", as_index=False)
        .head(1)
        .to_dict(orient="records")
    ):
        model_name = row["model"]
        img_size = int(row["img_size"])
        src_path = FINAL_MODEL_DIR / f"{model_name}_img{img_size}_final.keras"
        dst_path = FINAL_MODEL_DIR / f"{model_name}_best_final.keras"
        if not src_path.exists():
            print(f"[WARN] Best candidate missing on disk, skipping: {src_path.name}")
            continue
        dst_path.unlink(missing_ok=True)
        dst_path.symlink_to(src_path.name)
        best_per_arch_rows.append(
            {
                "model": model_name,
                "best_img_size": img_size,
                "f1_macro_mean": float(row["f1_macro_mean"]),
                "source_model_path": str(src_path),
                "best_model_path": str(dst_path),
            }
        )

    if best_per_arch_rows:
        pd.DataFrame(best_per_arch_rows).sort_values(
            "f1_macro_mean", ascending=False
        ).to_csv(RESULTS_DIR / "best_model_per_architecture.csv", index=False)


if __name__ == "__main__":
    main()
