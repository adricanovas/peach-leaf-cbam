import gc
import json
import os
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2
import albumentations as A
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Shared configuration
# =============================================================================

MODELS = [
    "mobilenetv2",
    "efficientnetb0",
    "mobilenetv3large",
    "densenet121",
    "efficientnetb3",
    "resnet50",
    "inceptionv3",
    "resnet101",
    "vgg16",
    "vgg19",
    "efficientnetb5",
]

MODEL_SPECS = {
    "mobilenetv2": (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    "efficientnetb0": (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
    "mobilenetv3large": (tf.keras.applications.MobileNetV3Large, tf.keras.applications.mobilenet_v3.preprocess_input),
    "densenet121": (tf.keras.applications.DenseNet121, tf.keras.applications.densenet.preprocess_input),
    "efficientnetb3": (tf.keras.applications.EfficientNetB3, tf.keras.applications.efficientnet.preprocess_input),
    "resnet50": (tf.keras.applications.ResNet50, tf.keras.applications.resnet.preprocess_input),
    "inceptionv3": (tf.keras.applications.InceptionV3, tf.keras.applications.inception_v3.preprocess_input),
    "resnet101": (tf.keras.applications.ResNet101, tf.keras.applications.resnet.preprocess_input),
    "vgg16": (tf.keras.applications.VGG16, tf.keras.applications.vgg16.preprocess_input),
    "vgg19": (tf.keras.applications.VGG19, tf.keras.applications.vgg19.preprocess_input),
    "efficientnetb5": (tf.keras.applications.EfficientNetB5, tf.keras.applications.efficientnet.preprocess_input),
}


# =============================================================================
# Seeds & determinism
# =============================================================================

def set_global_seeds(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# =============================================================================
# Data loading & augmentation
# =============================================================================

def create_augmentation_pipeline(
    is_training: bool = True,
    img_size: Tuple[int, int] = (224, 224),
) -> A.Compose:
    if is_training:
        return A.Compose([
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
            A.RandomResizedCrop(
                size=(img_size[0], img_size[1]),
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1),
                interpolation=cv2.INTER_LINEAR,
                p=0.3,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3),
            A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=5, val_shift_limit=4, p=0.2),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.95, 1.05),
                rotate=(0, 0),
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                p=0.3,
            ),
        ])
    else:
        return A.Compose([A.Resize(height=img_size[0], width=img_size[1])])


def load_paths_and_labels(data_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    paths, labels = [], []
    for cdir in class_dirs:
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(str(p))
                labels.append(class_to_idx[cdir.name])

    return np.array(paths), np.array(labels), class_names


def make_dataset(
    paths,
    labels,
    img_size,
    preprocess_fn,
    batch_size=32,
    training=False,
    use_augmentation=False,
    seed: int = 0,
):
    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    num_classes = tf.reduce_max(labels) + 1

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    augmenter = (
        create_augmentation_pipeline(is_training=True, img_size=(img_size, img_size))
        if training and use_augmentation
        else None
    )

    def _augment_image(img):
        aug = augmenter(image=img.numpy())
        return aug["image"].astype(np.uint8)

    def _load(path, y):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)

        if augmenter is not None:
            img = tf.py_function(_augment_image, [img], Tout=tf.uint8)
            img.set_shape([None, None, 3])

        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32)
        img = preprocess_fn(img)
        y = tf.one_hot(y, depth=num_classes)
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# Model building
# =============================================================================

def build_model(model_name, num_classes, img_size, model_specs):
    base_cls, _ = model_specs[model_name]
    base = base_cls(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base


class SpatialPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialPooling, self).__init__(**kwargs)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        return tf.concat([avg_pool, max_pool], axis=-1)

    def get_config(self):
        return super().get_config()


def cbam_block(input_tensor, reduction_ratio=16):
    channel = input_tensor.shape[-1]
    bottleneck = max(1, channel // reduction_ratio)
    shared_dense_one = tf.keras.layers.Dense(
        bottleneck,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
    )
    shared_dense_two = tf.keras.layers.Dense(
        channel,
        kernel_initializer="he_normal",
        use_bias=True,
    )

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = tf.keras.layers.BatchNormalization()(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_tensor)
    max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = tf.keras.layers.BatchNormalization()(max_pool)
    max_pool = shared_dense_two(max_pool)

    channel_attention = tf.keras.layers.Add()([avg_pool, max_pool])
    channel_attention = tf.keras.layers.Activation("sigmoid")(channel_attention)
    channel_refined = tf.keras.layers.Multiply()([input_tensor, channel_attention])

    concat = SpatialPooling()(channel_refined)
    spatial_attention = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=7,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(concat)
    refined = tf.keras.layers.Multiply()([channel_refined, spatial_attention])
    return refined


def build_multi_output_backbone(base, depth_fraction=0.7):
    """
    Wrap a backbone to expose two feature maps: one at ~depth_fraction of its depth
    (larger spatial resolution, better for spatial attention) and the final one.
    Returns None on failure so the caller can fall back to single-scale.
    """
    candidate_layers = []
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        try:
            out = layer.output
            if len(out.shape) == 4:
                candidate_layers.append(layer)
        except (AttributeError, RuntimeError):
            continue

    if len(candidate_layers) < 2:
        return None

    idx = max(0, int(len(candidate_layers) * depth_fraction) - 1)
    mid_layer = candidate_layers[idx]

    try:
        return tf.keras.Model(
            inputs=base.input,
            outputs=[mid_layer.output, base.output],
        )
    except Exception:
        return None


# =============================================================================
# Shared training utilities
# =============================================================================

def get_class_weight(labels, num_classes):
    cls_w = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels,
    )
    return {i: float(w) for i, w in enumerate(cls_w)}


def evaluate_predictions(model, dataset, y_true, class_names=None):
    """Evaluate model predictions. Per-class metrics are included when class_names is provided."""
    y_prob = model.predict(dataset, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    del y_prob
    gc.collect()
    result = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if class_names is not None:
        result["precision_per_class"] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        result["recall_per_class"] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        result["f1_per_class"] = f1_score(y_true, y_pred, average=None).tolist()
    return result


def set_finetune_layers(base_model, unfreeze_ratio):
    base_model.trainable = True
    n_layers = len(base_model.layers)
    start_idx = int(n_layers * (1 - unfreeze_ratio))
    for layer in base_model.layers[:start_idx]:
        layer.trainable = False
    for layer in base_model.layers[start_idx:]:
        # Explicit per-layer flag required: model-level trainable=True does not cascade
        # to individual layer flags when the model was built from a frozen backbone.
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)


def train_two_stage(
    model,
    base_model,
    train_ds,
    class_weight,
    head_epochs,
    finetune_epochs,
    head_lr,
    finetune_lr,
    unfreeze_ratio,
    val_ds=None,
):
    head_callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    finetune_callbacks = [tf.keras.callbacks.TerminateOnNaN()]

    if val_ds is not None:
        head_callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        )
        finetune_callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            )
        )
        finetune_callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
            )
        )
    base_model.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(head_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs,
        class_weight=class_weight,
        callbacks=head_callbacks,
        verbose=1,
    )

    set_finetune_layers(base_model, unfreeze_ratio)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(finetune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=finetune_epochs,
        class_weight=class_weight,
        callbacks=finetune_callbacks,
        verbose=1,
    )


# =============================================================================
# Transfer learning utilities
# =============================================================================

def find_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No backbone submodel found in model.")


def _mark_head_trainable(model, trainable):
    for layer in model.layers:
        if not isinstance(layer, tf.keras.Model):
            layer.trainable = trainable


def apply_fine_tuning_strategy(model, strategy, last_n_layers=20, freeze_batchnorm=True):
    backbone = find_backbone(model)

    if strategy == "feature_extractor":
        backbone.trainable = False
        _mark_head_trainable(model, True)
        return model

    if strategy == "fine_tune_last":
        backbone.trainable = True
        cutoff = max(0, len(backbone.layers) - last_n_layers)
        for layer in backbone.layers[:cutoff]:
            layer.trainable = False
        for layer in backbone.layers[cutoff:]:
            layer.trainable = not (
                freeze_batchnorm and isinstance(layer, tf.keras.layers.BatchNormalization)
            )
        _mark_head_trainable(model, True)
        return model

    if strategy == "fine_tune_all":
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = not (
                freeze_batchnorm and isinstance(layer, tf.keras.layers.BatchNormalization)
            )
        _mark_head_trainable(model, True)
        return model

    raise ValueError(f"Unknown strategy: {strategy}")


def rebuild_model_for_target_classes(model, num_classes):
    """Replace the classifier head while preserving all learned features (backbone + CBAM)."""
    if len(model.layers) >= 2 and isinstance(model.layers[-1], tf.keras.layers.Dense):
        inputs = model.input
        x = model.layers[-2].output
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="tl_head")(x)
        return tf.keras.Model(inputs, outputs)
    backbone = find_backbone(model)
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = backbone(inputs, training=False)
    if isinstance(x, (list, tuple)):
        x_mid, x_final = x
        x_mid = cbam_block(x_mid, reduction_ratio=16)
        x_mid = tf.keras.layers.GlobalAveragePooling2D()(x_mid)
        x_final = cbam_block(x_final, reduction_ratio=16)
        x_final = tf.keras.layers.GlobalAveragePooling2D()(x_final)
        x = tf.keras.layers.Concatenate()([x_mid, x_final])
    else:
        x = cbam_block(x, reduction_ratio=16)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="tl_head")(x)
    return tf.keras.Model(inputs, outputs)


def train_tl(model, train_ds, val_ds, class_weight, epochs=15, lr=1e-4):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )


def upsert_result(rows, row):
    """Upsert a TL result row keyed by (model, img_size, strategy)."""
    filtered = [
        item for item in rows
        if not (
            item["model"] == row["model"]
            and int(item["img_size"]) == int(row["img_size"])
            and item["strategy"] == row["strategy"]
        )
    ]
    filtered.append(row)
    return filtered


# =============================================================================
# CV training helpers
# =============================================================================

def fold_item_key(model_name, img_size, fold):
    return f"{model_name}|img{img_size}|fold{int(fold):02d}"


def find_fold_result(fold_results, model_name, img_size, fold):
    return next(
        (
            row for row in fold_results
            if (
                row["model"] == model_name
                and int(row["img_size"]) == img_size
                and int(row["fold"]) == fold
            )
        ),
        None,
    )


def upsert_fold_result(fold_results, row):
    filtered = [
        item for item in fold_results
        if not (
            item["model"] == row["model"]
            and int(item["img_size"]) == int(row["img_size"])
            and int(item["fold"]) == int(row["fold"])
        )
    ]
    filtered.append(row)
    return filtered


def upsert_model_metric(rows, row):
    filtered = [
        item for item in rows
        if not (
            item["model"] == row["model"]
            and int(item["img_size"]) == int(row["img_size"])
        )
    ]
    filtered.append(row)
    return filtered


def load_csv_records(path):
    return pd.read_csv(path).to_dict(orient="records") if Path(path).exists() else []


def load_partial_metrics(results_dir):
    fold_results = load_csv_records(results_dir / "cv_fold_metrics.csv")
    results = load_csv_records(results_dir / "cv_ranking.csv")
    return fold_results, results


def persist_partial_metrics(fold_results, results, results_dir):
    df_folds = pd.DataFrame(fold_results)
    df_results = pd.DataFrame(results)
    if not df_folds.empty:
        df_folds.to_csv(results_dir / "cv_fold_metrics.csv", index=False)
    if not df_results.empty:
        df_results.sort_values("f1_macro_mean", ascending=False).to_csv(
            results_dir / "cv_ranking.csv", index=False
        )


# =============================================================================
# Progress tracking & persistence
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def build_training_progress(
    total_steps: int,
    done_items: List[str],
    pending_items: List[str],
    current_item: Optional[str] = None,
) -> Dict:
    done = len(done_items)
    return {
        "updated_at_utc": utc_now_iso(),
        "total_steps": int(total_steps),
        "completed_steps": int(done),
        "pending_steps": int(max(total_steps - done, 0)),
        "progress": f"{done}/{total_steps}",
        "current": current_item,
        "completed_items": done_items,
        "pending_items": pending_items,
    }


# =============================================================================
# Visualisation
# =============================================================================

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, backbone_name, out_dir):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    cm_color = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    cm_color = np.nan_to_num(cm_color)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_color,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = out_dir / f"confusion_matrix_{backbone_name}.png"
    try:
        plt.savefig(out_path, dpi=300)
    finally:
        plt.close()
