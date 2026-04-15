import matplotlib
matplotlib.use("Agg")

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_erosion

# ================== CONFIG ==================

DATASETS = ["A", "B", "C", "D"]
INPUT_SHAPE = (256, 256, 1)
BATCH_SIZE = 8
EPOCHS = 25
AUTOTUNE = tf.data.AUTOTUNE

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ================== DATA ==================

def load_folder(path, dataset_name):
    files = sorted(os.listdir(path))
    data, names = [], []
    for f in files:
        img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=-1)
        data.append(img)
        names.append(f"{dataset_name}/image/{f}")
    return np.array(data, dtype=np.float32) / 255.0, names

def load_dataset(name):
    X, names = load_folder(os.path.join(name, "image"), name)
    y, _ = load_folder(os.path.join(name, "segment"), name)
    print(f"Loaded {name}: ", X.shape, y.shape)
    return X, y, names

def make_dataset(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X))
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ================== METRICS (HD95, ASD, Dice) ==================

def surface_distances(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    if not gt.any() and not pred.any():
        return np.array([0.0]), np.array([0.0])
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    gt_surf = gt ^ binary_erosion(gt)
    pred_surf = pred ^ binary_erosion(pred)
    d1 = dt_pred[gt_surf] if gt_surf.any() else np.array([0.0])
    d2 = dt_gt[pred_surf] if pred_surf.any() else np.array([0.0])
    return d1, d2

def hd95(gt, pred):
    d1, d2 = surface_distances(gt, pred)
    return float(np.percentile(np.hstack([d1, d2]), 95))

def asd(gt, pred):
    d1, d2 = surface_distances(gt, pred)
    return float((d1.mean() + d2.mean()) / 2.0)

def dice_score(gt, pred, smooth=1e-6):
    gt_f = gt.flatten().astype(np.float32)
    pr_f = pred.flatten().astype(np.float32)
    inter = np.sum(gt_f * pr_f)
    return float((2.0 * inter + smooth) / (np.sum(gt_f) + np.sum(pr_f) + smooth))

def evaluate_metrics(model, X_test, y_test):
    preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    if isinstance(preds, (list, tuple)):
        preds = preds[-1]
    preds_bin = (preds > 0.5).astype(np.uint8)

    hd_list, asd_list, dice_list = [], [], []

    for i in range(len(X_test)):
        gt = y_test[i, ..., 0].astype(np.uint8)
        pr = preds_bin[i, ..., 0].astype(np.uint8)

        hd_list.append(hd95(gt, pr))
        asd_list.append(asd(gt, pr))
        dice_list.append(dice_score(gt, pr))

    return (
        float(np.mean(hd_list)),
        float(np.mean(asd_list)),
        float(np.mean(dice_list))
    )

# ================== LOSSES ==================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# ================== MODELS ==================

def squeeze_excite_block(x, filters, ratio=16):
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return multiply([x, se])

def UNet(input_shape=(256,256,1)):
    inputs = Input(input_shape)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D()(c4)

    c5 = Conv2D(256, 3, activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, 2, strides=2, padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(32, 3, activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, 2, strides=2, padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(16, 3, activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    return Model(inputs, outputs, name="UNet")

def attention_gate(x, g, inter_channels):
    theta_x = Conv2D(inter_channels, 1, padding='same')(x)
    phi_g   = Conv2D(inter_channels, 1, padding='same')(g)
    act     = Activation('relu')(add([theta_x, phi_g]))
    psi     = Conv2D(1, 1, padding='same')(act)
    psi     = Activation('sigmoid')(psi)
    return multiply([x, psi])

def SEUNet_refiner(input_shape=(256,256,2)):
    inputs = Input(input_shape)

    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D()(c4)

    c5 = Conv2D(256, 3, activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)
    c5 = squeeze_excite_block(c5, 256)

    u6 = Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    c4_att = attention_gate(c4, u6, 64)
    u6 = concatenate([u6, c4_att])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    c3_att = attention_gate(c3, u7, 32)
    u7 = concatenate([u7, c3_att])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, 2, strides=2, padding='same')(c7)
    c2_att = attention_gate(c2, u8, 16)
    u8 = concatenate([u8, c2_att])
    c8 = Conv2D(32, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(32, 3, activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, 2, strides=2, padding='same')(c8)
    c1_att = attention_gate(c1, u9, 8)
    u9 = concatenate([u9, c1_att])
    c9 = Conv2D(16, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(16, 3, activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    return Model(inputs, outputs, name="SEUNetRefiner_Att")

def Cascaded_UNet_SEUNet(input_shape=(256,256,1), unet_backbone=None):
    inp = Input(input_shape)
    coarse = unet_backbone(inp)
    ref_input = concatenate([inp, coarse], axis=-1)
    se_refiner = SEUNet_refiner((256,256,2))
    refined = se_refiner(ref_input)
    x = concatenate([coarse, refined], axis=-1)
    x = Conv2D(8, 3, activation='relu', padding='same')(x)
    out = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inp, [coarse, out], name="Cascaded_UNet_SEUNet")

def SEUNet(input_shape=(256,256,1)):
    inputs = Input(input_shape)

    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D()(c4)

    c5 = Conv2D(256, 3, activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)
    c5 = squeeze_excite_block(c5, 256)

    u6 = Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, 2, strides=2, padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(32, 3, activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, 2, strides=2, padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(16, 3, activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    return Model(inputs, outputs, name="SEUNet")

# ================== CALLBACKS ==================

def get_callbacks(model_name, tag):
    ckpt_path = f"models/{tag}_{model_name}.weights.h5"
    checkpoint = ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0
    )
    return [checkpoint, early_stop, reduce_lr], ckpt_path

# ================== LOAD ALL DATA ==================

all_data = {}
for ds_name in DATASETS:
    all_data[ds_name] = load_dataset(ds_name)

# ================== 5-FOLD CROSS VALIDATION ==================

results_rows = []

for ds_name in DATASETS:
    print("\n======================================")
    print(f"5-FOLD CROSS-VALIDATION ON DATASET {ds_name}")
    print("======================================\n")

    X_all, y_all, names_all = all_data[ds_name]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_idx = 1
    fold_metrics = []

    for train_idx, test_idx in kf.split(X_all):
        print(f"\n--- Fold {fold_idx} ---")

        X_train_full, X_test = X_all[train_idx], X_all[test_idx]
        y_train_full, y_test = y_all[train_idx], y_all[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2,
            shuffle=True, random_state=42
        )

        train_ds = make_dataset(X_train, y_train, shuffle=True)
        val_ds   = make_dataset(X_val, y_val, shuffle=False)

        tag_base = f"{ds_name}_fold{fold_idx}"

        # UNet
        unet_model = UNet(INPUT_SHAPE)
        callbacks, ckpt_unet = get_callbacks("UNet", tag_base)
        unet_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=bce_dice_loss
        )
        unet_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=0
        )
        unet_model.load_weights(ckpt_unet)

        # SEUNet
        seunet_model = SEUNet(INPUT_SHAPE)
        callbacks, ckpt_se = get_callbacks("SEUNet", tag_base)
        seunet_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=bce_dice_loss
        )
        seunet_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=0
        )
        seunet_model.load_weights(ckpt_se)

        # Cascaded
        casc_unet_backbone = UNet(INPUT_SHAPE)
        casc_unet_backbone.set_weights(unet_model.get_weights())
        casc_unet_backbone.trainable = True

        casc_model = Cascaded_UNet_SEUNet(INPUT_SHAPE, unet_backbone=casc_unet_backbone)

        train_ds_cascade = train_ds.map(lambda x, y: (x, (y, y)))
        val_ds_cascade   = val_ds.map(lambda x, y: (x, (y, y)))

        callbacks, ckpt_cas = get_callbacks("Cascaded_UNet_SEUNet", tag_base)
        casc_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=[bce_dice_loss, bce_dice_loss],
            loss_weights=[0.3, 1.0]
        )
        casc_model.fit(
            train_ds_cascade,
            validation_data=val_ds_cascade,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=0
        )
        casc_model.load_weights(ckpt_cas)

        # Evaluation
        hd_u, asd_u, dice_u = evaluate_metrics(unet_model, X_test, y_test)
        hd_s, asd_s, dice_s = evaluate_metrics(seunet_model, X_test, y_test)
        hd_c, asd_c, dice_c = evaluate_metrics(casc_model, X_test, y_test)

        fold_metrics.append([
            hd_u, asd_u, dice_u,
            hd_s, asd_s, dice_s,
            hd_c, asd_c, dice_c
        ])

        fold_idx += 1

    fold_metrics = np.array(fold_metrics)
    mean_vals = fold_metrics.mean(axis=0)

    results_rows.append({
        "Dataset": ds_name,
        "UNet_HD95": mean_vals[0],
        "UNet_ASD": mean_vals[1],
        "UNet_Dice": mean_vals[2],
        "SEUNet_HD95": mean_vals[3],
        "SEUNet_ASD": mean_vals[4],
        "SEUNet_Dice": mean_vals[5],
        "Cascaded_HD95": mean_vals[6],
        "Cascaded_ASD": mean_vals[7],
        "Cascaded_Dice": mean_vals[8]
    })

results_df = pd.DataFrame(results_rows)
results_df.to_csv("results/5fold_results_HD95_ASD_Dice_UNet_SEUNet_Cascaded.csv", index=False)
print("\n===== FINAL 5-FOLD RESULTS (HD95, ASD, Dice) =====")
print(results_df)