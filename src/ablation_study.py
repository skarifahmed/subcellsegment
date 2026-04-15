import os
import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from scipy.stats import ttest_rel, wilcoxon
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 0. GLOBAL CONFIG
# ============================================================

image_dir = "D/image"
mask_dir  = "D/segment"
input_shape = (256, 256, 1)
BATCH_SIZE = 8
EPOCHS = 25
N_FOLDS = 5
AUTOTUNE = tf.data.AUTOTUNE

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("loss_curves", exist_ok=True)
os.makedirs("cv_figures", exist_ok=True)
os.makedirs("qualitative_samples", exist_ok=True)

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_folder(path):
    files = sorted(os.listdir(path))
    data = []
    for f in files:
        img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=-1)
        data.append(img)
    return np.array(data, dtype=np.float32) / 255.0

X = load_folder(image_dir)
y = load_folder(mask_dir)

print("Loaded:", X.shape, y.shape)

# ============================================================
# 2. DATASET PIPELINE
# ============================================================

def make_dataset(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ============================================================
# 3. LOSSES & METRICS
# ============================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# ============================================================
# 4. MODEL DEFINITIONS
# ============================================================

def squeeze_excite_block(input_tensor, filters, ratio=16):
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return multiply([input_tensor, se])

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

def build_fused_avg(unet, seunet, input_shape):
    inp = Input(input_shape)
    u = unet(inp)
    s = seunet(inp)
    out = Average()([u, s])
    return Model(inp, out, name="FusedAvg")

def build_fused_conv(unet, seunet, input_shape):
    inp = Input(input_shape)
    u = unet(inp)
    s = seunet(inp)
    x = concatenate([u, s])
    x = Conv2D(8, 3, padding='same', activation='relu')(x)
    out = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inp, out, name="FusedConv")

def build_fused_gated(unet, seunet, input_shape):
    inp = Input(input_shape)
    u = unet(inp)
    s = seunet(inp)

    eps = 1e-6
    u_logit = Lambda(lambda z: tf.math.log(z + eps) - tf.math.log(1.0 - z + eps))(u)
    s_logit = Lambda(lambda z: tf.math.log(z + eps) - tf.math.log(1.0 - z + eps))(s)

    x = concatenate([u_logit, s_logit])
    gate = Conv2D(1, 1, activation='sigmoid')(x)

    fused = gate * u + (1.0 - gate) * s
    return Model(inp, fused, name="FusedGated")

def build_uncertainty_aware_fusion(unet_model, seunet_model, input_shape):
    inp = Input(shape=input_shape, name="fusion_input")

    p_u = unet_model(inp)
    p_s = seunet_model(inp)

    eps = 1e-6

    def prob_to_logit(p):
        return tf.math.log(p + eps) - tf.math.log(1.0 - p + eps)

    z_u = Lambda(prob_to_logit, name="logit_unet")(p_u)
    z_s = Lambda(prob_to_logit, name="logit_seunet")(p_s)

    def entropy_map(p):
        return -(p * tf.math.log(p + eps) +
                 (1.0 - p) * tf.math.log(1.0 - p + eps))

    u_u = Lambda(entropy_map, name="uncert_unet")(p_u)
    u_s = Lambda(entropy_map, name="uncert_seunet")(p_s)

    c_u = Conv2D(1, 1, activation="sigmoid", name="conf_unet")(u_u)
    c_s = Conv2D(1, 1, activation="sigmoid", name="conf_seunet")(u_s)

    d = Lambda(lambda t: tf.math.abs(t[0] - t[1]), name="disagreement")([p_u, p_s])

    a = Conv2D(8, 3, padding="same", activation="relu", name="att_conv1")(d)
    a = Conv2D(1, 3, padding="same", activation="sigmoid", name="att_conv2")(a)

    gate_inp = Concatenate(name="gate_concat")([z_u, z_s, c_u, c_s, a])
    g = Conv2D(1, 1, activation="sigmoid", name="gate_map")(gate_inp)

    z_fused = Lambda(lambda t: t[0] * t[2] + t[1] * (1.0 - t[2]), name="z_fused")([z_u, z_s, g])

    p_fused = Activation("sigmoid", name="p_fused")(z_fused)

    res_inp = Concatenate(name="residual_concat")([p_fused, p_u, p_s, a])
    r = Conv2D(8, 3, padding="same", activation="relu", name="res_conv1")(res_inp)
    r = Conv2D(1, 3, padding="same", activation=None, name="res_conv2")(r)

    def add_residual(args):
        p_base, r_logit = args
        z_base = tf.math.log(p_base + eps) - tf.math.log(1.0 - p_base + eps)
        return z_base + r_logit

    z_final = Lambda(add_residual, name="z_final")([p_fused, r])

    p_final = Activation("sigmoid", name="p_final")(z_final)

    return Model(inp, p_final, name="FusedUncAware")

def build_boundary_aware_fusion(unet_model, seunet_model, input_shape):
    inp = Input(shape=input_shape, name="bafusion_input")

    p_u = unet_model(inp)
    p_s = seunet_model(inp)

    eps = 1e-6

    def sobel_mag(x):
        edges = tf.image.sobel_edges(x)
        dx = edges[..., 0]
        dy = edges[..., 1]
        mag = tf.sqrt(dx * dx + dy * dy + eps)
        return mag

    b_u = Lambda(sobel_mag, name="boundary_unet")(p_u)
    b_s = Lambda(sobel_mag, name="boundary_seunet")(p_s)

    def norm01(x):
        x_min = tf.reduce_min(x, axis=[1,2,3], keepdims=True)
        x_max = tf.reduce_max(x, axis=[1,2,3], keepdims=True)
        return (x - x_min) / (x_max - x_min + eps)

    b_u = Lambda(norm01, name="bnd_unet_norm")(b_u)
    b_s = Lambda(norm01, name="bnd_seunet_norm")(b_s)

    reg_in = Concatenate(name="region_concat")([p_u, p_s])
    reg_x = Conv2D(16, 3, padding="same", activation="relu")(reg_in)
    reg_x = Conv2D(8, 3, padding="same", activation="relu")(reg_x)
    p_reg = Conv2D(1, 1, padding="same", activation="sigmoid", name="reg_out")(reg_x)

    bnd_in = Concatenate(name="boundary_concat")([b_u, b_s])
    bnd_x = Conv2D(16, 3, padding="same", activation="relu")(bnd_in)
    bnd_x = Conv2D(8, 3, padding="same", activation="relu")(bnd_x)
    p_bnd = Conv2D(1, 1, padding="same", activation="sigmoid", name="bnd_out")(bnd_x)

    fuse_in = Concatenate(name="final_concat")([p_reg, p_bnd])
    gate = Conv2D(1, 1, padding="same", activation="sigmoid", name="final_gate")(fuse_in)

    def fuse_rb(t):
        preg, pbnd, g = t
        return g * pbnd + (1.0 - g) * preg

    p_final = Lambda(fuse_rb, name="final_fused")([p_reg, p_bnd, gate])

    return Model(inp, p_final, name="FusedBAware")

def build_cross_attention_fusion(unet_model, seunet_model, input_shape):
    inp = Input(input_shape)

    p_u = unet_model(inp)
    p_s = seunet_model(inp)

    q_u = Conv2D(16, 1, padding="same")(p_u)
    k_s = Conv2D(16, 1, padding="same")(p_s)
    v_s = Conv2D(16, 1, padding="same")(p_s)

    q_s = Conv2D(16, 1, padding="same")(p_s)
    k_u = Conv2D(16, 1, padding="same")(p_u)
    v_u = Conv2D(16, 1, padding="same")(p_u)

    att_us = Activation("sigmoid")(Conv2D(1, 1)(q_u + k_s))
    att_su = Activation("sigmoid")(Conv2D(1, 1)(q_s + k_u))

    ca_us = Multiply()([v_s, att_us])
    ca_su = Multiply()([v_u, att_su])

    x = Concatenate()([p_u, p_s, ca_us, ca_su])
    x = Conv2D(32, 3, padding="same", activation="relu")(x)
    x = Conv2D(16, 3, padding="same", activation="relu")(x)
    out = Conv2D(1, 1, activation="sigmoid")(x)

    return Model(inp, out, name="FusedCrossAttention")

def build_shape_prior_fusion(unet_model, seunet_model, input_shape):
    inp = Input(input_shape)

    p_u = unet_model(inp)
    p_s = seunet_model(inp)

    eps = 1e-6
    def entropy(p):
        return -(p * tf.math.log(p + eps) + (1-p) * tf.math.log(1-p + eps))

    u_u = Lambda(entropy)(p_u)
    u_s = Lambda(entropy)(p_s)

    sp = Conv2D(8, 5, padding="same", activation="relu")(inp)
    sp = Conv2D(4, 5, padding="same", activation="relu")(sp)
    sp = Conv2D(1, 5, padding="same", activation="sigmoid")(sp)

    w_u = Conv2D(1, 1, activation="sigmoid")(u_u)
    w_s = Conv2D(1, 1, activation="sigmoid")(u_s)

    fused = (1 - w_u) * p_u + (1 - w_s) * p_s + 0.3 * sp
    out = Conv2D(1, 1, activation="sigmoid")(fused)

    return Model(inp, out, name="FusedShapePrior")

def build_multiscale_consistency_fusion(unet_model, seunet_model, input_shape):
    inp = Input(input_shape)

    p_u = unet_model(inp)
    p_s = seunet_model(inp)

    s1 = AveragePooling2D(2)(p_u)
    s2 = AveragePooling2D(4)(p_u)
    s3 = AveragePooling2D(8)(p_u)

    s1 = UpSampling2D(2)(s1)
    s2 = UpSampling2D(4)(s2)
    s3 = UpSampling2D(8)(s3)

    x = Concatenate()([p_u, p_s, s1, s2, s3])
    x = Conv2D(16, 3, padding="same", activation="relu")(x)
    x = Conv2D(1, 1, activation="sigmoid")(x)

    return Model(inp, x, name="FusedMultiScale")

# ============================================================
# NEW ADVANCED FUSION MODELS (MH-CrossAttention + UCAF)
# ============================================================

def mhsa_block(x, heads=4, dim=16):
    heads_out = []
    for _ in range(heads):
        q = Conv2D(dim, 1, padding="same")(x)
        k = Conv2D(dim, 1, padding="same")(x)
        v = Conv2D(dim, 1, padding="same")(x)

        att = Activation("sigmoid")(q + k)
        out = Multiply()([v, att])
        heads_out.append(out)

    return Concatenate()(heads_out)

def build_mh_cross_attention_fusion(unet_model, seunet_model, input_shape):
    inp = Input(input_shape)

    p_u = unet_model(inp)
    p_s = seunet_model(inp)

    ca_us = mhsa_block(Concatenate()([p_u, p_s]), heads=4, dim=16)
    ca_su = mhsa_block(Concatenate()([p_s, p_u]), heads=4, dim=16)

    x = Concatenate()([p_u, p_s, ca_us, ca_su])
    x = Conv2D(32, 3, padding="same", activation="relu")(x)
    x = Conv2D(16, 3, padding="same", activation="relu")(x)
    out = Conv2D(1, 1, activation="sigmoid")(x)

    return Model(inp, out, name="FusedMH_CrossAttention")

def build_uncertainty_cross_attention_fusion(unet_model, seunet_model, input_shape):
    inp = Input(input_shape)

    p_u = unet_model(inp)
    p_s = seunet_model(inp)

    eps = 1e-6
    def entropy(p):
        return -(p * tf.math.log(p + eps) + (1-p) * tf.math.log(1-p + eps))

    u_u = Lambda(entropy)(p_u)
    u_s = Lambda(entropy)(p_s)

    c_u = Conv2D(1, 1, activation="sigmoid")(u_u)
    c_s = Conv2D(1, 1, activation="sigmoid")(u_s)

    att_us = Activation("sigmoid")(Conv2D(1, 1)(p_u + p_s))
    att_su = Activation("sigmoid")(Conv2D(1, 1)(p_s + p_u))

    ca_us = Multiply()([p_s, att_us])
    ca_su = Multiply()([p_u, att_su])

    x = Concatenate()([p_u, p_s, ca_us, ca_su, c_u, c_s])
    x = Conv2D(32, 3, padding="same", activation="relu")(x)
    x = Conv2D(16, 3, padding="same", activation="relu")(x)
    out = Conv2D(1, 1, activation="sigmoid")(x)

    return Model(inp, out, name="FusedUCAF")

# ============================================================
# CASCADED MODEL
# ============================================================

def attention_gate(x, g, inter_channels):
    theta_x = Conv2D(inter_channels, 1, padding='same')(x)
    phi_g   = Conv2D(inter_channels, 1, padding='same')(g)
    add_xg  = Activation('relu')(add([theta_x, phi_g]))
    psi     = Conv2D(1, 1, padding='same')(add_xg)
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
    c4_att = attention_gate(c4, u6, inter_channels=64)
    u6 = concatenate([u6, c4_att])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    c3_att = attention_gate(c3, u7, inter_channels=32)
    u7 = concatenate([u7, c3_att])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, 2, strides=2, padding='same')(c7)
    c2_att = attention_gate(c2, u8, inter_channels=16)
    u8 = concatenate([u8, c2_att])
    c8 = Conv2D(32, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(32, 3, activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, 2, strides=2, padding='same')(c8)
    c1_att = attention_gate(c1, u9, inter_channels=8)
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
    x = Conv2D(8, 3, padding='same', activation='relu')(x)
    out = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inp, [coarse, out], name="Cascaded_UNet_SEUNet")

# ============================================================
# 5. CALLBACKS
# ============================================================

def get_callbacks(model_name, fold_idx):
    ckpt_path = f"checkpoints/fold{fold_idx}_{model_name}.weights.h5"
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

# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    if isinstance(preds, (list, tuple)):
        preds = preds[-1]
    preds_bin = (preds > 0.5).astype(np.uint8)

    acc = np.mean(preds_bin == y_test)
    f1 = f1_score(y_test.flatten(), preds_bin.flatten())
    intersection = np.logical_and(y_test, preds_bin).sum()
    union = np.logical_or(y_test, preds_bin).sum()
    iou = intersection / (union + 1e-6)

    return acc, f1, iou

# ============================================================
# 7. 5-FOLD CROSS-VALIDATION (FULL)
# ============================================================

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results = {
    "UNet": [],
    "SEUNet": [],
    "FusedAvg": [],
    "FusedConv": [],
    "FusedGated": [],
    "FusedUncAware": [],
    "FusedBAware": [],
    "FusedCrossAttention": [],
    "FusedShapePrior": [],
    "FusedMultiScale": [],
    "FusedMH_CrossAttention": [],
    "FusedUCAF": [],
    "Cascaded_UNet_SEUNet": []
}

fold_idx = 1

for train_index, test_index in kf.split(X):
    print(f"\n==============================")
    print(f"===== Fold {fold_idx} / {N_FOLDS} =====")
    print(f"==============================\n")

    X_train_full, X_test = X[train_index], X[test_index]
    y_train_full, y_test = y[train_index], y[test_index]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    train_ds = make_dataset(X_train, y_train, shuffle=True)
    val_ds   = make_dataset(X_val, y_val)

    # ---------- UNet ----------
    unet_model = UNet(input_shape)
    callbacks, ckpt_unet = get_callbacks("UNet", fold_idx)
    unet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=bce_dice_loss,
        metrics=["accuracy", dice_coefficient]
    )
    hist_unet = unet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0
    )
    plt.figure()
    plt.plot(hist_unet.history["loss"], label="Train")
    plt.plot(hist_unet.history["val_loss"], label="Val")
    plt.title(f"Fold {fold_idx} - UNet Loss")
    plt.legend()
    plt.savefig(f"loss_curves/fold{fold_idx}_UNet_loss.png")
    plt.close()
    unet_model.load_weights(ckpt_unet)

    # ---------- SEUNet ----------
    seunet_model = SEUNet(input_shape)
    callbacks, ckpt_se = get_callbacks("SEUNet", fold_idx)
    seunet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=bce_dice_loss,
        metrics=["accuracy", dice_coefficient]
    )
    hist_se = seunet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0
    )
    plt.figure()
    plt.plot(hist_se.history["loss"], label="Train")
    plt.plot(hist_se.history["val_loss"], label="Val")
    plt.title(f"Fold {fold_idx} - SEUNet Loss")
    plt.legend()
    plt.savefig(f"loss_curves/fold{fold_idx}_SEUNet_loss.png")
    plt.close()
    seunet_model.load_weights(ckpt_se)

    # freeze for fusion
    unet_model.trainable = False
    seunet_model.trainable = False

    # ---------- Fusion models ----------
    FusedAvg   = build_fused_avg(unet_model, seunet_model, input_shape)
    FusedConv  = build_fused_conv(unet_model, seunet_model, input_shape)
    FusedGated = build_fused_gated(unet_model, seunet_model, input_shape)
    FusedUnc   = build_uncertainty_aware_fusion(unet_model, seunet_model, input_shape)
    FusedBA    = build_boundary_aware_fusion(unet_model, seunet_model, input_shape)
    FusedCA    = build_cross_attention_fusion(unet_model, seunet_model, input_shape)
    FusedSP    = build_shape_prior_fusion(unet_model, seunet_model, input_shape)
    FusedMSC   = build_multiscale_consistency_fusion(unet_model, seunet_model, input_shape)
    FusedMHCA  = build_mh_cross_attention_fusion(unet_model, seunet_model, input_shape)
    FusedUCAF  = build_uncertainty_cross_attention_fusion(unet_model, seunet_model, input_shape)

    fusion_dict = {
        "FusedAvg": FusedAvg,
        "FusedConv": FusedConv,
        "FusedGated": FusedGated,
        "FusedUncAware": FusedUnc,
        "FusedBAware": FusedBA,
        "FusedCrossAttention": FusedCA,
        "FusedShapePrior": FusedSP,
        "FusedMultiScale": FusedMSC,
        "FusedMH_CrossAttention": FusedMHCA,
        "FusedUCAF": FusedUCAF
    }

    for name, model in fusion_dict.items():
        callbacks, ckpt_f = get_callbacks(name, fold_idx)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=bce_dice_loss,
            metrics=["accuracy", dice_coefficient]
        )
        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=0
        )
        plt.figure()
        plt.plot(hist.history["loss"], label="Train")
        plt.plot(hist.history["val_loss"], label="Val")
        plt.title(f"Fold {fold_idx} - {name} Loss")
        plt.legend()
        plt.savefig(f"loss_curves/fold{fold_idx}_{name}_loss.png")
        plt.close()
        model.load_weights(ckpt_f)

    # ---------- Cascaded model ----------
    CascadedModel = Cascaded_UNet_SEUNet(input_shape, unet_backbone=unet_model)

    def ds_for_cascade(ds):
        return ds.map(lambda x, y: (x, (y, y)))

    train_ds_cascade = ds_for_cascade(train_ds)
    val_ds_cascade   = ds_for_cascade(val_ds)

    callbacks, ckpt_cas = get_callbacks("Cascaded_UNet_SEUNet", fold_idx)
    CascadedModel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=[bce_dice_loss, bce_dice_loss],
        loss_weights=[0.3, 1.0],
        metrics=["accuracy", dice_coefficient]
    )
    hist_cas = CascadedModel.fit(
        train_ds_cascade,
        validation_data=val_ds_cascade,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0
    )
    plt.figure()
    plt.plot(hist_cas.history["loss"], label="Train")
    plt.plot(hist_cas.history["val_loss"], label="Val")
    plt.title(f"Fold {fold_idx} - Cascaded Loss")
    plt.legend()
    plt.savefig(f"loss_curves/fold{fold_idx}_Cascaded_loss.png")
    plt.close()
    CascadedModel.load_weights(ckpt_cas)

    # ---------- Evaluate on this fold's test set ----------
    models_eval = {
        "UNet": unet_model,
        "SEUNet": seunet_model,
        "FusedAvg": FusedAvg,
        "FusedConv": FusedConv,
        "FusedGated": FusedGated,
        "FusedUncAware": FusedUnc,
        "FusedBAware": FusedBA,
        "FusedCrossAttention": FusedCA,
        "FusedShapePrior": FusedSP,
        "FusedMultiScale": FusedMSC,
        "FusedMH_CrossAttention": FusedMHCA,
        "FusedUCAF": FusedUCAF,
        "Cascaded_UNet_SEUNet": CascadedModel
    }

    for name, model in models_eval.items():
        try:
            acc, f1, iou = evaluate_model(model, X_test, y_test)
            fold_results[name].append((acc, f1, iou))
        except Exception as e:
            print(f"ERROR evaluating {name}: {e}")

    # ---------- QUALITATIVE GRIDS: ALL METHODS ON SAME SAMPLES ----------
    # pick up to 10 random test samples
    np.random.seed(100 + fold_idx)
    n_samples = min(10, len(X_test))
    idxs = np.random.choice(len(X_test), size=n_samples, replace=False)

    # run predictions for all models on these samples
    preds_dict = {}
    for name, model in models_eval.items():
        p = model.predict(X_test[idxs], batch_size=BATCH_SIZE, verbose=0)
        if isinstance(p, (list, tuple)):
            p = p[-1]
        preds_dict[name] = (p > 0.5).astype(np.uint8)

    model_order = [
        "UNet",
        "SEUNet",
        "FusedAvg",
        "FusedConv",
        "FusedGated",
        "FusedUncAware",
        "FusedBAware",
        "FusedCrossAttention",
        "FusedShapePrior",
        "FusedMultiScale",
        "FusedMH_CrossAttention",
        "FusedUCAF",
        "Cascaded_UNet_SEUNet"
    ]

    for j, idx in enumerate(idxs):
        img = X_test[idx, ..., 0]
        gt  = y_test[idx, ..., 0]

        # 4x4 grid: 16 slots -> Input, GT, 13 models, 1 empty
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()

        # slot 0: input
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")

        # slot 1: GT
        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # fill remaining with model predictions
        k = 2
        for name in model_order:
            if k >= len(axes):
                break
            pred = preds_dict[name][j, ..., 0]
            axes[k].imshow(pred, cmap="gray")
            axes[k].set_title(name, fontsize=8)
            axes[k].axis("off")
            k += 1

        # any leftover axes: turn off
        while k < len(axes):
            axes[k].axis("off")
            k += 1

        fig.suptitle(f"Fold {fold_idx} - Sample {j}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"qualitative_samples/fold{fold_idx}_sample{j}_grid.png", dpi=150)
        plt.close()

    fold_idx += 1

# ============================================================
# 8. AGGREGATE CV RESULTS & SAVE FIGURES
# ============================================================

summary_rows = []
for name, scores in fold_results.items():
    scores = np.array(scores)

    if len(scores) == 0:
        print(f"WARNING: No scores for model {name}. Skipping.")
        continue

    acc_mean, f1_mean, iou_mean = scores.mean(axis=0)
    acc_std,  f1_std,  iou_std  = scores.std(axis=0)

    summary_rows.append({
        "Model": name,
        "Acc_mean": acc_mean,
        "Acc_std": acc_std,
        "F1_mean": f1_mean,
        "F1_std": f1_std,
        "IoU_mean": iou_mean,
        "IoU_std": iou_std
    })

    plt.figure()
    plt.boxplot([scores[:,1], scores[:,2]], labels=["F1", "IoU"])
    plt.title(f"{name} - 5-fold F1/IoU")
    plt.savefig(f"cv_figures/{name}_boxplot.png")
    plt.close()

summary_df = pd.DataFrame(summary_rows)
print("\n===== 5-FOLD CV SUMMARY =====")
print(summary_df)

summary_df.to_csv("cv_figures/cv_summary.csv", index=False)

# ============================================================
# 9. STATISTICAL TESTS (Cascaded vs UNet)
# ============================================================

unet_scores = np.array(fold_results["UNet"])
cas_scores  = np.array(fold_results["Cascaded_UNet_SEUNet"])

for metric_idx, metric_name in zip([0,1,2], ["Accuracy", "F1", "IoU"]):
    u = unet_scores[:, metric_idx]
    c = cas_scores[:, metric_idx]

    t_stat, t_p = ttest_rel(c, u)
    try:
        w_stat, w_p = wilcoxon(c, u)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    print(f"\n=== Cascaded vs UNet ({metric_name}) ===")
    print(f"Paired t-test: t = {t_stat:.4f}, p = {t_p:.4e}")
    print(f"Wilcoxon:      W = {w_stat:.4f}, p = {w_p:.4e}")

plt.figure(figsize=(8,5))
models = list(fold_results.keys())
means  = [np.mean(np.array(fold_results[m])[:,1]) for m in models if len(fold_results[m]) > 0]
stds   = [np.std(np.array(fold_results[m])[:,1]) for m in models if len(fold_results[m]) > 0]
labels = [m for m in models if len(fold_results[m]) > 0]
plt.bar(labels, means, yerr=stds, capsize=5)
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1 Score")
plt.title("5-fold F1 (mean ± std)")
plt.tight_layout()
plt.savefig("cv_figures/F1_barplot.png")
plt.close()
