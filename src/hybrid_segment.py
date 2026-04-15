import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 1. LOAD 256×256 IMAGES & MASKS
# ============================================================

image_dir = "B/image"
mask_dir  = "B/segment"

def load_folder(path):
    files = sorted(os.listdir(path))
    data = []
    for f in files:
        img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=-1)
        data.append(img)
    return np.array(data, dtype=np.float32) / 255.0

X = load_folder(image_dir)
y = load_folder(mask_dir)

print("Loaded:", X.shape, y.shape)

# ============================================================
# 2. TRAIN / VAL / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# ============================================================
# 3. tf.data PIPELINE
# ============================================================

BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(X_train, y_train, shuffle=True)
val_ds   = make_dataset(X_val, y_val)
test_ds  = make_dataset(X_test, y_test)

# ============================================================
# 4. DICE + BCE LOSS
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
# 5. MODEL DEFINITIONS
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

    # Encoder
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

    # Bottleneck + SE
    c5 = Conv2D(256, 3, activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)
    c5 = squeeze_excite_block(c5, 256)

    # Decoder
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

input_shape = (256, 256, 1)
unet_model   = UNet(input_shape)
seunet_model = SEUNet(input_shape)

# ============================================================
# 6. TRAINING SETUP
# ============================================================

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("loss_curves", exist_ok=True)

def get_callbacks(model_name):
    ckpt_path = f"checkpoints/{model_name}.weights.h5"
    checkpoint = ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    return [checkpoint, early_stop, reduce_lr]

EPOCHS = 10
histories = {}

# ---------------- UNet ----------------
print("\n===== Training UNet =====")
unet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=bce_dice_loss,
    metrics=["accuracy", dice_coefficient]
)
callbacks = get_callbacks("UNet")
hist_unet = unet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
histories["UNet"] = hist_unet

# ---------------- SEUNet ----------------
print("\n===== Training SEUNet =====")
seunet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=bce_dice_loss,
    metrics=["accuracy", dice_coefficient]
)
callbacks = get_callbacks("SEUNet")
hist_seunet = seunet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
histories["SEUNet"] = hist_seunet

# Save loss curves for base models
for name, history in histories.items():
    plt.figure(figsize=(8,5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"{name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"loss_curves/{name}_loss_curve.png")
    plt.close()

# Reload best weights
unet_model.load_weights("checkpoints/UNet.weights.h5")
seunet_model.load_weights("checkpoints/SEUNet.weights.h5")

# Freeze backbones for fusion
unet_model.trainable = False
seunet_model.trainable = False

# ============================================================
# 7. FUSION MODELS (Avg, Conv, Gated)
# ============================================================

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
    """
    Gated logit fusion:
    - Convert probabilities to logits
    - Learn per-pixel gate α(x) in [0,1]
    - Fused = α * UNet + (1-α) * SEUNet
    """
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

FusedAvg   = build_fused_avg(unet_model, seunet_model, input_shape)
FusedConv  = build_fused_conv(unet_model, seunet_model, input_shape)
FusedGated = build_fused_gated(unet_model, seunet_model, input_shape)

fusion_models = {
    "FusedAvg": FusedAvg,
    "FusedConv": FusedConv,
    "FusedGated": FusedGated
}

for name, model in fusion_models.items():
    print(f"\n===== Training {name} =====")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=bce_dice_loss,
        metrics=["accuracy", dice_coefficient]
    )
    callbacks = get_callbacks(name)
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    histories[name] = hist

    plt.figure(figsize=(8,5))
    plt.plot(hist.history["loss"], label="Train Loss")
    plt.plot(hist.history["val_loss"], label="Val Loss")
    plt.title(f"{name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"loss_curves/{name}_loss_curve.png")
    plt.close()

# ============================================================
# 8. CASCADED UNET → SEUNET REFINER
# ============================================================

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
    return Model(inputs, outputs, name="SEUNetRefiner")

def Cascaded_UNet_SEUNet(input_shape=(256,256,1)):
    inp = Input(input_shape)

    coarse = unet_model(inp)  # stage 1

    ref_input = concatenate([inp, coarse], axis=-1)
    se_refiner = SEUNet_refiner((256,256,2))
    refined = se_refiner(ref_input)  # stage 2

    x = concatenate([coarse, refined], axis=-1)
    x = Conv2D(8, 3, padding='same', activation='relu')(x)
    out = Conv2D(1, 1, activation='sigmoid')(x)

    return Model(inp, out, name="Cascaded_UNet_SEUNet")

CascadedModel = Cascaded_UNet_SEUNet(input_shape)

print("\n===== Training Cascaded_UNet_SEUNet =====")
CascadedModel.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=bce_dice_loss,
    metrics=["accuracy", dice_coefficient]
)
callbacks = get_callbacks("Cascaded_UNet_SEUNet")
hist_cascade = CascadedModel.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
histories["Cascaded_UNet_SEUNet"] = hist_cascade

plt.figure(figsize=(8,5))
plt.plot(hist_cascade.history["loss"], label="Train Loss")
plt.plot(hist_cascade.history["val_loss"], label="Val Loss")
plt.title("Cascaded_UNet_SEUNet Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curves/Cascaded_UNet_SEUNet_loss_curve.png")
plt.close()

# ============================================================
# 9. EVALUATION ON TEST SET
# ============================================================

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test, batch_size=BATCH_SIZE)
    preds_bin = (preds > 0.5).astype(np.uint8)

    acc = np.mean(preds_bin == y_test)
    f1 = f1_score(y_test.flatten(), preds_bin.flatten())
    intersection = np.logical_and(y_test, preds_bin).sum()
    union = np.logical_or(y_test, preds_bin).sum()
    iou = intersection / (union + 1e-6)

    return acc, f1, iou, preds_bin

# Reload best weights
unet_model.load_weights("checkpoints/UNet.weights.h5")
seunet_model.load_weights("checkpoints/SEUNet.weights.h5")
FusedAvg.load_weights("checkpoints/FusedAvg.weights.h5")
FusedConv.load_weights("checkpoints/FusedConv.weights.h5")
FusedGated.load_weights("checkpoints/FusedGated.weights.h5")
CascadedModel.load_weights("checkpoints/Cascaded_UNet_SEUNet.weights.h5")

all_models = {
    "UNet": unet_model,
    "SEUNet": seunet_model,
    "FusedAvg": FusedAvg,
    "FusedConv": FusedConv,
    "FusedGated": FusedGated,
    "Cascaded_UNet_SEUNet": CascadedModel
}

results = {}
predictions = {}

for name, model in all_models.items():
    print(f"\n===== Evaluating {name} =====")
    acc, f1, iou, preds_bin = evaluate_model(model, X_test, y_test)
    results[name] = (acc, f1, iou)
    predictions[name] = preds_bin

df = pd.DataFrame(results, index=["Accuracy", "F1 Score", "IoU"]).T
print("\n===== Test Results =====")
print(df)

# ============================================================
# 10. PREDICTION VISUALIZATION
# ============================================================

def show_predictions(X, y_true, preds_dict, idx=0):
    plt.figure(figsize=(16, 4))
    n_models = len(preds_dict)
    plt.subplot(1, n_models + 2, 1)
    plt.imshow(X[idx].squeeze(), cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, n_models + 2, 2)
    plt.imshow(y_true[idx].squeeze(), cmap="gray")
    plt.title("GT Mask")
    plt.axis("off")

    col = 3
    for name, preds in preds_dict.items():
        plt.subplot(1, n_models + 2, col)
        plt.imshow(preds[idx].squeeze(), cmap="gray")
        plt.title(name)
        plt.axis("off")
        col += 1

    plt.tight_layout()
    plt.show()

for i in range(3):
    show_predictions(X_test, y_test, predictions, idx=i)
