import os
import cv2
import matplotlib
matplotlib.use('Agg')

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_e12a922e07b325050bd7a5dd362c2dcf'

import glob
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from unet_model import bce_dice_loss, dice_coefficient

# ==========================================
# CONFIGURACIÓN
# ==========================================
DATA_DIR = os.path.join('.', 'data', 'raw')
UNET_MODEL_PATH = 'best_unet_model.keras'
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 40

# ==========================================
# PRE-PROCESAMIENTO CLAHE (Idéntico a V2)
# ==========================================
def apply_clahe_numpy(image):
    image_uint8 = (image * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image_uint8.squeeze(-1))
    image_clahe = np.expand_dims(image_clahe, axis=-1)
    return (image_clahe / 255.0).astype(np.float32)

def apply_clahe_tf(img, label):
    img_clahe = tf.numpy_function(apply_clahe_numpy, [img], tf.float32)
    img_clahe.set_shape((IMG_SIZE[0], IMG_SIZE[1], 1))
    return img_clahe, label

# ==========================================
# UTILIDADES DE DATOS
# ==========================================
def get_local_images_and_labels(data_dir):
    valid_exts = ('.png', '.jpg', '.jpeg')
    all_images = [p for p in glob.glob(os.path.join(data_dir, '**', '*.*'), recursive=True) if p.lower().endswith(valid_exts)]
    
    if len(all_images) == 0:
        raise ValueError(f"No se encontraron imágenes en {data_dir}.")
        
    labels = []
    valid_images = []
    for img_path in all_images:
        filename = os.path.basename(img_path)
        try:
            label_str = filename.split('.')[0].split('_')[-1]
            label = int(label_str)
            if label in [0, 1]:
                labels.append(label)
                valid_images.append(img_path)
        except ValueError:
            continue
            
    return valid_images, labels

def check_imbalance(y, split_name):
    total = len(y)
    positives = sum(y)
    negatives = total - positives
    
    weight_for_0 = (1 / negatives) * (total / 2.0)
    weight_for_1 = (1 / positives) * (total / 2.0)
    weight_for_1 = weight_for_1 * 2.0 # Priorizar TB
    
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"[{split_name.upper()}] Sanos: {negatives}, TB: {positives}")
    print(f"   > Pesos calculados: {class_weights}")
    return class_weights

def process_path(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def create_dataset(paths, labels, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(apply_clahe_tf, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ==========================================
# ARQUITECTURA DENSENET
# ==========================================
def build_densenet_classifier(unet_model):
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 1), name="input_image")
    
    # Aumentación Robusta
    x = tf.keras.layers.RandomRotation(0.05)(inputs)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    # Segmentación con U-Net pre-entrenada
    mask = unet_model(x, training=False)
    if isinstance(mask, list): mask = mask[0]
    masked_img = tf.keras.layers.Multiply(name="masked_lungs")([x, mask])

    # Convertir a RGB y escalar
    rgb_img = tf.keras.layers.Lambda(
        lambda t: tf.image.grayscale_to_rgb(t),
        output_shape=(*IMG_SIZE, 3)
    )(masked_img)
    
    rescaled_img = tf.keras.layers.Rescaling(255.0)(rgb_img)

    # NUEVO: Base DenseNet121
    base_model = tf.keras.applications.DenseNet121(
        include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False # Fase 1: Todo congelado
    
    x = base_model(rescaled_img, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="tb_probability")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="DenseNet_TB_Model")

# ==========================================
# LOOP DE ENTRENAMIENTO
# ==========================================
def main():
    print("=============================================================")
    print(" ENTRENAMIENTO DENSENET121: ENSEMBLE CANDIDATE               ")
    print("=============================================================")
    
    # 1. Carga de U-Net Segura
    print("\n[INFO] Cargando modelo U-Net...")
    unet_model = tf.keras.models.load_model(
        UNET_MODEL_PATH, 
        custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coefficient': dice_coefficient},
        safe_mode=False
    )
    unet_model.trainable = False

    # 2. Datos y Split
    print("\n1. Obteniendo datos locales y split 70/30...")
    all_images, labels = get_local_images_and_labels(DATA_DIR)
    X_train, X_val, y_train, y_val = train_test_split(
        all_images, labels, test_size=0.30, stratify=labels, random_state=42
    )
    
    class_weights = check_imbalance(y_train, "Train")
    train_ds = create_dataset(X_train, y_train, is_training=True)
    val_ds = create_dataset(X_val, y_val, is_training=False)

    # 3. Inicializar Modelo
    model = build_densenet_classifier(unet_model)
    
    # FASE 1: Cabeza congelada
    print("\n[FASE 1] Entrenando Clasificador (Base DenseNet congelada)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
    )
    
    model.fit(
        train_ds, validation_data=val_ds, epochs=15, 
        class_weight=class_weights,
        callbacks=[EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True, mode='max')]
    )

    # FASE 2: Fine-Tuning Descongelando el bloque final (conv5_block)
    print("\n[FASE 2] Fine-Tuning (Descongelando Bloque 5 de DenseNet121)...")
    base_model = model.get_layer("densenet121")
    base_model.trainable = True
    
    # Congelamos todo lo que NO sea "conv5_block"
    for layer in base_model.layers:
        if layer.name.startswith("conv5_block"):
            layer.trainable = True
        else:
            layer.trainable = False
            
        # IMPORTANTE: Mantener siempre la Normalización congelada en transfer learning
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Recompilar con LR bajo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
    )
    
    callbacks = [
        ModelCheckpoint(filepath='best_densenet_tb.keras', monitor='val_auc', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_auc', patience=12, restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, min_lr=1e-7, mode='max', verbose=1)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights, callbacks=callbacks)
    print("\n[ÉXITO] Modelo best_densenet_tb.keras guardado.")

if __name__ == '__main__':
    main()
