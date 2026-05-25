import os
import matplotlib
matplotlib.use('Agg') # CRÍTICO: Evita errores de visualización en terminal/entornos sin display

# Autenticación automática de Kaggle usando el token proporcionado
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_e12a922e07b325050bd7a5dd362c2dcf'
import kagglehub

import glob
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Importar las métricas de la U-Net necesarias para cargar el modelo pre-entrenado
from unet_model import bce_dice_loss, dice_coefficient

# ==========================================
# HIPERPARÁMETROS Y CONFIGURACIÓN
# ==========================================
DATA_DIR = os.path.join('.', 'data', 'raw')
UNET_MODEL_PATH = 'best_unet_model.keras'
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 40 # Incrementado para dar más margen al fine-tuning
EXTERNAL_TEST_SAMPLES_PER_CLASS = 500

def get_local_images_and_labels(data_dir):
    """Busca imágenes locales y extrae etiquetas basándose en el patrón _0 o _1."""
    valid_exts = ('.png', '.jpg', '.jpeg')
    all_images = [p for p in glob.glob(os.path.join(data_dir, '**', '*.*'), recursive=True) if p.lower().endswith(valid_exts)]
    
    if len(all_images) == 0:
        raise ValueError(f"No se encontraron imágenes en {data_dir}. Revisa la ruta.")
        
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
    """Calcula pesos de clase para priorizar la clase TB (1)."""
    total = len(y)
    positives = sum(y)
    negatives = total - positives
    
    weight_for_0 = (1 / negatives) * (total / 2.0)
    weight_for_1 = (1 / positives) * (total / 2.0)
    
    # AGRESIVO: Damos el doble de importancia a la clase TB para evitar Falsos Negativos
    weight_for_1 = weight_for_1 * 2.0
    
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"[{split_name.upper()}] Sanos: {negatives}, TB: {positives}")
    print(f"   > Pesos calculados: {class_weights}")
    return class_weights

def process_path(img_path, label):
    """Carga y procesa la imagen para el dataset."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    """Aumentación básica (el modelo tendrá capas de aumentación integradas para mayor robustez)."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def create_dataset(paths, labels, is_training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def get_external_test_data(limit_per_class=EXTERNAL_TEST_SAMPLES_PER_CLASS):
    """Descarga de Kaggle con comprobaciones de seguridad."""
    print("\n[INFO] Descargando dataset externo desde Kaggle...")
    try:
        path = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
    except Exception as e:
        print(f"Error al descargar de Kaggle: {e}")
        return [], []

    normal_images = []
    tb_images = []
    valid_exts = ('.png', '.jpg', '.jpeg')
    
    for root, dirs, files in os.walk(path):
        folder_name = os.path.basename(root).lower()
        if 'normal' in folder_name:
            normal_images.extend([os.path.join(root, f) for f in files if f.lower().endswith(valid_exts)])
        elif 'tuberculosis' in folder_name:
            tb_images.extend([os.path.join(root, f) for f in files if f.lower().endswith(valid_exts)])
    
    if not normal_images or not tb_images:
        print("⚠️ No se encontraron imágenes en las carpetas esperadas de Kaggle.")
        return [], []

    random.seed(42)
    sample_n = random.sample(normal_images, min(limit_per_class, len(normal_images)))
    sample_tb = random.sample(tb_images, min(limit_per_class, len(tb_images)))
    
    test_paths = sample_n + sample_tb
    test_labels = [0]*len(sample_n) + [1]*len(sample_tb)
    return test_paths, test_labels

def build_end_to_end_classifier():
    """Modelo modular con Aumentación integrada y U-Net."""
    print("\n[INFO] Cargando modelo U-Net...")
    unet_model = tf.keras.models.load_model(
        UNET_MODEL_PATH, 
        custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coefficient': dice_coefficient}
    )
    unet_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 1), name="input_image")
    
    # 1. Aumentación Robusta (solo activa en entrenamiento)
    x = tf.keras.layers.RandomRotation(0.05)(inputs)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    # 2. Segmentación y Enmascaramiento
    mask = unet_model(x, training=False)
    if isinstance(mask, list): mask = mask[0]
    masked_img = tf.keras.layers.Multiply(name="masked_lungs")([x, mask])

    # 3. Preparación para EfficientNet (Grayscale -> RGB -> Rescale)
    rgb_img = tf.keras.layers.Lambda(
        lambda x: tf.image.grayscale_to_rgb(x),
        output_shape=(*IMG_SIZE, 3)
    )(masked_img)
    rescaled_img = tf.keras.layers.Rescaling(255.0)(rgb_img)

    # 4. Clasificador Base
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    x = base_model(rescaled_img, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="tb_probability")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="Final_TB_Pipeline")

def main():
    print("=============================================================")
    print(" PIPELINE TB MEJORADO: ALTA PRECISIÓN Y TEST EXTERNO ROBUSTO ")
    print("=============================================================")
    
    # 1. Datos Locales
    all_images, labels = get_local_images_and_labels(DATA_DIR)
    X_train, X_val, y_train, y_val = train_test_split(
        all_images, labels, test_size=0.30, stratify=labels, random_state=42
    )
    
    class_weights = check_imbalance(y_train, "Train")
    train_ds = create_dataset(X_train, y_train, is_training=True)
    val_ds = create_dataset(X_val, y_val, is_training=False)

    # 2. Modelo
    model = build_end_to_end_classifier()
    
    # Fase 1: Entrenamiento de Cabeza
    print("\n[FASE 1] Entrenando Clasificador (Base congelada)...")
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

    # Fase 2: Fine-Tuning (Descongelando SOLO el final de EfficientNet)
    print("\n[FASE 2] Fine-Tuning (Descongelando últimos bloques)...")
    
    # 1. Accedemos a la capa base
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True
    
    # 2. Congelamos todo EXCEPTO los bloques 6 y 7 (las capas más profundas)
    for layer in base_model.layers:
        if not layer.name.startswith("block6") and not layer.name.startswith("block7") and not layer.name.startswith("top"):
            layer.trainable = False
        
        # Mantenemos BatchNormalization congelado (CRÍTICO para transfer learning)
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # 3. Re-compilamos con un Learning Rate mucho más bajo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Aún más bajo para no romper pesos
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
    )
    
    callbacks = [
        ModelCheckpoint(filepath='best_tb_pipeline.keras', monitor='val_auc', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_auc', patience=12, restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, min_lr=1e-7, mode='max', verbose=1)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights, callbacks=callbacks)
    
    # 3. Evaluación Externa
    print("\n[EVALUACIÓN] Iniciando Test Externo...")
    t_paths, t_labels = get_external_test_data()
    if not t_paths: return
    
    ext_ds = create_dataset(t_paths, t_labels, is_training=False)
    
    # En lugar de load_model (que falla por deserializar la Lambda en algunas versiones de Keras),
    # construimos el modelo e inyectamos los pesos guardados por el checkpoint.
    best_model = build_end_to_end_classifier()
    best_model.load_weights('best_tb_pipeline.keras')
    
    # Keras requiere compilar el modelo antes de llamar a .evaluate()
    best_model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
    )
    
    print("\n--- Resultados Test Externo (Kaggle) ---")
    metrics = best_model.evaluate(ext_ds, verbose=1)
    for name, val in zip(best_model.metrics_names, metrics):
        print(f"   > {name}: {val:.4f}")

    # Priorización de Recall: Umbral 0.35
    print("\n7. Generando Matriz de Confusión (Umbral 0.35 para Priorizar Recall)...")
    probs = best_model.predict(ext_ds, verbose=1)
    THRESHOLD = 0.35 
    y_pred = (probs > THRESHOLD).astype(int).flatten()
    y_true = np.array(t_labels)
    
    print("\nReporte de Clasificación (Test Externo):")
    print(classification_report(y_true, y_pred, target_names=["Normal (0)", "TB (1)"]))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Normal (Pred)", "TB (Pred)"], yticklabels=["Normal (Real)", "TB (Real)"])
    plt.title(f'Matriz de Confusión Externa (Threshold {THRESHOLD})')
    plt.tight_layout()
    plt.savefig('external_test_confusion_matrix.png', dpi=300)
    print("   > Archivo 'external_test_confusion_matrix.png' creado.")

if __name__ == '__main__':
    main()
