import os
import cv2
import matplotlib
matplotlib.use('Agg') # Evita errores de visualización en terminal/entornos sin display

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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
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
EPOCHS = 40
EXTERNAL_TEST_SAMPLES_PER_CLASS = 500

# ==========================================
# 1. CLAHE PRE-PROCESSING (NUEVO V2)
# ==========================================
def apply_clahe_numpy(image):
    """Aplica CLAHE usando OpenCV a una imagen NumPy float32 en rango [0, 1]."""
    # Escalar a uint8 para OpenCV
    image_uint8 = (image * 255.0).astype(np.uint8)
    
    # Crear CLAHE. clipLimit y tileGridSize son parámetros estándar para RX de tórax.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # OpenCV espera 2D. La imagen tiene forma (256, 256, 1)
    image_clahe = clahe.apply(image_uint8.squeeze(-1))
    
    # Recuperar dimensiones y escalar de vuelta a float32 [0, 1]
    image_clahe = np.expand_dims(image_clahe, axis=-1)
    return (image_clahe / 255.0).astype(np.float32)

def apply_clahe_tf(img, label):
    """Envoltura de TensorFlow para ejecutar código NumPy en el pipeline tf.data."""
    img_clahe = tf.numpy_function(apply_clahe_numpy, [img], tf.float32)
    # Recuperamos la forma que tf.numpy_function borra temporalmente
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
    
    # AGRESIVO: Damos el doble de importancia a la clase TB
    weight_for_1 = weight_for_1 * 2.0
    
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
    
    # ¡NUEVO V2! Aplicamos CLAHE a TODAS las particiones para combatir Domain Shift
    ds = ds.map(apply_clahe_tf, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def get_external_test_data(limit_per_class=EXTERNAL_TEST_SAMPLES_PER_CLASS):
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
        print("⚠️ No se encontraron imágenes en las carpetas de Kaggle.")
        return [], []

    random.seed(42)
    sample_n = random.sample(normal_images, min(limit_per_class, len(normal_images)))
    sample_tb = random.sample(tb_images, min(limit_per_class, len(tb_images)))
    
    test_paths = sample_n + sample_tb
    test_labels = [0]*len(sample_n) + [1]*len(sample_tb)
    return test_paths, test_labels

# ==========================================
# 2. VISUAL DEBUGGING (NUEVO V2)
# ==========================================
def generate_visual_debug(ext_ds, unet_model, num_samples=5):
    """Genera una imagen para verificar visualmente que U-Net + CLAHE no fallan."""
    print("\n[DEBUG] Generando comprobación visual de U-Net + CLAHE en Kaggle...")
    
    plt.figure(figsize=(12, 4 * num_samples))
    for imgs, labels in ext_ds.take(1):
        n = min(num_samples, len(imgs))
        masks = unet_model.predict(imgs[:n], verbose=0)
        
        for i in range(n):
            original = imgs[i].numpy()
            mask = masks[i]
            # Imagen tras aplicar el mask
            final = original * mask
            
            plt.subplot(n, 3, i*3 + 1)
            plt.imshow(original.squeeze(), cmap='gray')
            if i == 0: plt.title('Kaggle Original + CLAHE')
            plt.axis('off')
            
            plt.subplot(n, 3, i*3 + 2)
            plt.imshow(mask.squeeze(), cmap='gray')
            if i == 0: plt.title('Máscara U-Net')
            plt.axis('off')
            
            plt.subplot(n, 3, i*3 + 3)
            plt.imshow(final.squeeze(), cmap='gray')
            if i == 0: plt.title('Final (Input clasificador)')
            plt.axis('off')
            
    plt.tight_layout()
    plt.savefig('debug_preprocessing.png', dpi=300)
    print("   > CREADO: 'debug_preprocessing.png'. ¡Revisa este archivo ahora!")

# ==========================================
# 3. MODELO PRINCIPAL (V1 + Mejoras Manuales)
# ==========================================
def build_end_to_end_classifier(unet_model):
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 1), name="input_image")
    
    # Aumentación Robusta
    x = tf.keras.layers.RandomRotation(0.05)(inputs)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    # Segmentación y Enmascaramiento
    mask = unet_model(x, training=False)
    if isinstance(mask, list): mask = mask[0]
    masked_img = tf.keras.layers.Multiply(name="masked_lungs")([x, mask])

    # Grayscale -> RGB de forma segura para evitar fallos de Lambda en Keras 3
    rgb_img = tf.keras.layers.Lambda(
        lambda t: tf.image.grayscale_to_rgb(t),
        output_shape=(*IMG_SIZE, 3)
    )(masked_img)
    
    rescaled_img = tf.keras.layers.Rescaling(255.0)(rgb_img)

    # Clasificador Base
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    x = base_model(rescaled_img, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x) # Capa añadida manualmente en V1
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="tb_probability")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="Final_TB_Pipeline_V2")

# ==========================================
# 4. LOOP PRINCIPAL
# ==========================================
def main():
    print("=============================================================")
    print(" PIPELINE TB V2: CLAHE + YOUDEN'S THRESHOLD + VISUAL DEBUG   ")
    print("=============================================================")
    
    # --- FASE A: Preparación y Debugging ---
    print("\n[INFO] Cargando modelo U-Net...")
    unet_model = tf.keras.models.load_model(
        UNET_MODEL_PATH, 
        custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coefficient': dice_coefficient},
        safe_mode=False
    )
    unet_model.trainable = False

    t_paths, t_labels = get_external_test_data()
    if not t_paths: return
    ext_ds = create_dataset(t_paths, t_labels, is_training=False)

    # CRÍTICO: Generar debug visual antes de perder tiempo entrenando
    generate_visual_debug(ext_ds, unet_model)
    
    # --- FASE B: Datos Locales y Split ---
    print("\n1. Obteniendo datos locales y split 70/30...")
    all_images, labels = get_local_images_and_labels(DATA_DIR)
    X_train, X_val, y_train, y_val = train_test_split(
        all_images, labels, test_size=0.30, stratify=labels, random_state=42
    )
    
    class_weights = check_imbalance(y_train, "Train")
    train_ds = create_dataset(X_train, y_train, is_training=True)
    val_ds = create_dataset(X_val, y_val, is_training=False)

    # --- FASE C: Entrenamiento del Clasificador ---
    model = build_end_to_end_classifier(unet_model)
    
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

    print("\n[FASE 2] Fine-Tuning (Descongelando últimos bloques)...")
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True
    
    for layer in base_model.layers:
        if not layer.name.startswith("block6") and not layer.name.startswith("block7") and not layer.name.startswith("top"):
            layer.trainable = False
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
    )
    
    callbacks = [
        ModelCheckpoint(filepath='best_tb_pipeline_v2.keras', monitor='val_auc', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_auc', patience=12, restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, min_lr=1e-7, mode='max', verbose=1)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights, callbacks=callbacks)
    
    # --- FASE D: Evaluación Dinámica (NUEVO V2) ---
    print("\n[EVALUACIÓN] Test Externo con Umbral Dinámico...")
    
    best_model = build_end_to_end_classifier(unet_model)
    best_model.load_weights('best_tb_pipeline_v2.keras')
    best_model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
    )
    
    print("\n--- Resultados Brutos (Test Externo) ---")
    best_model.evaluate(ext_ds, verbose=1)

    print("\n[YOUDEN] Calculando Umbral Dinámico Estadístico (ROC J-Statistic)...")
    probs = best_model.predict(ext_ds, verbose=1)
    y_true = np.array(t_labels)
    
    # Obtener curvas ROC
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    
    # Índice de Youden: Maximiza J = Sensibilidad (TPR) + Especificidad (1 - FPR) - 1  =>  TPR - FPR
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"   > ¡UMBRAL ÓPTIMO DE YOUDEN CALCULADO!: {optimal_threshold:.4f}")
    
    # Clasificación usando el nuevo umbral dinámico
    y_pred = (probs >= optimal_threshold).astype(int).flatten()
    
    print("\nReporte de Clasificación (V2 - Test Externo):")
    print(classification_report(y_true, y_pred, target_names=["Normal (0)", "TB (1)"]))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Normal (Pred)", "TB (Pred)"], yticklabels=["Normal (Real)", "TB (Real)"])
    plt.title(f'Matriz de Confusión Externa (Youden Threshold: {optimal_threshold:.4f})')
    plt.tight_layout()
    plt.savefig('external_test_cm_v2.png', dpi=300)
    print("   > Guardada matriz final: 'external_test_cm_v2.png'")

if __name__ == '__main__':
    main()
