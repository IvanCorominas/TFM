import os
import cv2
import matplotlib
matplotlib.use('Agg')

# Autenticación automática de Kaggle
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_e12a922e07b325050bd7a5dd362c2dcf'
import kagglehub

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Importar las métricas de la U-Net necesarias para cargar el modelo pre-entrenado
from unet_model import bce_dice_loss, dice_coefficient

# ==========================================
# CONFIGURACIÓN
# ==========================================
UNET_MODEL_PATH = 'best_unet_model.keras'
PIPELINE_MODEL_PATH = 'best_tb_pipeline_v2.keras' # El modelo entrenado en la V2
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EXTERNAL_TEST_SAMPLES_PER_CLASS = 500
CLINICAL_THRESHOLD = 0.20 # Prioridad máxima: Reducir Falsos Negativos (Alta Sensibilidad)

# ==========================================
# PRE-PROCESAMIENTO (Igual que V2)
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

def process_path(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def create_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(apply_clahe_tf, num_parallel_calls=tf.data.AUTOTUNE)
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
    
    random.seed(42)
    sample_n = random.sample(normal_images, min(limit_per_class, len(normal_images)))
    sample_tb = random.sample(tb_images, min(limit_per_class, len(tb_images)))
    
    test_paths = sample_n + sample_tb
    test_labels = [0]*len(sample_n) + [1]*len(sample_tb)
    return test_paths, test_labels

# ==========================================
# ARQUITECTURA DEL MODELO (Igual que V2)
# ==========================================
def build_end_to_end_classifier(unet_model):
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 1), name="input_image")
    
    mask = unet_model(inputs, training=False)
    if isinstance(mask, list): mask = mask[0]
    masked_img = tf.keras.layers.Multiply(name="masked_lungs")([inputs, mask])

    rgb_img = tf.keras.layers.Lambda(
        lambda t: tf.image.grayscale_to_rgb(t),
        output_shape=(*IMG_SIZE, 3)
    )(masked_img)
    
    rescaled_img = tf.keras.layers.Rescaling(255.0)(rgb_img)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    x = base_model(rescaled_img, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="tb_probability")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="Final_TB_Pipeline_V2")

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
def main():
    print("=============================================================")
    print(" EVALUACIÓN DE TRIAJE CLÍNICO (THRESHOLD AGRESIVO)           ")
    print("=============================================================")
    
    # 1. Carga de Datos
    t_paths, t_labels = get_external_test_data()
    if not t_paths:
        return
    ext_ds = create_dataset(t_paths, t_labels)
    
    # 2. Carga de la U-Net
    print("\n[INFO] Cargando modelo U-Net...")
    unet_model = tf.keras.models.load_model(
        UNET_MODEL_PATH, 
        custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coefficient': dice_coefficient},
        safe_mode=False
    )
    unet_model.trainable = False
    
    # 3. Reconstrucción del Clasificador e Inyección de Pesos
    print(f"\n[INFO] Cargando pesos del pipeline entrenado ({PIPELINE_MODEL_PATH})...")
    best_model = build_end_to_end_classifier(unet_model)
    best_model.load_weights(PIPELINE_MODEL_PATH)
    
    # 4. Inferencia
    print("\n[EVALUACIÓN] Calculando probabilidades (model.predict)...")
    probs = best_model.predict(ext_ds, verbose=1)
    y_true = np.array(t_labels)
    
    # 5. Aplicación del Umbral Clínico (Screening)
    print(f"\n   > Aplicando UMBRAL CLÍNICO AGRESIVO: {CLINICAL_THRESHOLD}")
    y_pred = (probs >= CLINICAL_THRESHOLD).astype(int).flatten()
    
    # 6. Reportes y Matrices
    print("\nReporte de Clasificación (Triaje Clínico):")
    print(classification_report(y_true, y_pred, target_names=["Sanos (0)", "Tuberculosis (1)"]))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=["Sanos (Pred)", "TB (Pred)"], yticklabels=["Sanos (Real)", "TB (Real)"])
    plt.title(f'Matriz de Confusión - Triaje (Threshold: {CLINICAL_THRESHOLD})')
    plt.xlabel('Predicción del Modelo')
    plt.ylabel('Verdadero Diagnóstico')
    plt.tight_layout()
    plt.savefig('clinical_threshold_cm.png', dpi=300)
    print("   > Guardada matriz final: 'clinical_threshold_cm.png'")

if __name__ == '__main__':
    main()
