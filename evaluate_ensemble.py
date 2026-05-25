import os
import cv2
import matplotlib
matplotlib.use('Agg')

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_e12a922e07b325050bd7a5dd362c2dcf'

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import kagglehub

from unet_model import bce_dice_loss, dice_coefficient

# Importar constructores de los scripts de entrenamiento
from train_tb_pipeline_v2 import build_end_to_end_classifier as build_effnet
from train_densenet_tb import build_densenet_classifier as build_densenet

# ==========================================
# CONFIGURACIÓN
# ==========================================
UNET_MODEL_PATH = 'best_unet_model.keras'
EFFNET_WEIGHTS = 'best_tb_pipeline_v2.keras'
DENSENET_WEIGHTS = 'best_densenet_tb.keras'
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EXTERNAL_TEST_SAMPLES_PER_CLASS = 500

# ==========================================
# PRE-PROCESAMIENTO CLAHE
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
# LOOP DE EVALUACIÓN
# ==========================================
def main():
    print("=============================================================")
    print(" ENSEMBLE EVALUATION: EFFICIENTNET-B0 + DENSENET121          ")
    print("=============================================================")
    
    # 1. Cargar Datos
    t_paths, t_labels = get_external_test_data()
    if not t_paths:
        return
    ext_ds = create_dataset(t_paths, t_labels)
    y_true = np.array(t_labels)
    
    # 2. Cargar U-Net
    print("\n[INFO] Cargando modelo U-Net...")
    unet_model = tf.keras.models.load_model(
        UNET_MODEL_PATH, 
        custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coefficient': dice_coefficient},
        safe_mode=False
    )
    unet_model.trainable = False

    # 3. Cargar EfficientNetB0 (V2)
    print(f"\n[INFO] Inicializando EfficientNetB0 e inyectando {EFFNET_WEIGHTS}...")
    model_effnet = build_effnet(unet_model)
    model_effnet.load_weights(EFFNET_WEIGHTS)
    print("   > Prediciendo con EfficientNetB0...")
    probs_effnet = model_effnet.predict(ext_ds, verbose=1)

    # 4. Cargar DenseNet121
    print(f"\n[INFO] Inicializando DenseNet121 e inyectando {DENSENET_WEIGHTS}...")
    model_densenet = build_densenet(unet_model)
    model_densenet.load_weights(DENSENET_WEIGHTS)
    print("   > Prediciendo con DenseNet121...")
    probs_densenet = model_densenet.predict(ext_ds, verbose=1)

    # 5. Ensemble: Soft Voting (Media)
    print("\n[ENSEMBLE] Fusionando predicciones (Media Aritmética)...")
    ensemble_probs = (probs_effnet + probs_densenet) / 2.0

 # 6. Aplicar Umbral Clínico de Triaje
    print("\n[TRIAJE] Descartando Youden para aplicar Umbral Clínico agresivo...")
    
    # Imponemos el umbral manualmente (0.25) para vaciar los Falsos Negativos
    optimal_threshold = 0.50 
    
    print(f"   > ¡UMBRAL CLÍNICO DEL ENSAMBLE FIJADO EN!: {optimal_threshold:.4f}")
    
    # 7. Evaluación Final
    y_pred = (ensemble_probs >= optimal_threshold).astype(int).flatten()
    
    print("\nReporte de Clasificación (Ensemble - Test Externo Kaggle):")
    print(classification_report(y_true, y_pred, target_names=["Sanos (0)", "Tuberculosis (1)"]))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=["Sanos (Pred)", "TB (Pred)"], yticklabels=["Sanos (Real)", "TB (Real)"])
    plt.title(f'Ensemble CM (EffNet+DenseNet) | Threshold: {optimal_threshold:.4f}')
    plt.xlabel('Predicción del Ensamble')
    plt.ylabel('Diagnóstico Real')
    plt.tight_layout()
    plt.savefig('ensemble_test_cm.png', dpi=300)
    print("   > Guardada matriz combinada: 'ensemble_test_cm.png'")

if __name__ == '__main__':
    main()
