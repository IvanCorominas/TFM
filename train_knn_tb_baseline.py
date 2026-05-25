import os
import cv2
import glob
import time
import random
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Evita errores en entornos sin interfaz gráfica (headless)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Importar las métricas y funciones de la U-Net necesarias para cargar el modelo pre-entrenado
from unet_model import bce_dice_loss, dice_coefficient

# ==========================================
# CONFIGURACIÓN Y CONSTANTES
# ==========================================
DATA_DIR = os.path.join('.', 'data', 'raw')
UNET_MODEL_PATH = 'best_unet_model.keras'
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EXTERNAL_TEST_SAMPLES_PER_CLASS = 500

# Autenticación automática de Kaggle utilizando el token activo de la sesión del TFM
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_e12a922e07b325050bd7a5dd362c2dcf'
import kagglehub

# ==========================================
# 1. CLAHE PRE-PROCESSING (Equivalente a DL V2)
# ==========================================
def apply_clahe_numpy(image):
    """Aplica CLAHE usando OpenCV a una imagen NumPy float32 en rango [0, 1]."""
    # Escalar a uint8 para OpenCV
    image_uint8 = (image * 255.0).astype(np.uint8)
    
    # Crear CLAHE. Parámetros estándar de tórax: clipLimit=2.0, tileGridSize=(8, 8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # OpenCV espera 2D. La imagen tiene forma (256, 256, 1)
    image_clahe = clahe.apply(image_uint8.squeeze(-1))
    
    # Recuperar dimensiones y escalar de vuelta a float32 [0, 1]
    image_clahe = np.expand_dims(image_clahe, axis=-1)
    return (image_clahe / 255.0).astype(np.float32)

def apply_clahe_tf(img, label):
    """Envoltura de TensorFlow para ejecutar código NumPy en el pipeline tf.data."""
    img_clahe = tf.numpy_function(apply_clahe_numpy, [img], tf.float32)
    img_clahe.set_shape((IMG_SIZE[0], IMG_SIZE[1], 1))
    return img_clahe, label

def process_path(img_path, label):
    """Carga y redimensiona la imagen."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# ==========================================
# 2. CARGA Y DESCARGA DE DATOS
# ==========================================
def get_local_images_and_labels(data_dir):
    """Busca todas las imágenes en data/raw recursivamente y deduce la etiqueta del sufijo (_0 o _1)."""
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

def get_external_test_data(limit_per_class=EXTERNAL_TEST_SAMPLES_PER_CLASS):
    """Descarga de forma segura el conjunto de pruebas externo de Kaggle y selecciona una muestra balanceada."""
    print("\n[INFO] Descargando/Verificando dataset externo desde Kaggle...")
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
        print("[ADVERTENCIA] No se encontraron imágenes en las carpetas de Kaggle.")
        return [], []

    random.seed(42)
    sample_n = random.sample(normal_images, min(limit_per_class, len(normal_images)))
    sample_tb = random.sample(tb_images, min(limit_per_class, len(tb_images)))
    
    test_paths = sample_n + sample_tb
    test_labels = [0]*len(sample_n) + [1]*len(sample_tb)
    return test_paths, test_labels

# ==========================================
# 3. EXTRACCIÓN Y APLANAMIENTO DE CARACTERÍSTICAS
# ==========================================
def extract_preprocessed_features(paths, labels, unet_model, batch_size=BATCH_SIZE):
    """
    Carga imágenes, aplica CLAHE, predice máscaras pulmonares con la U-Net,
    multiplica el canal CLAHE por la máscara obtenida, y aplana a un vector 1D de 65,536 características.
    """
    # Crear pipeline tf.data para procesamiento eficiente en memoria
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(apply_clahe_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    features = []
    extracted_labels = []
    
    for imgs, lbls in ds:
        # Predecir máscara del pulmón
        masks = unet_model(imgs, training=False)
        if isinstance(masks, list):
            masks = masks[0]
            
        # Multiplicación para aislar pulmones (rango [0, 1] en escala de grises)
        masked_imgs = imgs * masks
        
        # Aplanar cada imagen de 256 x 256 x 1 a un vector 1D continuo de 65,536 valores
        # Shape original: (Batch_Size, 256, 256, 1) -> Shape final: (Batch_Size, 65536)
        flat_imgs = tf.reshape(masked_imgs, (tf.shape(masked_imgs)[0], -1))
        
        features.append(flat_imgs.numpy())
        extracted_labels.append(lbls.numpy())
        
    X = np.concatenate(features, axis=0)
    y = np.concatenate(extracted_labels, axis=0)
    return X, y

# ==========================================
# LOOP PRINCIPAL DE EJECUCIÓN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Entrena y evalúa un modelo KNN Baseline con validación cruzada y test externo.")
    parser.add_argument("--dry-run", action="store_true", help="Ejecuta una prueba rápida con datos reducidos para depuración.")
    args = parser.parse_args()

    print("=============================================================")
    print("  EXPERIMENTO BASELINE CLASSIC ML: K-NEAREST NEIGHBORS (KNN) ")
    print("=============================================================")

    # 1. Cargar el modelo U-Net
    print(f"\n[INFO] Cargando modelo U-Net desde {UNET_MODEL_PATH}...")
    if not os.path.exists(UNET_MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo U-Net en {UNET_MODEL_PATH}")
        
    unet_model = tf.keras.models.load_model(
        UNET_MODEL_PATH, 
        custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coefficient': dice_coefficient},
        safe_mode=False
    )
    unet_model.trainable = False

    # 2. Descargar / Obtener rutas del Dataset Externo (Kaggle)
    limit_per_class = EXTERNAL_TEST_SAMPLES_PER_CLASS
    if args.dry_run:
        print("\n[MODO DRY-RUN ACTIVADO] Configurando ejecución mínima...")
        limit_per_class = 10

    t_paths, t_labels = get_external_test_data(limit_per_class=limit_per_class)
    if not t_paths:
        print("[ERROR] Error al obtener el dataset externo. Deteniendo ejecución.")
        return

    # 3. Cargar y Split del Dataset Local (70/30 estratificado)
    print("\n[INFO] Obteniendo rutas locales y dividiendo 70/30...")
    all_images, labels = get_local_images_and_labels(DATA_DIR)
    
    if args.dry_run:
        # Reducir drásticamente la muestra para el dry-run
        indices = np.random.choice(len(all_images), 30, replace=False)
        all_images = [all_images[i] for i in indices]
        labels = [labels[i] for i in indices]

    X_train_paths, X_val_paths, y_train_paths, y_val_paths = train_test_split(
        all_images, labels, test_size=0.30, stratify=labels, random_state=42
    )
    
    print(f"  - Muestras Entrenamiento Local: {len(X_train_paths)}")
    print(f"  - Muestras Validación Local:   {len(X_val_paths)}")
    print(f"  - Muestras Test Externo:       {len(t_paths)}")

    # 4. Extraer Características Preprocesadas y Aplanadas (U-Net + CLAHE -> Flatten)
    print("\n[INFO] Extrayendo características del conjunto de Entrenamiento Local...")
    X_train, y_train = extract_preprocessed_features(X_train_paths, y_train_paths, unet_model)
    
    print("\n[INFO] Extrayendo características del conjunto de Validación Local...")
    X_val, y_val = extract_preprocessed_features(X_val_paths, y_val_paths, unet_model)
    
    print("\n[INFO] Extrayendo características del conjunto de Test Externo (Kaggle)...")
    X_ext, y_ext = extract_preprocessed_features(t_paths, t_labels, unet_model)

    print(f"\n[INFO] Dimensiones de los conjuntos aplanados (Características = 65,536):")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_val shape:   {X_val.shape}")
    print(f"  - X_ext shape:   {X_ext.shape}")

    # 5. Optimización de Hiperparámetros (GridSearchCV)
    print("\n[INFO] Configurando GridSearchCV para KNN...")
    # Parámetros a optimizar
    n_neighbors_list = [3, 5, 11, 21] if not args.dry_run else [3, 5]
    metrics_list = ['euclidean', 'manhattan']
    
    param_grid = {
        'n_neighbors': n_neighbors_list,
        'metric': metrics_list
    }
    
    knn = KNeighborsClassifier()
    # cv=3 como requiere la tarea, scoring='accuracy'
    cv_folds = 3 if not args.dry_run else 2
    grid_search = GridSearchCV(knn, param_grid, cv=cv_folds, scoring='accuracy', verbose=2, n_jobs=-1)

    # 6. Entrenamiento y Eficiencia Temporal
    print("\n[INFO] Iniciando el entrenamiento del KNN (Búsqueda por Rejilla)...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    
    print("\n=============================================================")
    print(" RESULTADOS DE LA OPTIMIZACIÓN (GRIDSEARCH)")
    print("=============================================================")
    print(f"  - Mejores Hiperparámetros: {grid_search.best_params_}")
    print(f"  - Mejor puntuación de Accuracy en CV (k={cv_folds}): {grid_search.best_score_:.4f}")
    print(f"  - Tiempo de entrenamiento / Búsqueda: {training_time:.2f} segundos")

    # Evaluar en validación local
    best_model = grid_search.best_estimator_
    val_acc = best_model.score(X_val, y_val)
    print(f"  - Accuracy sobre el set de Validación Local: {val_acc:.4f}")

    # Medir espacio físico del modelo serializado
    model_filename = 'knn_baseline_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    
    model_size_bytes = os.path.getsize(model_filename)
    model_size_kb = model_size_bytes / 1024.0
    model_size_mb = model_size_kb / 1024.0
    print(f"  - Espacio ocupado por el modelo guardado (.pkl): {model_size_bytes:,} bytes ({model_size_kb:.2f} KB / {model_size_mb:.4f} MB)")
    print("=============================================================")

    # 7. Evaluación en Dataset Externo
    print("\n[INFO] Realizando predicción sobre el conjunto de test externo de Kaggle...")
    y_ext_pred = best_model.predict(X_ext)
    
    print("\n=============================================================")
    print(" REPORTE DE CLASIFICACIÓN (TEST EXTERNO - KAGGLE)")
    print("=============================================================")
    print(classification_report(y_ext, y_ext_pred, target_names=["Sano (0)", "Tuberculosis (1)"]))
    print("=============================================================")

    # Generar y guardar la matriz de confusión
    cm = confusion_matrix(y_ext, y_ext_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Sano (Pred)", "Tuberculosis (Pred)"], 
                yticklabels=["Sano (Real)", "Tuberculosis (Real)"])
    plt.title(f'Matriz de Confusión Externa - KNN Baseline (k={best_model.n_neighbors}, {best_model.metric})')
    plt.tight_layout()
    plt.savefig('knn_baseline_cm.png', dpi=300)
    print("\n[EXITO] Matriz de confusión guardada como 'knn_baseline_cm.png'")
    
    # Análisis comparativo rápido en consola
    print("\n-------------------------------------------------------------")
    print(" ANÁLISIS COMPARATIVO DE COMPLEJIDAD (ML CLÁSICO VS DEEP LEARNING)")
    print("-------------------------------------------------------------")
    print(f"  - KNN Baseline: Tiempo de Entrenamiento = {training_time:.2f} seg | Espacio Modelo = {model_size_mb:.4f} MB")
    print(f"  - Deep Learning (CNN/EfficientNet/DenseNet): Tiempo = Horas | Espacio Modelo = ~380-420 MB")
    print("  Nota: Consulta el walkthrough.md para ver un desglose detallado.")
    print("-------------------------------------------------------------")

if __name__ == '__main__':
    main()
