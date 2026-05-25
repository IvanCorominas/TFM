import os
import cv2
import glob
import random
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # Evita errores de visualización en entornos sin display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Importar las métricas de la U-Net necesarias para cargar el modelo pre-entrenado
from unet_model import bce_dice_loss, dice_coefficient

# ==========================================
# CONFIGURACIÓN Y CONSTANTES
# ==========================================
DATA_DIR = os.path.join('.', 'data', 'raw')
UNET_MODEL_PATH = 'best_unet_model.keras'
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 60
EXTERNAL_TEST_SAMPLES_PER_CLASS = 500

# Autenticación automática de Kaggle usando el token provisto
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_e12a922e07b325050bd7a5dd362c2dcf'
import kagglehub

# ==========================================
# 1. CLAHE PRE-PROCESSING
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
    img_clahe.set_shape((IMG_SIZE[0], IMG_SIZE[1], 1))
    return img_clahe, label

# ==========================================
# UTILIDADES DE DATOS
# ==========================================
def get_local_images_and_labels(data_dir):
    """Busca todas las imágenes en data/raw recursivamente y deduce la etiqueta."""
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
    """Calcula y muestra la distribución de clases, retornando los class_weights correspondientes."""
    total = len(y)
    positives = sum(y)
    negatives = total - positives
    
    weight_for_0 = (1 / negatives) * (total / 2.0)
    weight_for_1 = (1 / positives) * (total / 2.0)
    
    # Para ser coherentes con el experimento anterior, priorizamos la clase TB
    weight_for_1 = weight_for_1 * 2.0
    
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"[{split_name.upper()}] Sanos (0): {negatives} | TB (1): {positives}")
    print(f"   > Pesos calculados: {class_weights}")
    return class_weights

def process_path(img_path, label):
    """Carga y redimensiona la imagen."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    """Aplica aumentación básica de datos."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def create_dataset(paths, labels, batch_size=BATCH_SIZE, is_training=False):
    """Crea un dataset tf.data.Dataset con CLAHE pre-processing."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(apply_clahe_tf, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=len(paths))
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def get_external_test_data(limit_per_class=EXTERNAL_TEST_SAMPLES_PER_CLASS):
    """Descarga de forma segura el conjunto de pruebas externo desde Kaggle."""
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
        print("[ADVERTENCIA] No se encontraron imagenes en las carpetas de Kaggle.")
        return [], []

    random.seed(42)
    sample_n = random.sample(normal_images, min(limit_per_class, len(normal_images)))
    sample_tb = random.sample(tb_images, min(limit_per_class, len(tb_images)))
    
    test_paths = sample_n + sample_tb
    test_labels = [0]*len(sample_n) + [1]*len(sample_tb)
    return test_paths, test_labels

# ==========================================
# 2. CONSTRUCCIÓN DE LA CNN CUSTOM DESDE CERO
# ==========================================
def build_custom_cnn_classifier(unet_model):
    """
    Construye un modelo End-to-End que integra la U-Net congelada para la segmentación,
    y a continuación una CNN propia de 4 bloques diseñada desde cero en escala de grises.
    """
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 1), name="input_image")
    
    # 1. Aumentación de Datos espacial idéntica a experimentos previos
    x = tf.keras.layers.RandomRotation(0.05)(inputs)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    # 2. Segmentación con U-Net pre-entrenada
    mask = unet_model(x, training=False)
    if isinstance(mask, list): 
        mask = mask[0]
        
    # Multiplicación para aislar los pulmones (rango [0, 1] en escala de grises)
    masked_img = tf.keras.layers.Multiply(name="masked_lungs")([x, mask])

    # 3. Arquitectura CNN Custom from scratch (Grayscale - 1 canal)
    # Bloque Convolucional 1: 32 filtros
    xc = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', 
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="custom_conv1"
    )(masked_img)
    xc = tf.keras.layers.BatchNormalization(name="custom_bn1")(xc)
    xc = tf.keras.layers.Activation('relu', name="custom_relu1")(xc)
    xc = tf.keras.layers.MaxPooling2D((2, 2), name="custom_pool1")(xc)

    # Bloque Convolucional 2: 64 filtros
    xc = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', 
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="custom_conv2"
    )(xc)
    xc = tf.keras.layers.BatchNormalization(name="custom_bn2")(xc)
    xc = tf.keras.layers.Activation('relu', name="custom_relu2")(xc)
    xc = tf.keras.layers.MaxPooling2D((2, 2), name="custom_pool2")(xc)

    # Bloque Convolucional 3: 128 filtros
    xc = tf.keras.layers.Conv2D(
        128, (3, 3), padding='same', 
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="custom_conv3"
    )(xc)
    xc = tf.keras.layers.BatchNormalization(name="custom_bn3")(xc)
    xc = tf.keras.layers.Activation('relu', name="custom_relu3")(xc)
    xc = tf.keras.layers.MaxPooling2D((2, 2), name="custom_pool3")(xc)

    # Bloque Convolucional 4: 256 filtros
    xc = tf.keras.layers.Conv2D(
        256, (3, 3), padding='same', 
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="custom_conv4"
    )(xc)
    xc = tf.keras.layers.BatchNormalization(name="custom_bn4")(xc)
    xc = tf.keras.layers.Activation('relu', name="custom_relu4")(xc)
    xc = tf.keras.layers.MaxPooling2D((2, 2), name="custom_pool4")(xc)

    # 4. Clasificador Deno / Head de Clasificación
    gap = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pooling")(xc)
    dense_int = tf.keras.layers.Dense(
        128, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="custom_dense_int"
    )(gap)
    dropout = tf.keras.layers.Dropout(0.5, name="custom_dropout")(dense_int)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="tb_probability")(dropout)

    # Crear el modelo compuesto
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EndToEnd_CustomCNN_Classifier")
    return model

# ==========================================
# LOOP PRINCIPAL DE EJECUCIÓN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Entrena una CNN desde cero para detectar TB.")
    parser.add_argument("--dry-run", action="store_true", help="Ejecuta una prueba rápida de 1 época para depuración.")
    args = parser.parse_args()

    print("=============================================================")
    print(" EXPERIMENTO COMPARATIVO: CNN PERSONALIZADA DESDE CERO (1 CH) ")
    print("=============================================================")

    # Ajustes si es dry-run
    global EPOCHS, EXTERNAL_TEST_SAMPLES_PER_CLASS
    batch_size = BATCH_SIZE
    if args.dry_run:
        print("\n[MODO DRY-RUN ACTIVADO] Configurando ejecucion minima...")
        EPOCHS = 1
        EXTERNAL_TEST_SAMPLES_PER_CLASS = 10
        batch_size = 4

    # 1. Carga de U-Net Segura
    print(f"\n[INFO] Cargando modelo U-Net desde {UNET_MODEL_PATH}...")
    if not os.path.exists(UNET_MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo U-Net en {UNET_MODEL_PATH}")
        
    unet_model = tf.keras.models.load_model(
        UNET_MODEL_PATH, 
        custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coefficient': dice_coefficient},
        safe_mode=False
    )
    unet_model.trainable = False

    # 2. Descarga del dataset externo (Kaggle)
    t_paths, t_labels = get_external_test_data(limit_per_class=EXTERNAL_TEST_SAMPLES_PER_CLASS)
    if not t_paths:
        print("Error al obtener el dataset externo. Deteniendo ejecución.")
        return
        
    ext_ds = create_dataset(t_paths, t_labels, batch_size=batch_size, is_training=False)

    # 3. Carga y Split del Dataset Local (70/30)
    print("\n1. Obteniendo datos locales y split 70/30...")
    all_images, labels = get_local_images_and_labels(DATA_DIR)
    
    if args.dry_run:
        # Reducir drásticamente la muestra para el dry-run
        indices = np.random.choice(len(all_images), 40, replace=False)
        all_images = [all_images[i] for i in indices]
        labels = [labels[i] for i in indices]

    X_train, X_val, y_train, y_val = train_test_split(
        all_images, labels, test_size=0.30, stratify=labels, random_state=42
    )
    
    class_weights = check_imbalance(y_train, "Train")
    train_ds = create_dataset(X_train, y_train, batch_size=batch_size, is_training=True)
    val_ds = create_dataset(X_val, y_val, batch_size=batch_size, is_training=False)

    # 4. Inicializar y compilar el modelo
    print("\n2. Construyendo CNN propia desde cero...")
    model = build_custom_cnn_classifier(unet_model)
    model.summary()

    # Cálculo exacto de parámetros de la sección de clasificación (trainable)
    trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    total_params = model.count_params()
    unet_params = total_params - trainable_params
    
    print("\n-------------------------------------------------------------")
    print(f" RESUMEN DE PARÁMETROS:")
    print(f"  - Parámetros Totales del Modelo: {total_params:,}")
    print(f"  - Parámetros de la U-Net (Congelados): {unet_params:,}")
    print(f"  - Parámetros Entrenables (CNN Custom): {trainable_params:,}")
    print("-------------------------------------------------------------")

    # Compilamos con LR de 1e-4 como requiere la estrategia desde cero
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
    )

    # Callbacks de entrenamiento robusto desde cero
    callbacks = [
        ModelCheckpoint(
            filepath='best_custom_cnn_tb.keras', 
            monitor='val_auc', 
            save_best_only=True, 
            mode='max', 
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auc', 
            patience=15, 
            restore_best_weights=True, 
            mode='max', 
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_auc', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-7, 
            mode='max', 
            verbose=1
        )
    ]

    # 5. Entrenamiento
    print(f"\n3. Iniciando entrenamiento a {EPOCHS} épocas...")
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        class_weight=class_weights, 
        callbacks=callbacks
    )

    # 6. Evaluación Externa con Umbral Fijo 0.50
    print("\n4. Evaluación Final en el Conjunto de Test Externo...")
    
    # Cargar el mejor modelo guardado
    print(f"[INFO] Cargando el mejor modelo desde best_custom_cnn_tb.keras...")
    best_model = build_custom_cnn_classifier(unet_model)
    best_model.load_weights('best_custom_cnn_tb.keras')
    best_model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
    )
    
    print("\n--- Resultados Brutos (Test Externo) ---")
    best_model.evaluate(ext_ds, verbose=1)

    print("\n5. Evaluando con Umbral Fijo 0.50...")
    probs = best_model.predict(ext_ds, verbose=1)
    y_true = np.array(t_labels)
    
    THRESHOLD = 0.50
    y_pred = (probs >= THRESHOLD).astype(int).flatten()
    
    print(f"\nReporte de Clasificación (Custom CNN - Test Externo con Umbral {THRESHOLD}):")
    print(classification_report(y_true, y_pred, target_names=["Normal (0)", "TB (1)"]))
    
    # Generar y guardar la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Normal (Pred)", "TB (Pred)"], yticklabels=["Normal (Real)", "TB (Real)"])
    plt.title(f'Matriz de Confusión Externa - CNN Propia (Umbral: {THRESHOLD})')
    plt.tight_layout()
    plt.savefig('custom_cnn_test_cm.png', dpi=300)
    print("   > Guardada matriz final: 'custom_cnn_test_cm.png'")
    
    print("\n[EXITO] Experimento de CNN Custom desde cero finalizado con exito!")

if __name__ == '__main__':
    main()
