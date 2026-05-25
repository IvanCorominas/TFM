import os
import glob
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
DATA_DIR = os.path.join('.', 'data', 'processed')
UNET_MODEL_PATH = 'best_unet_model.keras'
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 30

def get_all_images_and_labels(data_dir):
    """
    Busca todas las imágenes en las carpetas train y test de data/processed.
    Extrae la etiqueta de la clase desde el nombre del archivo (termina en _0 o _1).
    """
    # Recolectar todas las rutas de imágenes disponibles
    train_images = glob.glob(os.path.join(data_dir, 'train', 'images', '*.png'))
    test_images = glob.glob(os.path.join(data_dir, 'test', 'images', '*.png'))
    all_images = train_images + test_images
    
    if len(all_images) == 0:
        raise ValueError(f"No se encontraron imágenes en {data_dir}. Asegúrate de haber ejecutado prepare_dataset.py")
        
    labels = []
    for img_path in all_images:
        filename = os.path.basename(img_path)
        # Ejemplo: CHNCXR_0001_0.png -> separamos por punto, luego por guion bajo y cogemos el último
        label_str = filename.split('.')[0].split('_')[-1]
        labels.append(int(label_str))
        
    return all_images, labels

def check_imbalance(y, split_name):
    """Verifica la proporción de clases y lanza una advertencia si hay desbalance crítico."""
    total = len(y)
    positives = sum(y)
    negatives = total - positives
    pos_pct = positives / total
    neg_pct = negatives / total
    
    print(f"[{split_name.upper():<10}] Total: {total:<4} | Sanos (_0): {negatives:<4} ({neg_pct:.1%}) | TB (_1): {positives:<4} ({pos_pct:.1%})")
    
    if pos_pct < 0.4 or neg_pct < 0.4:
        print(f"   ⚠️ ADVERTENCIA: Desbalance crítico detectado en {split_name}.")
        print("   Se sugiere usar 'class_weights' en el entrenamiento.")
        
    # Calcular class weights en caso de ser necesario
    weight_for_0 = (1 / negatives) * (total / 2.0)
    weight_for_1 = (1 / positives) * (total / 2.0)
    
    # NOTA: Hemos eliminado el multiplicador x1.5 para la clase TB 
    # ya que causaba un sesgo extremo resultando en demasiados falsos positivos.
    
    class_weights = {0: weight_for_0, 1: weight_for_1}
    return class_weights

def process_path(img_path, label):
    """Carga y procesa la imagen para el dataset."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    """Aplica aumentación de datos avanzada al conjunto de entrenamiento."""
    img = tf.image.random_flip_left_right(img)
    # Pequeñas variaciones de brillo y contraste simulan distintas máquinas de rayos X
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    # Asegurar que el rango sigue siendo válido
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def create_dataset(paths, labels, is_training=False):
    """Crea un pipeline tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def build_end_to_end_classifier():
    """
    Construye un modelo End-to-End que integra la U-Net congelada para el enmascaramiento
    y usa EfficientNetB0 para la clasificación final.
    """
    print("\n[INFO] Cargando modelo U-Net...")
    if not os.path.exists(UNET_MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo U-Net en {UNET_MODEL_PATH}")
        
    unet_model = tf.keras.models.load_model(
        UNET_MODEL_PATH, 
        custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coefficient': dice_coefficient}
    )
    # Congelar los pesos de la U-Net
    unet_model.trainable = False

    # 1. Capa de Entrada
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 1), name="input_image")

    # 2. Inferencia de Máscara (training=False asegura que capas como BatchNorm funcionen en modo inferencia)
    mask = unet_model(inputs, training=False)
    
    # Si el modelo devuelve una lista (por tener outputs=[outputs]), extraemos el primer tensor
    if isinstance(mask, list):
        mask = mask[0]

    # 3. Enmascaramiento: Multiplicar imagen original por la máscara
    masked_img = tf.keras.layers.Multiply(name="masked_lungs")([inputs, mask])

    # 4. Fusión de Canales (Channel Stacking) inteligente
    # Canal R: Imagen Original (contexto global)
    # Canal G: Máscara Predicha (foco de atención espacial)
    # Canal B: Imagen Enmascarada (detalle pulmonar limpio)
    rgb_stacked = tf.keras.layers.Concatenate(axis=-1, name="channel_stacking")([inputs, mask, masked_img])
    
    # IMPORTANTE: EfficientNet espera valores en el rango [0, 255] para usar sus pesos pre-entrenados correctamente.
    rescaled_img = tf.keras.layers.Rescaling(255.0, name="rescale_for_efficientnet")(rgb_stacked)

    # 5. Base de Clasificación: EfficientNetB0
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=(*IMG_SIZE, 3)
    )
    base_model._name = "efficientnet_base"
    # Congelamos el modelo base para evitar destruir los pesos pre-entrenados (Transfer Learning real)
    base_model.trainable = False
    
    x = base_model(rescaled_img, training=False)
    
    # 6. Capas Finales (Head del Clasificador)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_regularization")(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="tb_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EndToEnd_TB_Classifier")
    return model

def main():
    print("======================================================")
    print(" PIPELINE DE CLASIFICACIÓN TB CON ENMASCARAMIENTO U-NET ")
    print("======================================================")
    
    # 1. Obtener y Dividir Datos
    print("\n1. Obteniendo datos y realizando split estratificado...")
    all_images, labels = get_all_images_and_labels(DATA_DIR)
    
    # Split 70% Train, 30% Temporal (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, labels, test_size=0.30, stratify=labels, random_state=42
    )
    # Split 50% Val, 50% Test del Temporal -> 15% Val y 15% Test del total
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    # Validar desbalance de clases
    print("\nResumen de Clases por Partición:")
    class_weights = check_imbalance(y_train, "Train")
    _ = check_imbalance(y_val, "Val")
    _ = check_imbalance(y_test, "Test")
    
    # 2. Crear Datasets
    print("\n2. Creando Datasets (tf.data)...")
    train_ds = create_dataset(X_train, y_train, is_training=True)
    val_ds = create_dataset(X_val, y_val, is_training=False)
    # ATENCIÓN: Para la matriz de confusión necesitamos que test_ds NO se mezcle
    # y no use data augmentation. create_dataset(is_training=False) ya hace esto.
    test_ds = create_dataset(X_test, y_test, is_training=False) 

    # 3. Construir Modelo
    print("\n3. Construyendo Arquitectura del Clasificador...")
    model = build_end_to_end_classifier()
    model.summary()
    
    # ==========================================
    # FASE 1: ENTRENAMIENTO DE LA CAPA SUPERIOR
    # ==========================================
    print("\n4. FASE 1: Entrenamiento de la Capa de Clasificación (EfficientNet congelada)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
    )
    
    callbacks_fase1 = [
        EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True, verbose=1)
    ]
    
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15, # Entrenamos unas pocas épocas la cabeza
        callbacks=callbacks_fase1,
        class_weight=class_weights
    )

    # ==========================================
    # FASE 2: FINE-TUNING (Descongelar base_model)
    # ==========================================
    print("\n5. FASE 2: Fine-Tuning (Descongelando EfficientNetB0)...")
    # Buscamos la capa EfficientNet
    for layer in model.layers:
        if layer.name == "efficientnet_base":
            # Descongelamos los pesos
            layer.trainable = True
            
            # Mantenemos los BatchNorm congelados para no desestabilizar la red
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                    sub_layer.trainable = False

    # Recompilamos el modelo con un learning rate MUCHO menor
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # LR bajo es crítico aquí
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
    )
    
    callbacks_fase2 = [
        ModelCheckpoint(
            filepath='best_tb_classifier.keras',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auc',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=4,
            mode='max',
            min_lr=1e-7,
            verbose=1
        )
    ]

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS, # Épocas completas
        callbacks=callbacks_fase2,
        class_weight=class_weights
    )
    
    # 6. Evaluación en Test
    print("\n6. Evaluación Final en Conjunto de Test...")
    # Cargar el mejor modelo de la Fase 2
    best_model = tf.keras.models.load_model(
        'best_tb_classifier.keras',
        custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coefficient': dice_coefficient}
    )
    test_metrics = best_model.evaluate(test_ds, verbose=1)
    
    print("\nResultados en Test:")
    for name, val in zip(best_model.metrics_names, test_metrics):
        print(f"   ✓ {name}: {val:.4f}")

    # 7. Matriz de Confusión (CON UMBRAL AJUSTADO)
    print("\n7. Generando Matriz de Confusión (Optimizada para Recall)...")
    y_pred_probs = best_model.predict(test_ds, verbose=1)
    
    # UMBRAL ESTÁNDAR: Volvemos a 0.5 (el 0.4 anterior causaba muchos falsos positivos)
    THRESHOLD = 0.5 
    print(f"   * Usando umbral de clasificación: {THRESHOLD}")
    y_pred = (y_pred_probs > THRESHOLD).astype(int).flatten()
    y_true = np.array(y_test)
    
    # Mostrar reporte en consola
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=["Sanos (_0)", "TB (_1)"]))
    
    # Crear y guardar la gráfica
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Sanos (Pred)", "TB (Pred)"],
                yticklabels=["Sanos (Real)", "TB (Real)"])
    plt.title(f'Matriz de Confusión - TB (Umbral {THRESHOLD})')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("   ✓ Matriz de Confusión guardada como 'confusion_matrix.png'")
    
    print("\n¡Entrenamiento y Evaluación del Clasificador Finalizados con Éxito! 🚀")

if __name__ == '__main__':
    main()
