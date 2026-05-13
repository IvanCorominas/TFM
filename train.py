import os
import glob
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Importar el modelo y las funciones personalizadas
from unet_model import build_unet, dice_coefficient, bce_dice_loss

"""
================================================================
ESTRUCTURA DE CARPETAS ESPERADA:
================================================================
El script espera que tu dataset esté organizado de la siguiente manera
dentro de la carpeta principal del proyecto:

./data/raw/
├── train/
│   ├── images/   <-- Imágenes originales para entrenamiento (.png, .jpg)
│   └── masks/    <-- Máscaras binarias correspondientes (.png, .jpg)
└── test/
    ├── images/   <-- Imágenes originales para evaluación final
    └── masks/    <-- Máscaras binarias correspondientes

NOTA IMPORTANTE: Se asume que una imagen y su máscara comparten el 
mismo nombre de archivo para ser emparejadas correctamente 
(ej: train/images/001.png y train/masks/001.png).
"""

# ==========================================
# HIPERPARÁMETROS Y CONFIGURACIÓN
# ==========================================
DATA_DIR = os.path.join('.', 'data', 'processed')
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 50

def get_image_mask_paths(split_dir):
    """Obtiene las listas de rutas para imágenes y máscaras de una carpeta dada."""
    images_dir = os.path.join(DATA_DIR, split_dir, 'images')
    masks_dir = os.path.join(DATA_DIR, split_dir, 'masks')
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        raise ValueError(f"Faltan carpetas images/ o masks/ en {os.path.join(DATA_DIR, split_dir)}")
        
    image_paths = sorted(glob.glob(os.path.join(images_dir, '*.*')))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.*')))
    
    if len(image_paths) == 0:
        raise ValueError(f"No se encontraron imágenes en {images_dir}")
        
    if len(image_paths) != len(mask_paths):
        print(f"Advertencia: Número distinto de imágenes ({len(image_paths)}) y máscaras ({len(mask_paths)}) en {split_dir}")
        
    return image_paths, mask_paths

def process_path(img_path, mask_path):
    """Carga, decodifica, redimensiona y normaliza una imagen y su máscara."""
    # 1. Procesar Imagen
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0  # Normalización [0, 1]
    
    # 2. Procesar Máscara
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.math.round(mask)  # Asegurar valores estrictamente binarios (0 o 1)
    
    return img, mask

# Definir la capa de rotación fuera de la función map para eficiencia
rotation_layer = tf.keras.layers.RandomRotation(factor=0.05, fill_mode='nearest')

def augment(img, mask):
    """Aplica Data Augmentation sincronizado a imagen y máscara (solo para train)."""
    # Concatenar a lo largo del eje de canales para aplicar la misma transformación a ambos a la vez
    combined = tf.concat([img, mask], axis=-1)
    
    # Transformación 1: Volteo horizontal aleatorio
    combined = tf.image.random_flip_left_right(combined)
    
    # Transformación 2: Pequeñas rotaciones
    # Expandimos dimensiones (añadimos un eje de batch=1) ya que las Keras Layers esperan un batch
    combined = tf.expand_dims(combined, 0)
    combined = rotation_layer(combined)
    combined = tf.squeeze(combined, 0) # Devolvemos a la forma original
    
    # Separar imagen y máscara después de transformarlos juntos
    img = combined[..., 0:1]
    mask = combined[..., 1:2]
    
    return img, mask

def create_dataset(image_paths, mask_paths, is_training=False):
    """Genera un pipeline tf.data super eficiente."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    # Mapeo de carga
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Si es entrenamiento, aplicamos augmentation y barajamos
    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
        
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) # Pre-carga el siguiente batch en CPU mientras GPU entrena
    
    return dataset

def main():
    print("1. Cargando rutas y preparando particiones...")
    try:
        train_img_paths, train_mask_paths = get_image_mask_paths('train')
        test_img_paths, test_mask_paths = get_image_mask_paths('test')
    except Exception as e:
        print(f"Error cargando los datos: {e}")
        return
    
    # Separar 20% del entrenamiento de forma interna para Validación
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        train_img_paths, train_mask_paths, test_size=0.20, random_state=42
    )
    
    print(f"   ✓ Entrenamiento: {len(train_img_paths)} muestras")
    print(f"   ✓ Validación:    {len(val_img_paths)} muestras")
    print(f"   ✓ Test (Final):  {len(test_img_paths)} muestras")
    
    print("\n2. Creando Data Generators (tf.data)...")
    train_dataset = create_dataset(train_img_paths, train_mask_paths, is_training=True)
    val_dataset = create_dataset(val_img_paths, val_mask_paths, is_training=False)
    test_dataset = create_dataset(test_img_paths, test_mask_paths, is_training=False)
    
    print("\n3. Construyendo arquitectura U-Net...")
    model = build_unet(input_shape=(*IMG_SIZE, 1))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=bce_dice_loss, 
        metrics=[dice_coefficient]
    )
    
    print("\n4. Configurando Callbacks de entrenamiento...")
    callbacks = [
        # Guardar el mejor modelo detectando cuándo mejora la validación
        ModelCheckpoint(
            filepath='best_unet_model.keras',
            monitor='val_dice_coefficient',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Parar de entrenar si pasadas 12 épocas no se ha mejorado nada
        EarlyStopping(
            monitor='val_dice_coefficient',
            patience=12,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        # Reducir el learning rate a la mitad si el modelo se estanca durante 5 épocas
        ReduceLROnPlateau(
            monitor='val_dice_coefficient',
            factor=0.5,
            patience=5,
            mode='max',
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\n5. Iniciando el Entrenamiento...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    print("\n6. Guardando métricas (training_history.png)...")
    plt.figure(figsize=(14, 5))
    
    # Gráfica: Función de Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validación', linewidth=2)
    plt.title('Evolución de la Pérdida (Dice Loss)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfica: Coeficiente Dice
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'], label='Entrenamiento', linewidth=2)
    plt.plot(history.history['val_dice_coefficient'], label='Validación', linewidth=2)
    plt.title('Evolución del Dice Coefficient')
    plt.xlabel('Época')
    plt.ylabel('Puntuación Dice')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    
    print("\n7. Evaluación final en conjunto de TEST intacto...")
    # Cargar explícitamente el mejor modelo pasándole las funciones custom
    best_model = tf.keras.models.load_model(
        'best_unet_model.keras', 
        custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coefficient': dice_coefficient}
    )
    test_results = best_model.evaluate(test_dataset, verbose=1)
    print(f"   ✓ Test Loss Final: {test_results[0]:.4f}")
    print(f"   ✓ Test Dice Final: {test_results[1]:.4f}")
    
    print("\n8. Generando matriz de predicciones (test_predictions.png)...")
    sample_indices = random.sample(range(len(test_img_paths)), 3)
    plt.figure(figsize=(12, 12))
    
    for i, idx in enumerate(sample_indices):
        img_path = test_img_paths[idx]
        mask_path = test_mask_paths[idx]
        
        img, true_mask = process_path(img_path, mask_path)
        
        # Expandir dimensiones (Batch=1) para predecir
        img_batch = tf.expand_dims(img, 0)
        pred_mask = best_model.predict(img_batch, verbose=0)[0]
        
        # Eliminar dimensión de canal para matplotlib
        img_disp = tf.squeeze(img)
        true_mask_disp = tf.squeeze(true_mask)
        pred_mask_disp = tf.squeeze(pred_mask)
        
        # Columna 1: Imagen Original
        plt.subplot(3, 3, i*3 + 1)
        plt.imshow(img_disp, cmap='gray')
        if i == 0: plt.title('Radiografía Original')
        plt.axis('off')
        
        # Columna 2: Ground Truth
        plt.subplot(3, 3, i*3 + 2)
        plt.imshow(true_mask_disp, cmap='gray')
        if i == 0: plt.title('Máscara Real')
        plt.axis('off')
        
        # Columna 3: Predicción
        plt.subplot(3, 3, i*3 + 3)
        plt.imshow(pred_mask_disp, cmap='gray')
        if i == 0: plt.title('Predicción U-Net')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=300)
    print("   ✓ test_predictions.png creado.")
    print("\n¡Entrenamiento y Evaluación Finalizados con Éxito! 🚀")

if __name__ == '__main__':
    main()
