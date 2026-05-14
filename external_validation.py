import os
import glob
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import kagglehub

# Importar funciones personalizadas desde nuestro script de entrenamiento
from unet_model import dice_coefficient, bce_dice_loss

def process_path(img_path, mask_path):
    """
    Carga, decodifica, redimensiona y normaliza una imagen y su máscara.
    Se aplican las mismas transformaciones que en el entrenamiento para 
    garantizar consistencia y validez de la evaluación.
    """
    IMG_SIZE = (256, 256)
    
    # 1. Procesar Imagen (Escala de grises, 256x256, Normalizada [0, 1])
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    
    # 2. Procesar Máscara (Igual que entrenamiento: Nearest neighbor, Normalizada [0, 1], Binary)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.math.round(mask)
    
    return img, mask

def main():
    print("==========================================================================")
    print("  VALIDACIÓN EXTERNA ROBUSTA DEL MODELO U-NET (Dataset Kaggle)")
    print("==========================================================================")
    
    # 1. Descarga del Dataset
    print("\n1. Descargando dataset de Kaggle (tawsifurrahman/covid19-radiography-database)...")
    path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
    print(f"Dataset disponible en: {path}")

    # 2. Filtrado de Datos
    print("\n2. Localizando imágenes 'Normales' y sus máscaras...")
    
    # Buscar todas las imágenes posibles para poder emparejarlas
    all_images = glob.glob(os.path.join(path, '**', '*.*'), recursive=True)
    
    normal_images = []
    normal_masks = {}
    
    # Clasificamos archivos basándonos en su ruta y nombre
    for p in all_images:
        if not p.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        p_normalized = p.replace('\\', '/')
        if 'Normal' in p_normalized:
            filename = os.path.basename(p)
            if 'Mask' in p_normalized or 'mask' in p_normalized:
                normal_masks[filename] = p
            else:
                normal_images.append(p)
                
    # Emparejar usando el nombre de archivo (las máscaras suelen llamarse igual que la imagen original)
    paired_images = []
    paired_masks = []
    
    for img_p in normal_images:
        filename = os.path.basename(img_p)
        if filename in normal_masks:
            paired_images.append(img_p)
            paired_masks.append(normal_masks[filename])
            
    if not paired_images:
        print("Error: No se pudieron emparejar las imágenes 'Normal' con sus máscaras.")
        return
        
    print(f"Total encontrado: {len(paired_images)} pares de imágenes/máscaras sanas.")
    
    # Seleccionar una muestra aleatoria de entre 100 y 200 imágenes para ser representativos pero rápidos
    sample_size = min(150, len(paired_images))
    random.seed(42) # Fijo la semilla para reproducibilidad de esta validación
    indices = random.sample(range(len(paired_images)), sample_size)
    
    sampled_imgs = [paired_images[i] for i in indices]
    sampled_masks = [paired_masks[i] for i in indices]
    print(f"Se seleccionó una muestra aleatoria de {sample_size} pares para el Test Externo.")
    
    # Crear pipeline de datos con tf.data (Pre-procesamiento)
    print("\n3. Pre-procesando datos (Redimensión a 256x256, Normalización, Grayscale)...")
    dataset = tf.data.Dataset.from_tensor_slices((sampled_imgs, sampled_masks))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)
    
    # 4. Evaluación con el Modelo Guardado
    print("\n4. Cargando modelo 'best_unet_model.keras'...")
    try:
        model = tf.keras.models.load_model(
            'best_unet_model.keras',
            custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coefficient': dice_coefficient}
        )
    except Exception as e:
        print(f"Error fatal al cargar el modelo: {e}")
        return
        
    print("\n5. Ejecutando Evaluación sobre el Dataset Externo (model.evaluate)...")
    results = model.evaluate(dataset, verbose=1)
    
    print("\n" + "="*60)
    print(" 📊 RESULTADOS DEL EXTERNAL DICE COEFFICIENT 📊 ")
    print("="*60)
    print(f"External Loss (BCE + Dice): {results[0]:.4f}")
    print(f"External Dice Coefficient:  {results[1]:.4f}")
    print("="*60)
    
    # 5. Resultados Visuales
    print("\n6. Generando imagen comparativa de predicciones (external_test_results.png)...")
    for batch_imgs, batch_masks in dataset.take(1):
        pred_masks = model.predict(batch_imgs, verbose=0)
        
        plt.figure(figsize=(12, 20))
        num_examples = min(5, len(batch_imgs))
        
        for i in range(num_examples):
            img_disp = tf.squeeze(batch_imgs[i])
            true_mask_disp = tf.squeeze(batch_masks[i])
            pred_mask_disp = tf.squeeze(pred_masks[i])
            
            # Imagen Original
            plt.subplot(num_examples, 3, i*3 + 1)
            plt.imshow(img_disp, cmap='gray')
            if i == 0: plt.title('Radiografía Original\n(Dataset Externo)')
            plt.axis('off')
            
            # Máscara Real
            plt.subplot(num_examples, 3, i*3 + 2)
            plt.imshow(true_mask_disp, cmap='gray')
            if i == 0: plt.title('Máscara Real\n(Ground Truth)')
            plt.axis('off')
            
            # Predicción U-Net
            plt.subplot(num_examples, 3, i*3 + 3)
            plt.imshow(pred_mask_disp, cmap='gray')
            if i == 0: plt.title('Predicción U-Net\n(Nuestro Modelo)')
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig('external_test_results.png', dpi=300)
        print("   ✓ Archivo 'external_test_results.png' creado.")
        break
    
    # Comentarios sobre validación científica
    print("\n" + "*"*70)
    print(" 🔬 CONCLUSIÓN DE VALIDACIÓN CIENTÍFICA (TFM) 🔬")
    print("*"*70)
    print("¿Por qué un buen resultado en este dataset consolida el TFM?")
    print("1. Generalización: El dataset original de entrenamiento proviene de un")
    print("   entorno clínico específico (con sus propios equipos de Rayos X,")
    print("   calibraciones, y protocolos). Al probar en una base de datos")
    print("   completamente ajena (como este de Kaggle), demostramos que la")
    print("   U-Net ha aprendido la 'semántica anatómica' real de los pulmones")
    print("   y no solo artefactos o peculiaridades del dataset de origen.")
    print("2. Robustez contra Overfitting: Mantener un Dice Coefficient alto")
    print("   aquí significa que el modelo no sufre de sobreajuste crítico.")
    print("3. Valor Clínico: En un entorno de producción o despliegue hospitalario,")
    print("   los datos de entrada serán inherentemente diferentes ('domain shift').")
    print("   Este test simula exactamente ese escenario, otorgando rigor al trabajo.")
    print("*"*70 + "\n")

if __name__ == '__main__':
    main()
