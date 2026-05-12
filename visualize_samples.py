import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import supervisely as sly

def main():
    # 1. Definir la ruta del proyecto Supervisely en formato Windows (usando os.path.join para evitar errores)
    dataset_path = os.path.join('.', 'data', 'raw', 'chest-xray-masks-and-labels')
    
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró la carpeta del dataset en '{dataset_path}'.")
        print("Por favor, asegúrate de haber ejecutado setup_dataset.py primero.")
        return

    # 2. Cargar el proyecto Supervisely (en modo lectura)
    # Esto lee automáticamente la estructura de directorios y los archivos JSON de anotaciones
    print("Cargando el proyecto...")
    project = sly.Project(dataset_path, sly.OpenMode.READ)
    
    # 3. Acceder al primer dataset disponible (en este caso, 'CXRpng-train' u otro equivalente)
    dataset = list(project.datasets)[0]
    items = dataset.get_items_names()
    
    print(f"Dataset '{dataset.name}' cargado con {len(items)} imágenes.")
    
    # 4. Seleccionar 3 imágenes aleatorias
    sample_items = random.sample(items, 3)
    
    # 5. Configurar la figura de Matplotlib (3 filas x 3 columnas)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.canvas.manager.set_window_title("Análisis Exploratorio - Radiografías de Tórax")
    
    for i, item_name in enumerate(sample_items):
        # --- CARGA DE DATOS ---
        
        # A) Cargar la imagen original
        img_path = dataset.get_img_path(item_name)
        img = Image.open(img_path).convert('RGB')
        
        # B) Cargar las anotaciones (JSON) y generar la máscara binaria
        ann_path = dataset.get_ann_path(item_name)
        # Cargamos el JSON de anotación validando contra los metadatos del proyecto (las clases permitidas)
        ann = sly.Annotation.load_json_file(ann_path, project.meta)
        
        # Creamos una matriz de ceros del tamaño de la imagen original
        mask = np.zeros(ann.img_size, dtype=np.uint8)
        
        # Dibujamos las etiquetas (polígonos/bitmaps) sobre nuestra matriz
        # El color 1 nos dará una máscara binaria (0 = fondo, 1 = pulmones)
        for label in ann.labels:
            label.geometry.draw(mask, color=1)
            
        # Creamos un overlay (capa superpuesta) en color verde con canal Alfa (transparencia)
        # La máscara verde tendrá esta forma: (R=0, G=255, B=0, Alpha=128 donde hay pulmón, 0 donde no)
        overlay_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        overlay_rgba[mask == 1] = [0, 255, 0, 100] # Verde con transparencia (~40%)
        
        # --- VISUALIZACIÓN ---
        
        # Columna 1: Radiografía Original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original: {item_name}")
        axes[i, 0].axis('off')
        
        # Columna 2: Máscara de Segmentación
        # Mostramos la máscara binaria en blanco y negro
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Máscara de Pulmones")
        axes[i, 1].axis('off')
        
        # Columna 3: Overlay (Máscara sobre Original)
        axes[i, 2].imshow(img)           # Dibujamos la imagen original de fondo
        axes[i, 2].imshow(overlay_rgba)  # Dibujamos la máscara con canal alfa encima
        axes[i, 2].set_title("Overlay (Verde)")
        axes[i, 2].axis('off')

    # Ajustar espacios y mostrar
    plt.tight_layout()
    print("Mostrando visualización... Cierra la ventana emergente para terminar el script.")
    plt.show()

if __name__ == "__main__":
    main()
