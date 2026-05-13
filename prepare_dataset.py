import os
from PIL import Image
import numpy as np
import supervisely as sly
import shutil
from tqdm import tqdm

def convert_supervisely_to_standard(raw_dir, processed_dir):
    """
    Convierte un dataset en formato Supervisely (img/ y ann/ con JSONs)
    a un formato estándar de Machine Learning (images/ y masks/ con PNGs).
    """
    project_path = os.path.join(raw_dir, 'chest-xray-masks-and-labels')
    
    if not os.path.exists(project_path):
        print(f"Error: No se encontró el proyecto en {project_path}")
        return

    print("Cargando proyecto Supervisely...")
    project = sly.Project(project_path, sly.OpenMode.READ)
    
    # Crear carpetas de destino
    for split in ['train', 'test']:
        os.makedirs(os.path.join(processed_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(processed_dir, split, 'masks'), exist_ok=True)

    # Mapeo de los nombres de los datasets en Supervisely a nuestras carpetas
    # En este dataset en particular, el entrenamiento se llama 'CXRpng-train' y el test 'test'
    dataset_mapping = {
        'CXRpng-train': 'train',
        'test': 'test'
    }

    for dataset in project.datasets:
        dataset_name = dataset.name
        
        # Determinar si es train o test
        split = dataset_mapping.get(dataset_name)
        if not split:
            print(f"Dataset {dataset_name} ignorado (no es train ni test).")
            continue
            
        items = dataset.get_items_names()
        print(f"Procesando {len(items)} imágenes para el conjunto de '{split}'...")
        
        for item_name in tqdm(items):
            # 1. Rutas originales
            img_path = dataset.get_img_path(item_name)
            ann_path = dataset.get_ann_path(item_name)
            
            # 2. Cargar anotación JSON y generar máscara binaria
            ann = sly.Annotation.load_json_file(ann_path, project.meta)
            mask = np.zeros(ann.img_size, dtype=np.uint8)
            
            # Dibujar los polígonos de pulmones como 255 (blanco) en la máscara
            for label in ann.labels:
                label.geometry.draw(mask, color=255)
                
            # 3. Guardar la imagen en formato estándar
            # Copiamos la imagen original
            dest_img_path = os.path.join(processed_dir, split, 'images', item_name)
            shutil.copy(img_path, dest_img_path)
            
            # Guardamos la máscara generada como PNG
            # Aseguramos que la máscara se guarde con extensión .png aunque la imagen original sea .jpg
            base_name = os.path.splitext(item_name)[0]
            dest_mask_path = os.path.join(processed_dir, split, 'masks', f"{base_name}.png")
            
            # Si la imagen era .jpg, la renombramos a .png en el dest_img_path para tener coincidencia exacta
            if not dest_img_path.endswith('.png'):
                new_dest_img = os.path.join(processed_dir, split, 'images', f"{base_name}.png")
                # Convertir imagen original a png
                img = Image.open(dest_img_path)
                img.save(new_dest_img)
                os.remove(dest_img_path)
            
            # Guardar la máscara con PIL
            mask_img = Image.fromarray(mask)
            mask_img.save(dest_mask_path)

    print("\n¡Conversión completada con éxito!")
    print(f"Tus datos están listos en: {processed_dir}")

if __name__ == "__main__":
    RAW_DIR = os.path.join('.', 'data', 'raw')
    PROCESSED_DIR = os.path.join('.', 'data', 'processed')
    convert_supervisely_to_standard(RAW_DIR, PROCESSED_DIR)
