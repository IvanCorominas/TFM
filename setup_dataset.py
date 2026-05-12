import os
import dataset_tools as dtools

def main():
    # Definimos la ruta de destino dentro de la carpeta del proyecto
    dataset_dir = os.path.join('.', 'data', 'raw')
    
    # Comprobamos si el directorio existe y si tiene archivos
    if os.path.exists(dataset_dir) and os.listdir(dataset_dir):
        print(f"El dataset ya parece estar descargado en '{dataset_dir}'.")
        print("Omitiendo la descarga para ahorrar tiempo y ancho de banda (5GB).")
    else:
        print(f"Preparando la descarga en: {dataset_dir}")
        # Creamos los directorios si no existen
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Descargamos el dataset
        try:
            dtools.download(dataset='Chest Xray Masks and Labels', dst_dir=dataset_dir)
            print("¡Descarga completada con éxito!")
        except Exception as e:
            print(f"Ha ocurrido un error durante la descarga: {e}")

if __name__ == "__main__":
    main()
