import os
import matplotlib
matplotlib.use('Agg')

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_e12a922e07b325050bd7a5dd362c2dcf'
import kagglehub

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Importamos las funciones necesarias de tu script principal
from train_tb_pipeline import get_external_test_data, create_dataset, build_end_to_end_classifier

def main():
    print("=============================================================")
    print(" EVALUACIÓN DIRECTA DEL TEST EXTERNO (KAGGLE) ")
    print("=============================================================")
    
    print("\n[EVALUACIÓN] Iniciando Test Externo...")
    t_paths, t_labels = get_external_test_data()
    if not t_paths: 
        return
    
    ext_ds = create_dataset(t_paths, t_labels, is_training=False)
    
    print("\n[INFO] Cargando la arquitectura y los pesos del mejor modelo entrenado...")
    # Usamos la carga segura de pesos para evitar el error de deserialización de la Lambda
    best_model = build_end_to_end_classifier()
    best_model.load_weights('best_tb_pipeline.keras')
    
    # Keras requiere compilar el modelo antes de llamar a .evaluate()
    best_model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
    )
    
    print("\n--- Resultados Test Externo (Kaggle) ---")
    metrics = best_model.evaluate(ext_ds, verbose=1)
    for name, val in zip(best_model.metrics_names, metrics):
        print(f"   > {name}: {val:.4f}")

    # Priorización de Recall: Umbral 0.35
    THRESHOLD = 0.35 
    print(f"\n7. Generando Matriz de Confusión (Umbral {THRESHOLD} para Priorizar Recall)...")
    probs = best_model.predict(ext_ds, verbose=1)
    
    y_pred = (probs > THRESHOLD).astype(int).flatten()
    y_true = np.array(t_labels)
    
    print("\nReporte de Clasificación (Test Externo):")
    print(classification_report(y_true, y_pred, target_names=["Normal (0)", "TB (1)"]))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Normal (Pred)", "TB (Pred)"], yticklabels=["Normal (Real)", "TB (Real)"])
    plt.title(f'Matriz de Confusión Externa (Threshold {THRESHOLD})')
    plt.tight_layout()
    plt.savefig('external_test_confusion_matrix.png', dpi=300)
    print("   > Archivo 'external_test_confusion_matrix.png' creado.")

if __name__ == '__main__':
    main()
