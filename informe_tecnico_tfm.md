# INFORME TÉCNICO Y ACADÉMICO: PIPELINE EN CASCADA (CASCADED FRAMEWORK) PARA LA SEGMENTACIÓN Y CLASIFICACIÓN DE TUBERCULOSIS EN RADIOGRAFÍAS DE TÓRAX

Este documento ha sido redactado bajo estándares de rigor académico para su incorporación directa en los apartados de **Metodología, Experimentación y Resultados** de la memoria del Trabajo de Fin de Máster (TFM).

---

## 1. INTRODUCCIÓN Y ARQUITECTURA DEL PIPELINE EN CASCADA

### Justificación Clínica y Técnica: Eliminación del Sesgo Biológico
En el desarrollo de sistemas de diagnóstico asistido por computadora (CAD, por sus siglas en inglés) aplicados a radiografías de tórax (CXR), la aproximación convencional basada en un modelo único de clasificación de extremo a extremo (end-to-end) suele presentar deficiencias críticas en términos de generalización. La razón principal de este fallo radica en la presencia de **ruido anatómico y tecnológico de fondo** que no guarda relación con la patología diana (Tuberculosis). Elementos como la musculatura de los hombros, el cuello, el tejido adiposo subcutáneo, la silueta cardíaca extra-pulmonar y, de manera muy acusada, los artefactos tecnológicos (marcas de posicionamiento del paciente, electrodos, variaciones en el colimador de la máquina de rayos X y etiquetas de texto incrustadas en los metadatos visuales) actúan como características espurias de gran peso.

Desde la perspectiva del aprendizaje profundo, un clasificador convolucional expuesto a imágenes completas tiende a minimizar la pérdida empírica memorizando estos patrones espurios. Por ejemplo, si un determinado hospital con pacientes de tuberculosis utiliza una máquina de rayos X con una calibración específica o un tipo concreto de marcador físico, el modelo aprenderá a clasificar basándose en dichos marcadores en lugar de identificar las opacidades parenquimatosas, consolidaciones o patrones cavitarios propios de la tuberculosis (TB). Este fenómeno introduce un **sesgo biológico y técnico** que destruye la capacidad de generalización del modelo ante conjuntos de prueba externos.

Para mitigar este problema, en este trabajo se ha diseñado e implementado un **Marco de Trabajo en Cascada (Cascaded Framework)**. La hipótesis fundamental de este enfoque es la compartimentación del problema en dos etapas ortogonales:
1. **Fase de Segmentación**: Aislamiento geométrico y anatómico de la Región de Interés (ROI), correspondiente a los campos pulmonares izquierdo y derecho.
2. **Fase de Clasificación**: Diagnóstico patológico restringido única y exclusivamente a la textura y morfología interna del parénquima pulmonar segmentado.

Al forzar al clasificador a procesar exclusivamente la información contenida dentro de la máscara pulmonar, se elimina de raíz cualquier posibilidad de que la red aprenda atajos o correlaciones espurias en el tejido periférico o en los bordes de la placa, garantizando que las decisiones clínicas se fundamenten en biomarcadores radiológicos legítimos.

---

### Fase 1 (Segmentación): Arquitectura U-Net y Validación Externa
Para la delimitación precisa de los pulmones, se implementó una arquitectura **U-Net** clásica modificada y optimizada para imágenes médicas de alta resolución. La red consta de una ruta de contracción (encoder) destinada a la extracción jerárquica de características espaciales y una ruta de expansión (decoder) encargada de la reconstrucción de la resolución original para la generación de la máscara binaria pixel a pixel.

```
Input Image (256x256x1)
       |
   [Encoder] ---> (Skip Connections) ---> [Decoder]
       |                                      |
   Max Pooling                            ConvTranspose
       v                                      v
   Bottleneck (Dropout 0.5) -----------> Output Mask (256x256x1)
```

#### Detalles de la Arquitectura
*   **Encoder**: Cuatro bloques convolucionales dobles con convoluciones de $3 \times 3$ (inicialización de pesos `he_normal` y activación ReLU), seguidos de capas de Max Pooling de $2 \times 2$.
*   **Bridge (Cuello de Botella)**: Dos capas convolucionales de $1024$ filtros con una capa de regularización por descarte (*Dropout* de 0.5) para evitar el sobreajuste (*overfitting*) en las representaciones latentes de mayor abstracción.
*   **Decoder**: Cuatro bloques de expansión compuestos por capas de convolución traspuesta (*Conv2DTranspose*) de $2 \times 2$ con *strides* de $2 \times 2$, concatenaciones (*Skip Connections*) que recuperan la información espacial de alta frecuencia del encoder, y convoluciones dobles de reconstrucción.
*   **Capa de Salida**: Una convolución de $1 \times 1$ con activación sigmoide para predecir la probabilidad de pertenencia al parénquima pulmonar para cada píxel.

#### Función de Pérdida Híbrida
Para entrenar la U-Net, se definió una función de pérdida combinada basada en la entropía cruzada binaria (BCE) y la pérdida de coeficientes Dice (*Dice Loss*), denotada como $L_{BCE-Dice}$:

$$\mathcal{L}_{BCE-Dice}(Y, \hat{Y}) = \mathcal{L}_{BCE}(Y, \hat{Y}) + \left(1 - \text{Dice}(Y, \hat{Y})\right)$$

Donde el coeficiente Dice se define formalmente como:

$$\text{Dice}(Y, \hat{Y}) = \frac{2 \sum_{i} y_i \hat{y}_i + \epsilon}{\sum_{i} y_i + \sum_{i} \hat{y}_i + \epsilon}$$

La componente BCE estabiliza los gradientes en las fases iniciales del entrenamiento evitando que la red colapse en predicciones triviales (por ejemplo, clasificar todo el fondo como cero), mientras que la componente Dice optimiza directamente la intersección sobre la unión de los pulmones, afinando de forma quirúrgica la delimitación de los bordes costodiafragmáticos y apicales.

#### Resultados de Segmentación
Durante la fase de validación en un conjunto de datos independiente y geográficamente disjunto (validación externa), la U-Net entrenada localmente demostró una robustez excepcional, alcanzando un **External Dice Coefficient de 0.9436**. Este elevado valor garantiza que las máscaras generadas cubren de forma precisa la anatomía pulmonar sin recortar zonas de parénquima relevante (lo cual podría ocultar lesiones apicales o derrames pleurales sospechosos de TB) y sin incluir ruido del cuello o de la cavidad abdominal.

---

### Fase 2 (Clasificación): Integración de la ROI y Pre-procesamiento de la Máscara
Una vez obtenida la máscara probabilística $\hat{M} \in [0, 1]^{H \times W \times 1}$ por la U-Net, se procede a su integración en el flujo de trabajo del clasificador patológico. El proceso de preparación y acondicionamiento de los datos se realiza a través de las siguientes sub-etapas matemáticas y lógicas:

1.  **Enmascaramiento de la Imagen (Masking)**: La imagen original en escala de grises $I_{in} \in [0, 1]^{H \times W \times 1}$ se multiplica elemento a elemento por la máscara generada para aislar la ROI pulmonar, anulando el fondo de manera determinista:
    $$I_{masked} = I_{in} \odot \hat{M}$$
2.  **Conversión de Canales (`grayscale_to_rgb`)**: Dado que las arquitecturas clasificadoras más robustas de la literatura científica (e.g., EfficientNet, DenseNet) se encuentran pre-entrenadas sobre el corpus de imágenes de ImageNet (el cual consta de 3 canales de color RGB), es imprescindible adaptar la dimensión del tensor. Se aplica una transformación lineal para duplicar el canal monocromo sobre los canales R, G y B:
    $$I_{rgb} = \text{grayscale\_to\_rgb}(I_{masked}) \in [0, 1]^{H \times W \times 3}$$
3.  **Re-escalado a $[0, 255]$**: A diferencia de otras redes neuronales que aceptan rangos normalizados de $[0, 1]$ o $[-1, 1]$, la implementación estándar de EfficientNetB0 en Keras integra capas internas de normalización específicas de ImageNet. Por tanto, para preservar la validez de los pesos pre-entrenados del modelo base, se introduce una capa física de re-escalado:
    $$I_{final} = I_{rgb} \times 255.0$$

Este pipeline de integración garantiza que el clasificador reciba una imagen estructurada de 3 canales, con el fondo estrictamente en valor de cero (negro puro) y la región pulmonar expuesta en rango de píxeles estándar $[0, 255]$, lista para la extracción de características finas.

---

## 2. EVOLUCIÓN METODOLÓGICA (DE LA V1 AL ENSEMBLE)

El desarrollo del módulo clasificador del parénquima pulmonar se llevó a cabo bajo un enfoque iterativo y de mejora continua. A continuación, se detalla la evolución metodológica del sistema.

```
+------------------------------------------+
|  Modelo V1: EfficientNetB0 Básico        |
|  - Desbloqueo total / safe_mode=True     |
|  - Overfitting masivo (~68% Accuracy)    |
+------------------------------------------+
                     |
                     v
+------------------------------------------+
|  Modelo V2: CLAHE + Fine-Tuning Quirúrgico|
|  - Mitigación del Domain Shift con CLAHE |
|  - Congelación de BatchNormalization     |
|  - Desbloqueo exclusivo Bloques 6 y 7    |
+------------------------------------------+
                     |
                     v
+------------------------------------------+
|  Modelo V3: Ensemble (Comité Médico)     |
|  - Fusión EfficientNetB0 + DenseNet121   |
|  - Fine-Tuning Bloque 5 de DenseNet      |
|  - Soft Voting (Promedio de Probabilidades)|
+------------------------------------------+
```

---

### Modelo V1 (EfficientNetB0 Básico) y el Fenómeno de Olvido Catastrófico
El primer intento de clasificación se construyó utilizando la espina dorsal (*backbone*) de **EfficientNetB0**, conectada a una cabeza de clasificación completamente conectada (*Fully Connected Head*) compuesta por una capa densa de 128 neuronas (con ReLU y Dropout de 0.5) y una neurona de salida sigmoidea. 

En esta primera iteración, se optó por un desbloqueo total de los pesos del modelo pre-entrenado desde el inicio del entrenamiento (`base_model.trainable = True`). Esta aproximación causó un **estancamiento temprano del rendimiento**, con una exactitud (*accuracy*) en validación externa que no superó el **~68%**, acompañada de curvas de pérdida que divergían rápidamente a partir de la época 5 (típico escenario de sobreajuste o *overfitting*).

#### Análisis Crítico del Fallo de la V1
1.  **Olvido Catastrófico (*Catastrophic Forgetting*)**: Al actualizar la totalidad de los coeficientes del extractor de características con un conjunto de datos local relativamente limitado, los gradientes del clasificador (inicializados aleatoriamente y con un error muy elevado al principio) se propagaron hacia las capas convolucionales iniciales. Esto destruyó los detectores de bordes, texturas y formas genéricas previamente aprendidos en ImageNet, forzando a la red a ajustarse únicamente a los detalles estadísticos irrelevantes del conjunto de entrenamiento.
2.  **Problema de Serialización en Keras (Lambda y safe_mode)**: Durante los procesos de persistencia del modelo (`ModelCheckpoint`), el guardado directo fallaba o corrompía el grafo computacional al intentar recargar el modelo. La causa subyacente se localizó en la capa `Lambda` encargada de ejecutar `tf.image.grayscale_to_rgb`. En Keras 3, la deserialización de funciones lambda anónimas está restringida bajo el protocolo de seguridad por defecto para evitar la ejecución de código no seguro. La resolución consistió en forzar la carga de pesos e instanciación mediante `safe_mode=False` y la declaración explícita de objetos personalizados, permitiendo recuperar la modularidad completa del pipeline.

---

### Modelo V2 (Filtro CLAHE y Fine-Tuning Quirúrgico)
Para superar el rendimiento de la V1, se rediseñó el flujo de datos y la estrategia de entrenamiento bajo dos pilares: la homogeneización de la iluminación y la restricción del flujo de gradientes.

#### A. Ecualización Adaptativa del Contraste Limitada por Histograma (CLAHE)
El dataset de validación externa (proveniente de Kaggle) y el dataset local presentaban un marcado **Domain Shift** (desplazamiento de dominio) debido a las diferencias en la tecnología de los detectores de rayos X, la dosis de radiación y el procesado digital de las imágenes. Mientras que el conjunto de entrenamiento local presentaba un histograma estrecho y de bajo contraste, el conjunto externo mostraba una alta dispersión de luminancia. 

Para mitigar esta discrepancia de dominio, se integró el algoritmo **CLAHE** en el pre-procesamiento de todas las imágenes. A diferencia de la ecualización clásica de histogramas, CLAHE opera sobre pequeñas regiones contextuales (losetas o *tiles* de $8 \times 8$ píxeles en nuestro caso) y limita la amplificación del contraste a un valor umbral (parámetro `clipLimit = 2.0`). Esto previene la amplificación del ruido de alta frecuencia en el parénquima pulmonar y normaliza las intensidades relativas de las opacidades focales, logrando alinear la distribución de características espaciales entre ambos datasets.

#### B. Protocolo de Fine-Tuning Quirúrgico y Congelación de BatchNormalization
Para conservar las características genéricas de ImageNet y adaptar únicamente las características complejas de alto nivel a la radiología, se aplicó una estrategia de ajuste fino en dos fases:
*   **Fase 1 (Feature Extraction)**: Se bloqueó por completo la base (`base_model.trainable = False`), permitiendo únicamente la actualización de la cabeza clasificadora a un ritmo de aprendizaje moderado ($\eta = 10^{-4}$).
*   **Fase 2 (Fine-Tuning Quirúrgico)**: Se desbloqueó la base de manera selectiva. Únicamente se declararon entrenables los **bloques profundos 6 y 7** de EfficientNetB0 (que modelan características de alta abstracción semántica) junto con la capa superior (*top layer*).

La decisión de mantener `layer.trainable = False` para todas las capas de `BatchNormalization` (BN), incluso en los bloques declarados como entrenables, es un aspecto de ingeniería crítico. Si las capas de BN se descongelaran, estimarían de manera dinámica la media y la varianza con los minilotes del conjunto de entrenamiento local. Dado el tamaño de batch relativamente pequeño ($N=16$) y la naturaleza médica de las imágenes, esto desajustaría los parámetros de escala ($\gamma$) y sesgo ($\beta$) acumulados por ImageNet, provocando una inestabilidad severa del gradiente y degradando de inmediato la exactitud global en el dataset externo.

---

### Modelo V3 (Ensemble Learning o "Comité Médico")
A pesar de la optimización del Modelo V2, las limitaciones intrínsecas de EfficientNetB0 (basado en convoluciones separables en profundidad y escalado optimizado por ancho/profundidad/resolución) limitaban la captura de ciertos patrones de textura difusos muy sutiles. Para empujar el límite del área bajo la curva ROC (AUC-ROC), se propuso un enfoque de **Ensemble Learning** combinando dos paradigmas arquitectónicos complementarios.

#### A. Integración de DenseNet121
Se entrenó de manera independiente una arquitectura **DenseNet121** bajo el mismo pipeline en cascada. La justificación científica de DenseNet reside en sus conexiones densas (*Dense Blocks*), donde cada capa recibe como entrada los mapas de características (*feature maps*) de todas las capas precedentes mediante concatenación:

$$x_l = H_l([x_0, x_1, \dots, x_{l-1}])$$

Esta arquitectura maximiza la reutilización de características de bajo nivel (bordes de costillas, siluetas parenquimatosas) y combate el problema del desvanecimiento del gradiente. Para el ajuste fino de DenseNet121, se descongeló exclusivamente el **bloque denso 5** (`conv5_block`), aplicando rigurosamente el mismo criterio de congelación de Batch Normalization.

#### B. Fusión por Soft Voting
El ensamble final se estructuró como un "Comité Médico" automatizado que opera mediante **Soft Voting (votación blanda)**. En lugar de tomar una decisión por mayoría simple (Hard Voting), el ensemble calcula la media aritmética de las probabilidades continuas de tuberculosis emitidas por ambos modelos individuales:

$$P_{Ensemble}(x) = \frac{P_{EffNet}(x) + P_{DenseNet}(x)}{2}$$

Este promedio de probabilidades calibradas reduce la varianza del error de generalización. Al combinar las representaciones altamente optimizadas en parámetros de EfficientNetB0 con la capacidad de preservación de características de DenseNet121, el ensamble neutraliza los sesgos individuales de cada arquitectura, ofreciendo una frontera de decisión mucho más estable y robusta ante imágenes de test externo de baja calidad o artefactos residuales de segmentación.

---

## 3. ANÁLISIS CRÍTICO DE LA CALIBRACIÓN CLÍNICA (EL UMBRAL DE CORTE)

Un error frecuente en el desarrollo de modelos predictivos en el ámbito médico es asumir por defecto el umbral de decisión matemático estándar de 0.50 (o basarse de manera estricta en optimizaciones teóricas). En este proyecto se analizó en detalle el impacto del umbral de decisión en el rendimiento del ensemble sobre el conjunto de test externo de Kaggle ($N = 1000$ casos balanceados: 500 Sanos y 500 con Tuberculosis).

A continuación se contrastan las tres filosofías de calibración analizadas:

| Métrica / Umbral | Criterio Matemático (Youden $J = 0.74$) | Criterio Clínico Sensible ($Threshold = 0.25$) | Umbral Calibrado de Triaje ($Threshold = 0.50$) |
| :--- | :---: | :---: | :---: |
| **Sensibilidad (Recall)** | Baja / Moderada | **Alta (92.4%)** | Balanceada / Alta |
| **Falsos Negativos (FN)** | **186 casos escapados** | Reducido (~Mínimo) | Controlado / Aceptable |
| **Falsos Positivos (FP)** | Muy Bajo (Alta Especificidad) | **401 casos sobre-diagnosticados** | Moderado (Bajo) |
| **Comportamiento Clínico** | Inaceptable (Peligro de contagio) | "Paranoia" (Saturación del sistema) | Óptimo para Triaje Hospitalario |

---

### A. El Umbral Matemático Puro (Estadístico J de Youden = 0.74)
El índice J de Youden busca maximizar la distancia vertical entre la curva ROC y la línea de clasificación aleatoria para encontrar el punto de equilibrio óptimo entre sensibilidad y especificidad:

$$J(\tau) = \text{Sensibilidad}(\tau) + \text{Especificidad}(\tau) - 1$$

Al aplicar este criterio matemático, la optimización fijó un umbral muy elevado de **0.74**.
*   **Consecuencia**: El modelo priorizó la especificidad, asegurando que prácticamente cualquier paciente catalogado como "Tuberculosis" presentara signos patológicos indiscutibles.
*   **Impacto Clínico Catastrófico**: Bajo este umbral, el modelo produjo **186 Falsos Negativos**. En un contexto de salud pública, que 186 pacientes con tuberculosis activa sean dados de alta como "sanos" representa un fallo crítico: la enfermedad continuará progresando en el individuo y, simultáneamente, propagándose dentro de su comunidad, anulando el propósito de una herramienta de detección temprana.

---

### B. El Umbral Clínico Ultrasensible (Threshold = 0.25)
Con el fin de mitigar los falsos negativos, se evaluó una política de umbral clínico muy bajo de **0.25**, diseñada para actuar como un filtro de máxima sensibilidad.
*   **Consecuencia**: El modelo elevó la sensibilidad hasta el **92.4%**, identificando de forma correcta a la inmensa mayoría de los pacientes enfermos de TB.
*   **Impacto de "Paranoia" Clínica**: Esta sensibilidad extrema se obtuvo a costa de un volumen inaceptable de falsos positivos (**401 Falsos Positivos**). El modelo categorizó como patológicos patrones pulmonares normales basándose en sutiles variaciones de brillo o densidad. En un hospital real, esto genera un cuello de botella logístico: 401 personas sanas requerirán pruebas confirmatorias de alto coste y carácter invasivo (cultivos de esputo, PCR o tomografías computarizadas), saturando los servicios de radiología y laboratorios, además de infligir un estrés psicológico innecesario en los pacientes.

---

### C. Justificación de la Calibración Final (Decisión de Ingeniería Médica)
La elección del umbral operativo final (fijado en **0.50** para el ensemble ponderado de Soft Voting o ajustado según las necesidades epidemiológicas del centro de salud) no debe responder a una fórmula matemática fría ni a una postura defensiva que ignore los recursos disponibles. Se debe enfocar como una **decisión estratégica de ingeniería médica**:

1.  **Balance de Triaje (Screening vs. Diagnóstico)**: En el flujo de trabajo hospitalario real, este pipeline en cascada se sitúa como una herramienta de triaje. Su objetivo principal es cribar rápidamente las radiografías normales para que los radiólogos concentren su atención en los casos potencialmente positivos.
2.  **Calibración en 0.50 (o compromiso intermedio en 0.35/0.50)**: Se establece como el punto óptimo de funcionamiento para el Ensemble. Al combinar el comportamiento estable del ensamble con CLAHE, este punto operativo estabiliza la tasa de falsos negativos por debajo de los límites epidemiológicos aceptables, manteniendo a su vez un volumen de falsos positivos lo suficientemente bajo como para no colapsar la capacidad de confirmación diagnóstica del hospital. Esta calibración controlada maximiza la utilidad del modelo en condiciones reales de implementación clínica.
