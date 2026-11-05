# Práctica 3

**Introducción a la Ciencia de Datos**

## Objetivo

Predecir, al cierre del primer semestre de licenciatura, la situación académica de un estudiante al término de la duración normal de su carrera: **`dropout`** (desertor), **`enrolled`** (todavía inscrito) o **`graduate`** (graduado).

La práctica compara cinco clasificadores de la unidad: **árbol de decisión, Naive Bayes, kNN, SVM y regresión logística**. No se utilizan ensambles ni redes neuronales. La métrica principal es **F1 macro** para dar la misma importancia a las tres clases.

## Datos

Se utilizan datos de estudiantes del Instituto Politécnico de Portalegre (Portugal), derivados del conjunto *Predict Students' Dropout and Academic Success* disponible en [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697).

Los dos archivos utilizados están publicados en el [repositorio del curso](https://github.com/husseinlopez/icd2026/tree/main/datos):

- `desercion-entrenamiento.csv`: **3,539** estudiantes etiquetados; contiene `Target` y variables correspondientes al primer y segundo semestre.
- `desercion-prueba.csv`: **885** estudiantes sin `Target`; contiene únicamente los atributos disponibles al cierre del primer semestre.

Los dos archivos incluyen una columna **`id`**. Esta columna se conserva en las predicciones, pero no se usa como atributo predictivo.

Para evitar fuga de información, los modelos destinados a predecir el conjunto de prueba usan únicamente los atributos compartidos por ambos CSV. La información del segundo semestre se utiliza **solo** en el experimento comparativo de esperar más tiempo antes de tomar la decisión.

## Archivos del proyecto

```text
practica_desercion/
├──Practica3.ipynb
├── README.md
├── requirements.txt
├── predicciones.csv                # archivo final generado por el notebook
└── datos/                         # creada automáticamente si no existe
    ├── desercion-entrenamiento.csv
    └── desercion-prueba.csv
```

Los archivos `.joblib` de las búsquedas se conservan **aparte**, en Google Drive, dentro de `MyDrive/practica_desercion/`. No forman parte obligatoria del repositorio público.

## Requisitos y ejecución

- Python 3.10 o superior, con una versión de scikit-learn compatible con los checkpoints que se vayan a cargar.
- `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `jupyter`. Para manejar checkpoints se utiliza también `joblib`, incluido como dependencia de scikit-learn.


## Metodología

1. **Exploración y preparación:** revisión de clases, tipos de variables, valores faltantes y separación temporal de atributos.
2. **Preprocesamiento en `Pipeline`:** codificación one-hot de atributos nominales; imputación y escalamiento cuando el clasificador lo requiere.
3. **Optimización inicial:** `RandomizedSearchCV` con **50 configuraciones por clasificador**, **5 pliegues estratificados compartidos** y F1 macro como métrica de selección.
4. **Diagnóstico:** curvas de mejor F1 acumulado, media y desviación estándar entre pliegues, sensibilidad y especificidad por clase, tiempo de búsqueda y matriz de confusión fuera de pliegue.
5. **Comparación temporal:** evaluación del modelo ganador incluyendo información del segundo semestre, solo como análisis complementario.
6. **Ingeniería de características:** cinco atributos derivados del primer semestre, entre ellos tasas de aprobación y evaluación y una medida combinada de rendimiento.
7. **Reoptimización:** comparación de SVM y regresión logística con las nuevas características.
8. **Evaluación adicional:** validación cruzada estratificada repetida (cinco pliegues, tres repeticiones) para comparar los finalistas. Se realizó también validación anidada de regresión logística; la de SVM se consideró, pero **no se completó por su costo computacional**.
9. **Predicción final:** selección de SVM mejorado utilizando el F1 medio de la validación repetida y generación del archivo requerido para prueba.

## Resultados principales

### Búsquedas originales (primer semestre)

| Modelo | Mejor F1 macro de CV |
| --- | ---: |
| Árbol de decisión | 0.6419 |
| Naive Bayes | 0.5890 |
| kNN (corregido) | 0.5769 |
| SVM | **0.6812** |
| Regresión logística | 0.6805 |

La matriz de confusión del **SVM original** identificó correctamente 761 de 1,137 desertores; 255 desertores fueron clasificados como `enrolled` y 121 como `graduate`. La clase `enrolled` resultó particularmente difícil de predecir. **Esta matriz corresponde al SVM original, no al SVM mejorado elegido al final.**

Con información adicional del segundo semestre, el SVM pasó de **0.6812** a **0.7171** de F1 macro. El aumento no justifica usar esos atributos para la predicción temprana, pues no existen todavía en el momento de decisión ni aparecen en prueba.

### Modelos mejorados y selección

| Evaluación | SVM mejorado | Regresión logística mejorada |
| --- | ---: | ---: |
| Búsqueda con cinco nuevas características | 0.6853 | **0.6870** |
| Validación cruzada repetida, F1 medio | **0.6858** | 0.6851 |
| Validación anidada, F1 medio | No completada | 0.6868 |

En una prueba exploratoria de pesos de clase, SVM obtuvo **0.6862** con pesos personalizados, frente a **0.6853** con `balanced`. La mejora fue muy pequeña y no se consideró suficiente para sustituir la configuración más estable.

**Modelo elegido para el CSV:** **SVM mejorado** con las cinco características adicionales del primer semestre. La elección se basa en su ligera ventaja de F1 macro en la **validación repetida** (0.6858 frente a 0.6851), no en una superioridad estadística demostrada. La comparación anidada entre ambos modelos quedó incompleta.

Las puntuaciones anteriores corresponden a validación sobre el conjunto de entrenamiento. **No son resultados del conjunto de prueba**: sus etiquetas reales son desconocidas hasta que el profesor evalúe las predicciones.

## Errores encontrados y decisiones de implementación

- **kNN:** algunas configuraciones fallaron en la ruta optimizada del cálculo de distancias con etiquetas de clase de texto. Se solucionó utilizando `KNeighborsClassifier(algorithm="kd_tree")`, conservando las 50 configuraciones de búsqueda.
- **Gráficas de convergencia:** algunas curvas no se visualizaron correctamente; se corrigió el trazado con una figura explícita y un máximo acumulado que maneja los resultados `NaN`.
- **Tiempo de cómputo:** las búsquedas de SVM fueron especialmente lentas en CPU. Para no repetirlas tras desconexiones se guardaron modelos y resultados en Google Drive.
- **Validación anidada de SVM:** se intentó, pero se dejó fuera del procedimiento final por su costo y el tiempo disponible. No se presentan resultados que no llegaron a completarse.
- **Ingeniería de características:** se conservaron únicamente variables derivadas de información existente al finalizar el primer semestre, de modo que también puedan calcularse en prueba.

El notebook documenta estos problemas y decisiones en celdas de texto sin mantener celdas de diagnóstico fallidas dentro de la versión de entrega.



## Referencia

Realinho, V., Vieira Martins, M., Machado, J. y Baptista, L. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
