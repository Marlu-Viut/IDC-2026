# Práctica 4 · Ensambles sobre clasificadores ajustados

**Materia:** Introducción a la Ciencia de Datos  
**Estudiante:** María Guadalupe Villanueva Utrilla  
**Notebook:** `practica4_Meta_aprendizaje.ipynb`

## 1. Objetivo

En esta práctica comparo distintas estrategias de ensamble con los cinco clasificadores base de la Práctica 3 (árbol de decisión, Naive Bayes, kNN, SVM y regresión logística). La idea es revisar cuánto se gana al combinar modelos que ya fueron ajustados, sin volver a optimizar esos cinco clasificadores. Después aplico varias de las estrategias al conjunto **Bank Marketing**.

La métrica principal es **F1 macro**, porque da el mismo peso a cada clase, independientemente de su frecuencia. También reviso la sensibilidad de `enrolled` en deserción y de `yes` en Bank Marketing, la variación entre pliegues y el tiempo de entrenamiento.

## 2. Conjuntos de datos

### Deserción estudiantil

- **Origen:** [repositorio de datos del curso](https://github.com/husseinlopez/icd2026/tree/main/datos).
- **Archivos:** `desercion-entrenamiento.csv` y `desercion-prueba.csv`.
- **Registros:** 3,539 en entrenamiento y 885 en prueba.
- **Variable objetivo:** `Target`, con las clases `dropout`, `enrolled` y `graduate`.
- **Entradas:** variables disponibles hasta el primer semestre; se excluye toda variable del segundo semestre.
- **Atributos derivados:** cinco indicadores del desempeño del primer semestre, como `tasa_aprobacion` y `rendimiento_ponderado`, empleados en los experimentos que los requieren.

El notebook descarga los CSV del repositorio del curso cuando no están disponibles localmente.

### Bank Marketing

- **Origen:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).
- **Archivo:** `bank-additional-full.csv`.
- **Registros:** 41,188.
- **Entradas:** 19 atributos, después de retirar la variable objetivo y `duration`.
- **Variable objetivo:** `y`, convertida a `1` (`yes`) y `0` (`no`).
- **Distribución:** 4,640 registros `yes` (11.27 %) y 36,548 `no` (88.73 %).

Se excluye `duration` porque es la duración de la llamada: ese dato no se conocería al tomar la decisión previa a la llamada.

## 3. ¿Por qué se incluye `parametros_originales_p3.json`?

La instrucción de esta práctica es **recuperar los cinco clasificadores base con el preprocesamiento y los hiperparámetros elegidos en la Práctica 3, sin repetir su optimización**.

En el desarrollo original, las búsquedas de hiperparámetros estaban almacenadas en un archivo serializado llamado `busquedas.joblib`. Como ese archivo no siempre está disponible en otra cuenta de Colab, se extrajeron de él los parámetros ganadores del **árbol de decisión** y **Naive Bayes** y se guardaron en un JSON pequeño, legible y fácil de incluir en GitHub.

El archivo `datos_modelos_p3/parametros_originales_p3.json` contiene:

| Clasificador | Parámetros recuperados de la Práctica 3 |
|---|---|
| Árbol de decisión | `criterion='entropy'`, `max_depth=9`, `min_samples_leaf=20`, `class_weight='balanced'`, `ccp_alpha=2.0475343987736834e-05` |
| Naive Bayes | Variante `BernoulliNB`, `alpha=0.26080600945523663`, `binarize=0.25` |

**Qué hace la sección 5 del notebook:**

1. Si encuentra `busquedas.joblib` de la Práctica 3, extrae estos hiperparámetros y puede generar el JSON.
2. Si no encuentra ese checkpoint, lee el JSON desde `datos_modelos_p3/parametros_originales_p3.json`.
3. Verifica que estén los valores necesarios y reconstruye los clasificadores dentro de sus pipelines.
4. Si no encuentra ni el checkpoint ni el JSON, detiene la ejecución con un mensaje explicativo, en lugar de utilizar silenciosamente parámetros de respaldo.
5. Comprueba la firma de estos parámetros y, si han cambiado, invalida las evaluaciones guardadas del árbol, Naive Bayes y las combinaciones que dependen de ellos. No necesita borrar las búsquedas independientes de los ensambles optimizados.

**El JSON almacena configuraciones, no modelos ya entrenados.** No es necesario compartir `busquedas.joblib` con el profesor cuando el JSON se encuentra en la ruta indicada. Los modelos se ajustan en los pliegues correspondientes al ejecutar el notebook. El JSON no requiere instalar ninguna biblioteca adicional: Python incluye `json` en su biblioteca estándar.

En cuanto a kNN, SVM y regresión logística, el notebook recupera sus búsquedas guardadas cuando están disponibles; si no, usa los hiperparámetros documentados en el código de la sección 5. **Eso no significa que el JSON contenga las cinco configuraciones:** contiene únicamente las dos señaladas arriba.

## 4. Metodología

### Parte 1 · Deserción estudiantil

- Comparación de un árbol profundo y **bagging** hasta 400 árboles, para estudiar la varianza.
- Comparación de un **tocón** (`max_depth=1`) y **AdaBoost** hasta 400 rondas, para estudiar el sesgo.
- Votación dura, votación suave y **stacking** de los cinco clasificadores. El SVM se calibra dentro de los datos de entrenamiento de cada partición.
- Correlación entre errores de los modelos base.
- Optimización de **Random Forest, Extra-Trees, LightGBM y CatBoost**, con 50 configuraciones por algoritmo y gráficas de la meseta de búsqueda.
- Tabla comparativa con F1 macro (media ± desviación), sensibilidad de `enrolled`, tiempos y matriz de confusión del mejor modelo.

Se utilizan los mismos cinco pliegues estratificados:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Parte 2 · Bank Marketing

Se prueban regresión logística, árbol, Naive Bayes, Random Forest, Extra-Trees, LightGBM, CatBoost, una variante con ajuste interno del umbral y la familia que obtuvo el mejor resultado en deserción. El preprocesamiento que aprende de los datos permanece dentro del pipeline de cada clasificador.

La evaluación externa utiliza:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
```

Los resultados se calculan con `cross_validate` y F1 macro. Para las búsquedas de LightGBM y CatBoost del banco, la optimización se realiza dentro de los pliegues de entrenamiento externos. El modelo seleccionado se ajusta finalmente con todos los registros del banco y se exporta a `.joblib`.

## 5. Resultados de la ejecución documentada

Las siguientes cifras corresponden a las **salidas guardadas en el notebook antes de incorporar la recuperación obligatoria del JSON**. Sirven para documentar los experimentos realizados, pero **no deben presentarse como una nueva evaluación del árbol y Naive Bayes con los parámetros originales**. Al ejecutar de nuevo con el JSON se recalculan los resultados dependientes de esos dos modelos, y algunas posiciones o interpretaciones pueden variar.

### Deserción

| Modelo | F1 macro (media ± desviación) | Sensibilidad `enrolled` |
|---|---:|---:|
| CatBoost | **0.6975 ± 0.0092** | 0.6110 |
| LightGBM | 0.6960 ± 0.0152 | 0.6016 |
| Stacking logístico* | 0.6883 ± 0.0075 | 0.6299 |
| Regresión logística | 0.6870 ± 0.0110 | 0.5890 |
| Extra-Trees | 0.6869 ± 0.0157 | 0.5764 |
| Random Forest | 0.6861 ± 0.0132 | 0.5512 |
| SVM | 0.6853 ± 0.0062 | 0.6173 |
| Votación dura* | 0.6790 ± 0.0055 | 0.6268 |
| Votación suave* | 0.6779 ± 0.0081 | 0.6646 |
| Bagging (400 árboles) | 0.6620 ± 0.0227 | 0.3307 |
| Árbol base* | 0.6281 ± 0.0148 | 0.5449 |
| AdaBoost (400 rondas) | 0.6257 ± 0.0074 | 0.2173 |
| Árbol profundo | 0.5957 ± 0.0127 | 0.3732 |
| kNN | 0.5769 ± 0.0088 | 0.2378 |
| Tocón | 0.4920 ± 0.0135 | 0.0000 |
| Naive Bayes base* | 0.2242 ± 0.0244 | 0.9244 |

\* **Resultado que requiere actualización:** en la ejecución guardada se usaron configuraciones de respaldo para el árbol y Naive Bayes. Los resultados de estos modelos y de votación/stacking deben recalcularse con el JSON original. No se debe atribuir el F1 anterior de Naive Bayes a `BernoulliNB` optimizado.

### Bank Marketing

| Modelo | F1 macro (media ± desviación) | Sensibilidad `yes` | Tiempo de entrenamiento en CV (s) |
|---|---:|---:|---:|
| CatBoost optimizado | **0.7174 ± 0.0069** | 0.534 | 2365.891 |
| Random Forest | 0.7115 ± 0.0054 | 0.592 | 38.082 |
| LightGBM optimizado | 0.6920 ± 0.0048 | 0.624 | 413.898 |
| Extra-Trees | 0.6915 ± 0.0071 | 0.609 | 43.579 |
| Familia ganadora de deserción | 0.6872 ± 0.0040 | 0.643 | 15.026 |
| Regresión logística | 0.6739 ± 0.0045 | 0.629 | 7.181 |
| Árbol | 0.6727 ± 0.0148 | 0.631 | 2.510 |
| Naive Bayes | 0.6477 ± 0.0082 | 0.602 | 1.057 |
| LightGBM con umbral | 0.5716 ± 0.0067 | 0.114 | 29.776 |

CatBoost obtuvo el F1 macro más alto en ambos problemas durante la ejecución guardada, aunque en Bank Marketing su ventaja frente a Random Forest fue de **0.0059 puntos**, con un costo de entrenamiento bastante mayor. Las diferencias pequeñas deben interpretarse con cautela y no equivalen, por sí solas, a una prueba de superioridad estadística.

## 6. Archivos del proyecto y checkpoints opcionales

La versión actual del notebook incluye una celda nueva, situada después de la configuración de la **sección 2**, que busca y extrae automáticamente `resultados_p4.zip`. Este ZIP es **opcional**: facilita una revisión rápida, pero no reemplaza el código de los experimentos.

Para compartir el proyecto con los resultados ya calculados, se recomienda esta estructura:

```text
practica4/
├── practica4_Meta_aprendizaje.ipynb
├── README.md
├── requirements.txt
├── resultados_p4.zip                  # opcional: checkpoints de la Práctica 4
├── MaríaVillanueva-catboost_optimizado.joblib   # modelo final para entregar
└── datos_modelos_p3/
    └── parametros_originales_p3.json
```

Los archivos dentro de `resultados_p4.zip` son productos **intermedios** (`.joblib`) de las curvas, búsquedas y evaluaciones. **No son equivalentes** al archivo `MaríaVillanueva-catboost_optimizado.joblib`, que contiene el modelo final entrenado para Bank Marketing.

### Cómo preparar `resultados_p4.zip` en Colab

Generar el ZIP **después de que terminen las secciones y sus búsquedas**, para que incluya los checkpoints más recientes. Desde la carpeta de trabajo del notebook, ejecutar:

```python
from pathlib import Path
import shutil

carpeta = Path("resultados_p4")
if not carpeta.is_dir():
    raise FileNotFoundError("no existe resultados_p4")

ruta_zip = shutil.make_archive(
    "resultados_p4", "zip", root_dir=".", base_dir="resultados_p4"
)
print("archivo generado:", ruta_zip)
print("tamaño (MB):", round(Path(ruta_zip).stat().st_size / 1024**2, 2))
```

Para descargarlo en Colab:

```python
from google.colab import files
files.download("resultados_p4.zip")
```

**Precaución:** el ZIP debe prepararse con los experimentos definitivos y los parámetros del JSON original de la Práctica 3. Si algún checkpoint procede de una ejecución anterior, el hecho de que exista no garantiza que corresponda a la configuración actual. El código contiene invalidación para algunas evaluaciones dependientes de los parámetros de la Práctica 3, pero no comprueba automáticamente la procedencia de cada archivo del ZIP.

## 7. Requisitos y entorno

El entorno utilizado para generar las salidas anteriores registró las siguientes versiones:

| Componente | Versión registrada |
|---|---|
| Python | 3.13.18 |
| scikit-learn | 1.6.1 |
| pandas | 2.2.3 |
| NumPy | 2.4.6 |
| SciPy | 1.16.3 |
| LightGBM | 4.6.0 |
| CatBoost | 1.2.10 |
| joblib | 1.6.0 |

El archivo `requirements.txt` indica las dependencias y los intervalos de versión aceptados por el notebook. Para instalar en un entorno local:

```bash
python -m pip install -r requirements.txt
```

Para revisar las versiones instaladas puedes ejecutar la primera celda del notebook. En Google Colab, parte de estas bibliotecas ya viene incluida. **El JSON no aparece en `requirements.txt` porque no es una dependencia instalable.**

## 8. Cómo ejecutar el notebook

### Opción A: revisión rápida con checkpoints

1. Clonar o descargar el proyecto; mantener `datos_modelos_p3/parametros_originales_p3.json` en la ruta indicada.
2. Colocar **`resultados_p4.zip` en la misma carpeta de trabajo desde la que se ejecuta el notebook**. En Colab es necesario subir ese archivo al entorno; tenerlo únicamente en GitHub o en Drive no lo descarga automáticamente.
3. Instalar las dependencias cuando corresponda: `python -m pip install -r requirements.txt`.
4. Mantener **`usar_drive = False`** en la sección 2. Esta variable controla la copia a una cuenta personal de Google Drive; **no** controla la lectura del ZIP local.
5. Ejecutar **Restart & Run All**. La celda añadida tras la sección 2 extraerá del ZIP los archivos `.joblib` a `resultados_p4/`, siempre que todavía no existan localmente, y mostrará cuántos recuperó.
6. Las funciones de recuperación leerán esos archivos; cada resultado **que falte o no pueda cargarse** se volverá a calcular y guardar. La sección del modelo final puede volver a serializar el modelo ya recuperado.

El notebook admite dos formas de crear el ZIP: con la carpeta `resultados_p4/` dentro o con los `.joblib` directamente en la raíz. Si ya existe localmente un checkpoint con el mismo nombre, **la extracción no lo sobrescribe**, aunque sea anterior; para una prueba limpia conviene comenzar sin una carpeta local `resultados_p4/` preexistente.

### Opción B: reproducción íntegra de los entrenamientos

1. Conservar notebook, README, `requirements.txt` y el JSON de la Práctica 3.
2. No colocar el ZIP (o retirarlo de la carpeta de trabajo) y comenzar con `resultados_p4/` vacía o inexistente.
3. Mantener `usar_drive = False` y ejecutar **Restart & Run All** desde la raíz del proyecto, con conexión a Internet para descargar los dos conjuntos de datos.
4. Esperar las búsquedas de 50 configuraciones y cinco pliegues, el stacking y los entrenamientos del banco. La ejecución puede tardar bastante, especialmente en CPU.
5. Revisar las tablas, las matrices de confusión y el `.joblib` final generado en la raíz del proyecto.

**Diferencia importante:** la opción rápida **reutiliza resultados calculados previamente**; solo la opción completa reproduce todos los entrenamientos desde cero. Los checkpoints pueden presentar incompatibilidades si se cargan con otras versiones de scikit-learn, CatBoost, LightGBM u otras dependencias. Los archivos `.joblib` deben provenir de una fuente confiable.

**Sobre las interpretaciones:** varias celdas `###Revisión` documentan una ejecución anterior y son texto fijo. Si las puntuaciones cambian al recalcular algún modelo, esas interpretaciones deben contrastarse con las tablas nuevas.

**Estado de la prueba:** se revisó la lógica de extracción del ZIP y las rutas del notebook, pero **no se ha comprobado una ejecución completa de Restart & Run All en una sesión limpia**. La disponibilidad del JSON y del ZIP reduce dependencias y tiempo de cómputo, pero no garantiza por sí sola una ejecución libre de errores.

## 9. Modelo final y entrega

La entrega solicitada por el curso incluye el **enlace al repositorio**, el **notebook** y un archivo **`.joblib` con el mejor modelo de Bank Marketing**. Si se decide compartir checkpoints para acelerar la revisión, agregar también `resultados_p4.zip` como recurso opcional del repositorio o proporcionar un enlace de descarga accesible. El archivo `.joblib` se entrega por separado en la plataforma según las instrucciones del profesor; no sustituye al JSON.

La diferencia es:

- `parametros_originales_p3.json`: hiperparámetros del **árbol** y **Naive Bayes** de la Práctica 3; permite reconstruir esos modelos base.
- `MaríaVillanueva-catboost_optimizado.joblib`: **modelo final entrenado para Bank Marketing**, producto de la Práctica 4; sirve para cargarlo y generar predicciones sin volver a ajustarlo.

Los archivos `.joblib` se deben cargar únicamente de fuentes confiables, ya que son objetos serializados de Python. Si el ZIP excede el límite de archivos de GitHub, alojarlo como recurso descargable y documentar su ubicación; en ese caso, el ZIP **debe colocarse manualmente** junto al notebook antes de ejecutar la opción rápida.

## 10. Limitaciones y conclusiones

- Bagging y AdaBoost mejoraron claramente a sus árboles individuales en la ejecución guardada, pero no todos los ensambles superaron a los clasificadores fuertes ya ajustados.
- En deserción, `enrolled` siguió siendo una clase difícil de recuperar; en Bank Marketing, `yes` representa una proporción minoritaria.
- La comparación de familias y tiempos depende del conjunto de datos, los pliegues, el hardware y las versiones de las bibliotecas.
- En deserción, buscar hiperparámetros y reportar su rendimiento con los mismos pliegues puede producir resultados optimistas; la comparación no reemplaza una validación externa independiente.
- El JSON conserva los **valores de hiperparámetros** exportados; no demuestra por sí mismo la procedencia histórica de `busquedas.joblib` ni almacena los pesos entrenados.

En los resultados registrados, CatBoost obtuvo el F1 macro más alto en los dos conjuntos, pero con ventajas relativamente pequeñas frente a otros métodos. El aporte principal de la práctica es mostrar que los ensambles pueden ayudar de distintas maneras y que su utilidad depende tanto del problema como del costo y la forma de evaluación.

## 11. Fuentes de los datos

- Deserción estudiantil: https://github.com/husseinlopez/icd2026/tree/main/datos
- Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing
