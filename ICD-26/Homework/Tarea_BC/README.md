# Tarea Breast cancer wisconsin

Este proyecto corresponde a una práctica de Ciencia de datos usando el Dataset breast cancer Wisconsin original.

## archivos

- `practica_breast_cancer.ipynb`: notebook principal.
- `breast-cancer-wisconsin.data`: dataset que debe colocarse en la misma carpeta del notebook.

## acciones

Se realizó un análisis exploratorio para revisar dimensiones, tipos de datos, valores faltantes, duplicados y distribución de clases.

Después se aplicaron tres tipos de procesamiento:

1. Limpieza de datos:
   - conversión de los valores `?` de `bare_nuclei` a valores faltantes.
   - reemplazo de esos faltantes usando la mediana.
   - eliminación de filas duplicadas exactas.
   - comparación gráfica antes y después.

2. Extracción de características:
   - creación de `mean_cell_uniformity` como promedio de `uniformity_cell_size` y `uniformity_cell_shape`.
   - comparación gráfica entre las variables originales y la característica nueva.

3. Reducción de dimensionalidad:
   - estandarización de nueve variables numéricas.
   - aplicación de pca con dos componentes.
   - comparación visual entre dos variables originales y la representación obtenida con PCA.

## Aumento de datos

No se aplicó aumento de datos. El conjunto contiene información clínica y, para esta práctica, se decidieron conservar las observaciones originales en lugar de generar registros artificiales sin una justificación médica y estadística.

## Cómo ejecutar

1. Colocar `practica_breast_cancer.ipynb` y `breast-cancer-wisconsin.data` en la misma carpeta.
2. Abrir el notebook en jupyter notebook, jupyterlab o google colab.
3. Ejecutar las celdas en orden desde la primera hasta la última.

## Librerías utilizadas

- pandas
- numpy
- matplotlib
- scikit-learn

El notebook está dividido por etapas y después de cada celda de código se incluye una interpretación corta de los resultados.
