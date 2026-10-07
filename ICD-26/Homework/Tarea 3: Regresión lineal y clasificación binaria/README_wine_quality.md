# Práctica: regresión y clasificación con Wine Quality

## Archivos

- `practica_wine_quality_regresion_clasificacion.ipynb`: notebook principal.
- `winequality-red.csv`: dataset que debe colocarse en la misma carpeta que el notebook.

## Objetivo

La práctica utiliza el dataset de calidad de vino tinto de UCI para comparar dos usos de una regresión lineal:

1. **Regresión:** predecir directamente `quality`.
2. **Clasificación:** crear `bueno = 1` cuando `quality >= 7` y utilizar la misma regresión lineal como clasificador mediante un umbral de 0.5.

La partición solicitada es 80% entrenamiento y 20% prueba con `random_state=42`.

## Qué contiene el notebook

El trabajo está dividido en bloques pequeños:

- carga de librerías;
- lectura del dataset;
- revisión básica de tipos, nulos y duplicados;
- exploración de la distribución de `quality`;
- revisión sencilla de correlaciones;
- división 80/20;
- ajuste de regresión lineal;
- cálculo de RMSE y R²;
- comparación contra predecir siempre el promedio;
- creación de la variable binaria `bueno`;
- ajuste de la misma regresión a la variable binaria;
- clasificación con umbral 0.5;
- cálculo de accuracy;
- comparación contra predecir siempre “no bueno”;
- conteo de predicciones fuera del intervalo [0, 1];
- conclusión.

Después de cada bloque de código se agregó una interpretación corta en Markdown para explicar qué se observó o qué significa el resultado, sin hacer un análisis innecesariamente avanzado.

## Cómo ejecutarlo

1. Coloca `winequality-red.csv` en la misma carpeta que el notebook.
2. Abre el `.ipynb` en Google Colab, Jupyter Notebook o VS Code.
3. Ejecuta las celdas de arriba hacia abajo.

El archivo original de UCI está separado por punto y coma (`;`), por eso se carga con:

```python
pd.read_csv("winequality-red.csv", sep=";")
```

## Dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```



## Fuente

Cortez, P., Cerdeira, A., Almeida, F., Matos, T. y Reis, J. (2009). *Wine Quality*. UCI Machine Learning Repository. DOI: 10.24432/C56S3T.
