# Práctica 2 - procesamiento de dataset con inside airbnb

## contenido

- `practica_2_procesamiento_de_dataset`: notebook  de la práctica.
- `readme.md`: instrucciones para ejecutar el notebook.

## dataset

inside airbnb. mexico city, distrito federal, mexico listings data (15 june 2026).

archivo: `listings.csv`, versión detallada.

fuente:
https://insideairbnb.com/get-the-data/

archivo directo:
https://data.insideairbnb.com/mexico/df/mexico-city/2026-06-15/data/listings.csv.gz

licencia indicada por inside airbnb:
creative commons attribution 4.0 international (cc by 4.0).

## qué hace el notebook

el trabajo se divide en cinco partes:

1. Análisis exploratorio:
   - tamaño del dataset,
   - tipos de variables,
   - valores faltantes,
   - duplicados,
   - distribución de tipos de habitación y precio.

2. Limpieza de datos:
   - conversión de `price` a formato numérico,
   - eliminación de duplicados,
   - eliminación de registros sin precio o capacidad,
   - eliminación de precios extremos usando los percentiles 1 y 99,
   - comparación gráfica antes y después.

3. Extracción de características:
   - creación de `price_per_person`,
   - comparación entre precio total y precio por persona.
   
4. Selección de características:
   - selección de las características a estudiar,
   - comparación de columnas antes y después de la selección

5. Reducción de dimensionalidad:
   - selección de variables elegidas,
   - reemplazo de valores faltantes mediante la mediana,
   - estandarización de las variables,
   - pca con dos componentes,
   - comparación gráfica antes y después.

## cómo ejecutarlo

### opción 1: google colab

1. subir el notebook a google colab;
2. opcionalmente subir `listings.csv` a la misma sesión;
3. ejecutar las celdas en orden.


### opción 2: jupyter notebook

colocar el notebook y `listings.csv` en la misma carpeta y ejecutar todas las celdas.

## dependencias

- pandas
- numpy
- matplotlib
- scikit-learn

