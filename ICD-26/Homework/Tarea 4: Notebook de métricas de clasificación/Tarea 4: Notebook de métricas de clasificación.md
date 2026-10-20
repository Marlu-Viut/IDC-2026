# Bank Marketing - regresión logística

## contenido

Este repositorio contiene una práctica de clasificación con el dataset **Bank Marketing** de UCI Machine Learning Repository.

Archivo principal:

- `bank_marketing_modelo.ipynb`

## objetivo

El banco quiere estimar qué clientes tienen mayor probabilidad de aceptar un depósito a plazo. El notebook construye una regresión logística y revisa tanto métricas de clasificación como una decisión basada en el costo de las llamadas.

## qué se hace

1. se carga `bank-additional-full.csv`, separado por punto y coma.
2. se revisa la descripción del dataset.
3. se elimina `duration`, ya que la duración de la llamada no se conoce antes de decidir a quién llamar.
4. se separa 20 % de los datos para prueba con partición estratificada.
5. se calcula la exactitud de un clasificador que siempre responde `no`.
6. se construye un `Pipeline` con:
   - escalado de variables numéricas;
   - one-hot para variables categóricas;
   - regresión logística.
7. se calculan matriz de confusión, exactitud, precisión, sensibilidad, especificidad y f1.
8. se grafican la curva roc y la curva precisión-sensibilidad.
9. se calcula el umbral económico usando:
   - costo por llamada = 5;
   - ingreso por aceptación = 50.
10. se compara la ganancia usando umbral 0.5 y umbral 0.10.
11. se repite el modelo incluyendo `duration` para mostrar el efecto de la fuga de información.

## archivos del dataset


- `bank-additional.zip`

Si no encuentra ninguno, intenta descargar el archivo oficial de UCI.

## cómo ejecutarlo

Se puede abrir en JupyterLab o Google Colab y ejecutar las celdas de arriba hacia abajo.

Dependencias principales:

```text
numpy
pandas
matplotlib
scikit-learn
```


## fuente

Moro, S., Rita, P. y Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. DOI: 10.24432/C5K306.
