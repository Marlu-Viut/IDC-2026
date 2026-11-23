# Diseñar una CNN compacta con Global Average Pooling
María Guadalupe Villanueva Utrilla. 


Se entrenaron diez variantes con GAP y las dos CNN de referencia. Cada corrida completó 20 épocas. Las seis primeras no alcanzaron 95 %; se agregaron cuatro variantes con mayor reducción espacial, usando solo validación para orientar la búsqueda. Se eligió la variante con menos parámetros entre las que superaron estrictamente el 95 % de validación y usaron menos de 54 330 parámetros. Después se evaluaron Flatten base, GAP base y el GAP seleccionado a 0, 2 y 4 píxeles, con los mismos pesos.

El notebook está ejecutado: contiene tablas, gráficas y comentarios breves a la derecha de las líneas más relevantes. Después de cada celda de código hay una interpretación. Los nombres de variables, comentarios y rótulos de las gráficas están en minúsculas; las clases y constantes de las bibliotecas conservan su escritura obligatoria.

## Datos y configuración

Se usa MNIST oficial. Se recuperan de `icd-21-cnn.ipynb` las particiones y los ajustes:

| ajuste | valor |
|---|---|
| entrenamiento | 8 000 imágenes, subconjunto estratificado de las 60 000 oficiales |
| validación | 1 500 imágenes del conjunto oficial de entrenamiento, sin coincidir con entrenamiento |
| prueba | 2 000 imágenes del conjunto oficial de prueba, subconjunto estratificado |
| semilla | 42 para particiones, inicialización y mezcla |
| preparación | píxeles / 255, margen negro de 4 px, canal único; entrada 36 × 36 × 1 |
| entrenamiento | 20 épocas, lote de 128, Adam, tasa constante 0.001 |
| pérdida | entropía cruzada categórica con etiquetas enteras |
| evaluación de validación | pesos al terminar la época 20 |
| selección | menor número de parámetros; en empate, mayor validación |
| desplazamiento | función original del notebook; cuatro direcciones, sin recorte ni envolvimiento |

No se aplica aumento de datos ni parada temprana. La extracción de características ocurre dentro de las convoluciones. Los ejemplos sobrantes de MNIST no se usan. Los índices están guardados en `resultados/particiones.npz`.

Las referencias originales tienen dos convoluciones de 16 filtros de 3 × 3, ReLU y padding `same`, seguidas de un max pooling 2 × 2. Después se aplica Flatten o GAP y una capa densa de diez clases. Sus conteos son 54 330 y 2 650. Se reentrenaron con las mismas 20 épocas que las variantes, porque el archivo no contiene sus pesos. La tabla histórica del enunciado permanece separada de los nuevos resultados. El texto final del notebook original menciona tres épocas, pero se siguieron las **20 épocas del código y de la rúbrica**.

## Resultados de validación

| modelo | parámetros entrenables | validación (%) | cumple |
|---|---:|---:|:---:|
| gap_1 | 1,010 | 53.20 | no |
| gap_2 | 1,674 | 70.67 | no |
| gap_3 | 3,994 | 85.93 | no |
| gap_4 | 15,466 | 90.73 | no |
| gap_5 | 3,722 | 83.07 | no |
| gap_6 | 1,674 | 60.80 | no |
| gap_7 | 15,466 | 94.80 | no |
| gap_8 | 24,714 | 96.13 | sí |
| gap_9 | 3,994 | 92.60 | no |
| gap_10 | 6,058 | 93.20 | no |

Se seleccionó **gap_8**, con **24,714 parámetros** y **96.13 % de validación**. Ahorra **29,616 parámetros (54.51 %)** frente a Flatten. Es el mínimo entre las variantes evaluadas que cumplen.

Las comparaciones de una decisión incluyen los filtros de la última convolución, agregar una capa, duplicar todos los filtros, ampliar el último kernel y mover una operación de pooling. La tabla `comparaciones.csv` registra el cambio observado en puntos porcentuales.

## Prueba y desplazamientos

| modelo | 0 px (%) | 2 px (%) | 4 px (%) | caída 0 a 4 (pp) |
|---|---:|---:|---:|---:|
| flatten_base | 97.10 | 82.75 | 37.29 | 59.81 |
| gap_base | 46.95 | 46.95 | 46.91 | 0.04 |
| gap_8 | 96.40 | 92.14 | 78.58 | 17.83 |

A 2 y 4 px se promedian las exactitudes de arriba, abajo, izquierda y derecha sobre las mismas 2 000 imágenes. La caída es la resta de exactitudes a 0 y 4 px en puntos porcentuales. Los cálculos usan valores sin redondear.

La ventaja frente a Flatten ya se aprecia a **2 px**: 92.14 % frente a 82.75 %. A 4 px la diferencia es de 41.29 puntos.

## Archivos

- `mnist_cnn_gap.ipynb`: notebook ejecutado y editable.
- `requirements.txt`: versiones usadas en esta ejecución.
- `datos/mnist.npz`: copia de MNIST oficial, para ejecutar sin descargar de nuevo.
- `modelos/`: los doce modelos completos en formato `.keras`, con sus pesos de la época 20.
- `resultados/`: arquitecturas, validación, selección, comparaciones, índices, historias y exactitudes por dirección, además de las gráficas.

El ZIP incluye todos esos archivos. Para revisar las tablas y gráficas del notebook no es necesario volver a entrenar.



## Referencias

- Notebook `icd-21-cnn.ipynb` proporcionado para la actividad.
- [MNIST en Keras](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data).
- [GlobalAveragePooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D).
- Enunciado y resultados históricos de la actividad.
