# 🎵 Inteligencia Musical y Predicción de Popularidad de Canciones

**Proyecto:** Evaluación Parcial N.º 1 — Machine Learning (MLY1101)  
**Caso:** C — Spotify Tracks  
**Institución:** Duoc UC  
**Metodología:** CRISP-DM  
**Fecha:** 2026  

---

> **Actualización de preparación de datos:** columnas en español desde la carga;
> eliminación de filas sin artista/canción y con ceros en las métricas de audio
> indicadas; álbum desconocido imputado; compás imputado con moda de entrenamiento.
>
> **Pendiente de ejecución real:** el CSV no se adjuntó a esta revisión. Las cifras
> estáticas conservadas en este informe pertenecen a la versión anterior y **no se
> han revalidado** con la nueva limpieza. No usar sus porcentajes, correlaciones o
> tamaños de conjuntos en la presentación sin recalcularlos. El notebook entrega
> los nuevos conteos automáticamente. Las pruebas de código usan datos sintéticos
> separados, nunca presentados como observaciones de Spotify.

## Cómo ejecutar esta versión

1. Instalar las dependencias con `pip install -r requirements.txt`.
2. Ubicar `Spotify_Tracks_Dataset.csv` en `data/raw/`, junto al notebook o indicar
   su ubicación en `RUTA_DATOS`, en la sección 0.
3. Abrir `EP1_Spotify_ML_EDA_Preparacion_COMPLETO.ipynb`, reiniciar el kernel y
   ejecutar todas las celdas en orden.
4. Revisar `balance_limpieza.csv`, la auditoría y los gráficos de `images/`.

Se requiere **scikit-learn ≥ 1.2**, porque el notebook usa `sparse_output` en
`OneHotEncoder`. [Documentación de OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

Los datos y tablas usan identificadores españoles sin espacios ni tildes; los
gráficos usan etiquetas con tildes y unidades. Los nombres propios de canciones,
artistas y géneros se conservan. El CSV original no se sobrescribe.

## Tabla de contenidos

1. [Descripción del problema de negocio](#1-descripción-del-problema-de-negocio)
2. [Objetivos del proyecto](#2-objetivos-del-proyecto)
3. [KPIs](#3-kpis)
4. [Fuentes de datos y herramientas colaborativas](#4-fuentes-de-datos-y-herramientas-colaborativas)
5. [Metodología CRISP-DM](#5-metodología-crisp-dm)
6. [Descripción del dataset](#6-descripción-del-dataset)
7. [Análisis exploratorio de datos (EDA)](#7-análisis-exploratorio-de-datos-eda)
8. [Calidad de datos](#8-calidad-de-datos)
9. [Tratamiento de anomalías y valores faltantes](#9-tratamiento-de-anomalías-y-valores-faltantes)
10. [Preparación y transformación de datos](#10-preparación-y-transformación-de-datos)
11. [Variables relevantes para Machine Learning](#11-variables-relevantes-para-machine-learning)
12. [Sesgos, ética y privacidad](#12-sesgos-ética-y-privacidad)
13. [Hallazgos clave](#13-hallazgos-clave)
14. [Conclusiones](#14-conclusiones)
15. [Trabajo futuro](#15-trabajo-futuro)
16. [Análisis exploratorio avanzado](#16-análisis-exploratorio-avanzado)
17. [Auditoría contra la rúbrica](#17-auditoría-contra-la-rúbrica)
18. [Preguntas de defensa oral](#18-preguntas-de-defensa-oral)

---

## 1. Descripción del problema de negocio

### Contexto

Spotify es la plataforma de streaming musical más grande del mundo, con más de 600 millones de usuarios activos mensuales. La plataforma procesa millones de canciones de decenas de miles de artistas y géneros. En este contexto, la capacidad de anticipar qué canciones alcanzarán mayor popularidad tiene un valor estratégico enorme: permite optimizar decisiones de curaduría editorial, promoción algorítmica, contratos con sellos discográficos y diseño de playlists.

### Problema

Un equipo de inteligencia musical necesita determinar si los **atributos medibles de audio de una canción** (bailabilidad, energia, volumen_db, tempo_bpm, entre otros) permiten **anticipar su nivel de popularidad** (`popularidad`, escala 0–100). El problema tiene dos dimensiones:

- **Técnica:** ¿Los features de audio son suficientes para predecir popularidad? ¿Qué tanto explican?
- **Ética:** ¿Usar un modelo de predicción para priorizar canciones puede reforzar sesgos de exposición?

### Relevancia de negocio

Si el modelo logra identificar canciones con alto potencial de popularidad, se pueden:
- Priorizar recursos de promoción hacia canciones con mayor potencial.
- Diseñar playlists algorítmicas más efectivas.
- Detectar artistas emergentes antes de que alcancen masividad.
- Apoyar decisiones editoriales de manera más objetiva y auditable.

---

## 2. Objetivos del proyecto

### Objetivo general

Construir una base analítica reproducible para desarrollar un modelo de Machine Learning capaz de estimar el nivel de popularidad de canciones de Spotify a partir de sus atributos musicales medibles.

### Objetivo de negocio

Apoyar las decisiones de curaduría, promoción y recomendación musical mediante un sistema que identifique, con base en datos objetivos, canciones con potencial de alto alcance.

### Objetivo analítico

Mediante análisis exploratorio, determinar qué atributos musicales se asocian con mayor popularidad, identificar patrones por género, detectar problemas de calidad en los datos y preparar un pipeline reproducible para modelamiento futuro.

### Objetivo de Machine Learning

Desarrollar un modelo supervisado de **regresión** que prediga el valor numérico de `popularidad` (0–100) a partir de los atributos de audio y género de cada canción, utilizando variables no derivadas de popularidad para evitar data leakage.

---

## 3. KPIs

---

### 3.1 Clasificación de los KPIs

Los KPIs del proyecto se organizan en tres categorías con propósitos distintos:

**KPIs de calidad de datos**
Miden si el dataset está en condiciones de ser usado para entrenar un modelo. Si estos fallan, el resto del proyecto pierde sentido, porque un modelo entrenado con datos sucios produce resultados que no se pueden confiar. Son verificables ahora mismo, sin necesidad de entrenar ningún modelo.

**KPIs de desempeño del modelo**
Miden qué tan bien predice el modelo una vez entrenado. Son métricas técnicas estándar en Machine Learning. En este proyecto aún corresponden a una **etapa futura (EP2)**: se definen ahora como metas de referencia, pero no pueden ser validadas hasta que exista un modelo entrenado y evaluado sobre el conjunto de prueba.

**KPIs de utilidad para el negocio**
Miden si el modelo realmente sirve para resolver el problema original: identificar canciones con alto potencial de popularidad. Una métrica técnica puede verse aceptable en papel pero ser inútil para el negocio, por eso este tipo de KPI es necesario como complemento.

---

### 3.2 Tabla resumen

| KPI | Tipo | Qué mide | Fórmula / cálculo | Meta | Estado actual |
|-----|------|----------|-------------------|------|---------------|
| Cobertura de datos | Calidad de datos | % de filas completas en variables clave | `(filas sin nulos / total filas) × 100` | ≥ 99% | Pendiente de recalcular con esta versión |
| Tasa de nulos | Calidad de datos | % de celdas con valor faltante | `(celdas nulas / total celdas) × 100` | < 1% | Pendiente de recalcular con esta versión |
| Contaminación train/test | Calidad de datos | % de `id_cancion` que aparecen en train Y en test | `(id_cancion solapados / total id_cancion) × 100` | 0% | Pendiente de recalcular con esta versión |
| MAE | Desempeño del modelo | Error promedio de predicción en puntos de popularidad | `promedio(|valor_real − valor_predicho|)` | < 10 puntos | 🔄 Pendiente de modelamiento |
| RMSE | Desempeño del modelo | Error promedio penalizando errores grandes | `√(promedio((valor_real − valor_predicho)²))` | < 15 puntos | 🔄 Pendiente de modelamiento |
| R² | Desempeño del modelo | Proporción de varianza de popularidad explicada por el modelo | `1 − (suma errores modelo / suma errores baseline)` | > 0.20 | 🔄 Pendiente de modelamiento |
| Recall top cuartil | Utilidad para el negocio | % de canciones realmente populares que el modelo logra identificar | `canciones populares detectadas / total canciones populares reales` | > 0.60 | 🔄 Pendiente de modelamiento |

> **Importante:** las metas de MAE, RMSE, R² y Recall son **referencias iniciales** definidas antes de entrenar el modelo. Deben compararse contra el baseline (predecir siempre la media de 33.24) una vez que exista un modelo entrenado. No representan estándares universales ni valores demostrados.

---

### 3.3 Explicación detallada de cada KPI

---

#### KPI 1 — Cobertura de datos
**Tipo:** Calidad de datos

**Qué mide:** el porcentaje de filas del dataset que tienen todos sus valores completos en las variables clave para el análisis (features de audio y target `popularidad`).

**Para qué sirve en este proyecto:** garantiza que el modelo va a entrenarse con registros completos. Si muchas filas tuvieran datos faltantes en `bailabilidad`, `energia` o `popularidad`, el modelo aprendería de una muestra sesgada o incompleta.

**Cómo se calcula:**
```
Cobertura = (filas sin ningún nulo en variables clave / total de filas) × 100
```

**Cómo interpretar el resultado:** un 100% significa que todos los registros están completos. Un valor bajo (por ejemplo, 70%) significaría que casi un tercio del dataset no podría usarse directamente.

**Buen resultado vs. mal resultado:**
- Bueno: ≥ 99% — el dataset es prácticamente completo.
- Malo: < 90% — se pierde demasiada información o se necesita imputación masiva.

**Meta del proyecto:** ≥ 99%.

**Por qué ese umbral:** es una meta conservadora pero realista para datasets de APIs externas bien mantenidas. El umbral del 99% no es un estándar universal, sino una expectativa razonable dado el origen (API oficial de Spotify).

**Estado de la versión actual:** cobertura pendiente de recalcular. El informe anterior mezclaba la cobertura de metadatos de texto con la de predictores. La auditoría actual informa nulos por variable y registra por separado ceros y símbolos «?».

**Relación con el negocio:** un dataset casi completo significa que el modelo podrá aprender de prácticamente todas las canciones disponibles, sin descartarlas por falta de información.

---

#### KPI 2 — Tasa de nulos
**Tipo:** Calidad de datos

**Qué mide:** el porcentaje de celdas individuales con valor faltante sobre el total de celdas del dataset.

**Para qué sirve en este proyecto:** complementa la cobertura de datos, pero a nivel de celda. Una fila puede tener solo 1 valor faltante de 20, lo que no siempre implica descartarla. Esta métrica ayuda a dimensionar el problema de forma más fina.

**Cómo se calcula:**
```
Tasa de nulos = (total de celdas con NaN / total de celdas) × 100
Total de celdas = 114,000 filas × 21 columnas = 2,394,000 celdas
```

**Buen resultado vs. mal resultado:**
- Bueno: < 1% — los nulos son marginales y manejables.
- Malo: > 10% — se necesita una estrategia de imputación seria.

**Meta del proyecto:** < 1%.

**Por qué ese umbral:** es una referencia inicial razonable para este tipo de dataset de audio. No es un estándar demostrado estadísticamente para este problema en particular.

**Resultado consignado en el informe previo (no revalidado):** solo **3 celdas son nulas** de 2,394,000. La tasa real es **0.000125%**, prácticamente cero. Además, la única fila afectada tiene un detalle relevante: `duracion_ms = 0` y `popularidad = 0`, lo que sugiere que es un registro de una canción eliminada de Spotify entre la extracción de features y la de metadatos. Se eliminó esa fila.

**Relación con el negocio:** datos casi sin nulos significan menos decisiones arbitrarias de imputación, lo que reduce el riesgo de introducir sesgos artificiales en el modelo.

---

#### KPI 3 — Contaminación train/test
**Tipo:** Calidad de datos

**Qué mide:** si alguna canción aparece al mismo tiempo en el conjunto de entrenamiento y en el conjunto de prueba del modelo. Si esto ocurre, el modelo "ya vio" esa canción antes de ser evaluado, lo que infla artificialmente las métricas.

**Para qué sirve en este proyecto:** el dataset tiene 114,000 filas pero solo 89,741 canciones únicas (`id_cancion` únicos). Hay 24.259 filas adicionales respecto de los IDs únicos; esa diferencia no es la cantidad de canciones repetidas. Si una de esas canciones cae en train con el género "pop" y en test con el género "k-pop", el modelo se beneficia de haberla visto antes. Esto se llama **data leakage** y hace que el modelo parezca mejor de lo que realmente es.

**Cómo se calcula:**
```
Contaminación = (id_cancion que aparecen en train Y en test / total id_cancion únicos) × 100
```

**Buen resultado vs. mal resultado:**
- Bueno: 0% — ninguna canción se repite entre train y test.
- Malo: cualquier valor > 0% — existe leakage.

**Meta del proyecto:** 0% (obligatorio, no negociable).

**Por qué ese umbral:** no es una meta "aspiracional" como las métricas del modelo. Es una condición de validez básica del experimento. Un modelo evaluado con leakage no mide nada útil.

**Cómo se logró:** se usó `GroupShuffleSplit` de scikit-learn, agrupando por `id_cancion`. Esto garantiza que todas las apariciones de una misma canción queden en el mismo conjunto.

**Resultado consignado en el informe previo (no revalidado):** verificado con código. `id_cancion` compartidos entre train y test: **0**. ✅

**Relación con el negocio:** si el modelo se evalúa con leakage, las métricas de negocio (¿el modelo realmente identifica canciones populares nuevas?) serán falsas. La contaminación cero es condición previa para que cualquier otro KPI sea confiable.

---

#### KPI 4 — MAE (Error Absoluto Medio)
**Tipo:** Desempeño del modelo  
**Estado:** pendiente de modelamiento (EP2)

**Qué mide:** el error promedio del modelo en puntos de popularidad. Si el modelo predice 45 y la canción real tiene 60, el error es 15 puntos. El MAE promedia todos esos errores individuales.

**Para qué sirve en este proyecto:** permite saber, en términos concretos y directos, qué tan lejos están las predicciones del modelo de la realidad. Es la métrica más fácil de explicar e interpretar.

**Cómo se calcula:**
```
MAE = promedio(|popularity_real − popularity_predicha|)
```

Por ejemplo: si el modelo comete errores de 8, 12, 5 y 15 puntos en 4 canciones:
```
MAE = (8 + 12 + 5 + 15) / 4 = 10 puntos
```

**Cómo interpretar el resultado:** un MAE de 10 significa que, en promedio, el modelo se equivoca en 10 puntos de popularidad. En una escala de 0 a 100, eso equivale a un error del 10%.

**Buen resultado vs. mal resultado:**
- Bueno: < 10 puntos — error promedio menor al 10% de la escala.
- Malo: > 20 puntos — el modelo se equivoca tanto que no aporta valor sobre simplemente predecir la media.

**Meta del proyecto:** < 10 puntos (meta inicial de referencia).

**Por qué ese umbral:** es una meta razonable pero no demostrada. Debe compararse contra el **baseline**: un modelo que predice siempre 33.24 (la media del dataset) tendría un MAE de aproximadamente 17–18 puntos. Superar ese baseline es el primer objetivo real; el umbral de 10 puntos es una aspiración más ambiciosa para etapas posteriores.

**Diferencia con RMSE:** el MAE trata todos los errores por igual. Un error de 20 puntos no pesa más que dos errores de 10 puntos.

**Relación con el negocio:** un MAE bajo significa que las estimaciones de popularidad son confiables para priorizar canciones. Un MAE alto implica que la predicción es tan incierta que no conviene usarla para tomar decisiones.

---

#### KPI 5 — RMSE (Raíz del Error Cuadrático Medio)
**Tipo:** Desempeño del modelo  
**Estado:** pendiente de modelamiento (EP2)

**Qué mide:** similar al MAE, pero penaliza más los errores grandes. Está en la misma escala que la popularidad (0 a 100), lo que facilita su interpretación.

**Para qué sirve en este proyecto:** detecta si el modelo comete errores muy grandes en algunos casos, aunque en promedio parezca razonable. En el contexto de Spotify, un error puntual de 50 puntos (predecir popularidad 20 para una canción que tiene 70) puede significar perder una canción con alto potencial.

**Cómo se calcula:**
```
RMSE = √(promedio((popularity_real − popularity_predicha)²))
```

Por ejemplo, con los mismos 4 errores (8, 12, 5, 15):
```
RMSE = √((64 + 144 + 25 + 225) / 4) = √(458 / 4) = √114.5 ≈ 10.7 puntos
```

El RMSE (10.7) es mayor que el MAE (10) porque el error de 15 puntos "pesa" más al elevar al cuadrado.

**Cómo interpretar el resultado:** si el RMSE es notablemente mayor que el MAE, significa que el modelo tiene algunos errores muy grandes que el MAE "disimula". Un RMSE similar al MAE indica errores más uniformes.

**Buen resultado vs. mal resultado:**
- Bueno: < 15 puntos.
- Malo: > 25 puntos o RMSE muy superior al MAE (errores extremos frecuentes).

**Meta del proyecto:** < 15 puntos (meta inicial de referencia, debe validarse contra el baseline).

**Relación con el negocio:** desde el punto de vista del negocio, los errores grandes son especialmente costosos. Perder una canción con popularidad 80 porque el modelo predijo 30 tiene un impacto mayor que cometer muchos errores pequeños.

---

#### KPI 6 — R² (Coeficiente de Determinación)
**Tipo:** Desempeño del modelo  
**Estado:** pendiente de modelamiento (EP2)

**Qué mide:** qué proporción de la variación de popularidad entre canciones logra explicar el modelo. Un R² de 1.0 significaría predicción perfecta. Un R² de 0 significa que el modelo no explica nada más de lo que explica simplemente predecir siempre la media.

**Para qué sirve en este proyecto:** permite saber si el modelo agrega valor real sobre el baseline más simple posible. Si el R² es negativo o cercano a cero, los features de audio prácticamente no explican la popularidad.

**Cómo se calcula:**
```
R² = 1 − (suma de errores al cuadrado del modelo / suma de errores al cuadrado del baseline)
```

El baseline debe predecir la media calculada solo con entrenamiento; 33.24 era una referencia del análisis previo. Si el modelo comete los mismos errores que ese baseline, R² = 0. Si los comete peor, R² es negativo.

**Cómo interpretar el resultado:**
- R² = 0.25: el modelo explica el 25% de la variación de popularidad entre canciones.
- R² = 0.0: el modelo no mejora en nada predecir siempre la media.
- R² negativo: el modelo es peor que no hacer nada.

**Buen resultado vs. mal resultado:**
- Bueno para este problema: > 0.20, es decir, que el modelo explique al menos el 20% de la variación.
- Malo: ≤ 0 — el modelo no aporta nada.

**Meta del proyecto:** > 0.20 (meta inicial de referencia).

**Por qué ese umbral y por qué es bajo:** las correlaciones lineales entre los features de audio y `popularidad` son todas menores a 0.10 en valor absoluto (la más fuerte es `instrumentalidad` con -0.095). Esto indica que predecir popularidad a partir de audio es un problema difícil. Un R² de 0.20 ya sería un resultado interesante en este contexto. Exigir R² > 0.80 sería irreal dado lo que los datos permiten.

**Importante — R² no es suficiente solo:** un modelo puede tener R² = 0.25 pero cometer errores enormes justo en las canciones más populares, que son las que más importan para el negocio. Por eso el R² debe complementarse con el Recall del top cuartil.

**Relación con el negocio:** el R² mide el desempeño global del modelo, pero desde el punto de vista de la curaduría musical puede ser más relevante saber si el modelo identifica bien las canciones del top, no si predice bien el promedio.

---

#### KPI 7 — Recall del top cuartil de popularidad
**Tipo:** Utilidad para el negocio  
**Estado:** pendiente de modelamiento (EP2)

**Qué mide:** de todas las canciones que realmente son muy populares (popularidad ≥ 50, que corresponde al percentil 75 del dataset), qué porcentaje logra el modelo identificar correctamente como "de alto potencial".

**Para qué sirve en este proyecto:** es el KPI más directamente ligado al problema de negocio. Si el equipo de curaduría quiere priorizar las canciones con mayor potencial, necesita saber cuántas de las realmente buenas está detectando el sistema, no cuántas canciones promedio predice bien.

**Cómo se calcula:** se convierte la predicción en una decisión binaria: si el modelo predice popularidad ≥ 50, clasifica la canción como "potencialmente popular".

```
Recall = canciones con popularity_real ≥ 50 que el modelo predijo ≥ 50
         ─────────────────────────────────────────────────────────────
                  total de canciones con popularity_real ≥ 50
```

**Ejemplo simple:** si en el conjunto de prueba hay 100 canciones realmente populares (popularidad ≥ 50) y el modelo predice correctamente que 70 de ellas son populares, entonces:
```
Recall = 70 / 100 = 0.70 → 70%
```
El modelo detectó el 70% de las canciones importantes y dejó pasar el 30%.

**En el dataset real:** el percentil 75 de `popularidad` es 50. Hay aproximadamente 29,367 canciones con popularidad ≥ 50 en el dataset completo (25.8% del total). En el conjunto de prueba (20% del dataset), serían aproximadamente 5,800 canciones del top cuartil.

**Buen resultado vs. mal resultado:**
- Bueno: > 0.60 — el modelo detecta más del 60% de las canciones populares.
- Malo: < 0.40 — el modelo deja pasar la mayoría de las canciones con potencial.

**Meta del proyecto:** > 0.60 (meta inicial de referencia, debe validarse contra el baseline).

**Por qué ese umbral:** un sistema que detecta al menos 6 de cada 10 canciones populares ya es útil como herramienta de priorización. Por debajo de 0.40 probablemente no supera lo que un curador experto haría manualmente. El umbral de 0.60 es una referencia razonable, no un estándar estadístico establecido para este dominio.

**Relación con el negocio:** este KPI responde directamente la pregunta del cliente: "¿el modelo me ayuda a encontrar las canciones con más potencial, o estoy perdiendo tiempo usándolo?" Un Recall bajo significa que el sistema deja pasar demasiadas oportunidades.

---

### 3.4 Interpretación conjunta

**¿Por qué necesitamos siete KPIs y no uno solo?**

Porque cada métrica mide un aspecto diferente del problema, y ninguna sola es suficiente:

- Los **KPIs de calidad** (cobertura, nulos, contaminación) son condiciones previas. Si fallan, los resultados del modelo no tienen validez, sin importar lo bien que se vean las otras métricas.

- El **MAE y el RMSE** miden el error promedio del modelo, pero en escala global. Un modelo puede tener MAE razonable pero cometer errores enormes justo en las canciones más importantes.

- El **R²** indica si el modelo explica algo de la variación de popularidad, pero no dice nada sobre si identifica bien las canciones del top.

- El **Recall del top cuartil** es el que más se acerca a la pregunta del negocio, pero no informa sobre la magnitud de los errores en el resto del dataset.

Un sistema de evaluación completo necesita los tres tipos. Por ejemplo: un modelo podría tener R² = 0.25 (razonable) pero Recall = 0.30 (muy malo), lo que significaría que es un modelo técnicamente mediocre para el propósito real del proyecto. O podría tener MAE bajo pero Recall alto solo porque predice todo como popular, lo que también sería un problema.

La combinación de los siete KPIs permite tener una visión honesta y completa del desempeño del proyecto.

---

### 3.5 Versión para defensa oral

> Texto de referencia anterior: sustituir las cifras de este guion por los resultados de la nueva ejecución.

> Aproximadamente 1 minuto — en lenguaje directo, sin leer fórmulas.

---

*"Nuestro proyecto tiene dos tipos de KPIs. Los primeros tres miden la calidad de los datos, y ya los tenemos calculados: el dataset tiene prácticamente cero valores faltantes, el 99.999% de las filas están completas, y verificamos que ninguna canción aparece al mismo tiempo en entrenamiento y prueba, lo que garantiza que el modelo no hace trampa al evaluarse.*

*Los otros cuatro KPIs son para el modelo, que todavía no hemos entrenado, así que son metas de referencia para la próxima etapa. El MAE y el RMSE nos van a decir, en puntos de popularidad, qué tan lejos están las predicciones de la realidad. La diferencia es que el RMSE penaliza más cuando el modelo se equivoca por mucho en una canción concreta.*

*El R² nos dice si el modelo explica algo de por qué unas canciones son más populares que otras, o si básicamente da lo mismo que predecir siempre el promedio, que en nuestro dataset es 33 puntos.*

*Y el Recall del top cuartil es el más importante desde el punto de vista del negocio: de las canciones que realmente son las más populares, ¿cuántas logra detectar el sistema? Si hay 100 canciones populares y el modelo identifica 70, el Recall es del 70%. Eso es lo que realmente le importa a un equipo de curaduría musical.*

*Definimos estas metas antes de entrenar el modelo, por eso son referencias iniciales. El verdadero juicio viene cuando las comparemos contra el baseline, que es simplemente predecir siempre el promedio de popularidad."*

---

## 4. Fuentes de datos y herramientas colaborativas

### Fuente de datos

| Atributo | Valor |
|----------|-------|
| **Nombre del archivo** | `Spotify_Tracks_Dataset.csv` |
| **Origen declarado** | Kaggle — *Spotify Tracks Dataset* (asociado a MaharshiPandya) |
| **Tamaño** | 20.1 MB |
| **Registros** | 114,000 filas × 21 columnas |
| **Fecha de extracción** | No declarada en el CSV (limitación) |
| **Versión de API** | No declarada (limitación) |
| **Licencia** | Pública para uso académico en Kaggle |

**Limitación importante:** el archivo no incluye metadatos de extracción (fecha, versión de API de Spotify). Esto impide verificar si los valores de popularidad están actualizados, ya que ese indicador varía en el tiempo según los hábitos de reproducción de los usuarios.

### Herramientas colaborativas

| Herramienta | Uso |
|-------------|-----|
| **Python / pandas / numpy** | Manipulación y análisis de datos |
| **matplotlib / seaborn** | Visualización |
| **scikit-learn** | Pipeline de preprocesamiento y splits |
| **Jupyter Notebook** | Integración de código, análisis y documentación |
| **Git / GitHub** | Control de versiones y trazabilidad del trabajo |
| **Google Colab** | Ejecución compartida entre integrantes sin mismo equipo |

---

## 5. Metodología CRISP-DM

| Fase | Alcance |
|---|---|
| Comprensión del negocio | EP1 |
| Comprensión de datos | EP1 |
| Preparación | EP1 |
| Modelamiento | EP2, pendiente |
| Evaluación | EP2, pendiente |
| Despliegue | Etapa posterior |

### Fase 1: Business Understanding ✅

Se identificó el problema de negocio (anticipar popularidad de canciones), se definieron objetivos analíticos y de ML, y se establecieron los KPIs de calidad y de desempeño futuro del modelo. Ver secciones 1–3.

### Fase 2: Data Understanding ✅

Se inspeccionaron las 114,000 filas × 21 columnas del dataset. Se calcularon estadísticas descriptivas completas, se identificaron valores faltantes, duplicados conceptuales, outliers y relaciones entre variables. Ver secciones 6–11.

### Fase 3: Data Preparation ✅

La nueva preparación implementa las reglas de la sección 9 y crea `duracion_min`. Reserva una partición por `id_cancion`, imputa compás con la moda de entrenamiento y prepara `StandardScaler`, `SimpleImputer` y `OneHotEncoder`. El número real de filas eliminadas se obtiene al ejecutar el notebook con el CSV.

### Fase 4: Modeling 🔄 (Trabajo futuro)

Entrenar modelos de regresión (Ridge, Random Forest, Gradient Boosting) y comparar contra baseline de media constante. Evaluar MAE, RMSE y R².

### Fase 5: Evaluation 🔄 (Trabajo futuro)

Evaluar desempeño global y por género. Analizar si el modelo tiene desempeño diferencial por género (sesgo de modelo). Usar validación cruzada con grupos.

### Fase 6: Deployment 🔄 (Trabajo futuro)

Implementar scoring reproducible. Monitorear drift en popularidad. Documentar limitaciones para usuarios finales.

---

## 6. Descripción del dataset

### ¿Qué representa cada observación?

Cada fila corresponde a una canción registrada en Spotify con sus atributos de audio calculados por la API de Spotify. Algunas canciones aparecen múltiples veces si fueron categorizadas en más de un género.

### Nombres originales y nombres en español

| Original | Español |
|---|---|
| `track_id` | `id_cancion` |
| `artists` | `artistas` |
| `album_name` | `nombre_album` |
| `track_name` | `nombre_cancion` |
| `popularity` | `popularidad` |
| `duration_ms` | `duracion_ms` |
| `explicit` | `contenido_explicito` |
| `danceability` | `bailabilidad` |
| `energy` | `energia` |
| `key` | `tonalidad` |
| `loudness` | `volumen_db` |
| `mode` | `modo` |
| `speechiness` | `presencia_habla` |
| `acousticness` | `acusticidad` |
| `instrumentalness` | `instrumentalidad` |
| `liveness` | `presencia_en_vivo` |
| `valence` | `positividad` |
| `tempo` | `tempo_bpm` |
| `time_signature` | `compas` |
| `track_genre` | `genero_musical` |

### Diccionario de variables

> Los conteos y rangos siguientes proceden del informe anterior; recalcularlos con el CSV. Los «?» no estaban incluidos en el conteo de nulos explícitos.

| Variable | Tipo | Descripción | Valores únicos | Nulos | % Nulos | Min | Max |
|----------|------|-------------|---------------|-------|---------|-----|-----|
| `id_cancion` | Texto | Identificador único de Spotify | 89,741 | 0 | 0% | — | — |
| `artistas` | Texto | Nombre(s) del/los artista(s) | 31,437 | 1 | 0.001% | — | — |
| `nombre_album` | Texto | Nombre del álbum | — | 1 | 0.001% | — | — |
| `nombre_cancion` | Texto | Nombre de la canción | 73,608 | 1 | 0.001% | — | — |
| `popularidad` | Numérica (int) | **TARGET** — Popularidad (0–100) | 101 | 0 | 0% | 0 | 100 |
| `duracion_ms` | Numérica (int) | Duración en milisegundos | — | 0 | 0% | 8,586 | 5,237,295 |
| `contenido_explicito` | Booleana | Contenido explícito (True/False) | 2 | 0 | 0% | — | — |
| `bailabilidad` | Numérica (float) | Aptitud para bailar (0–1) | — | 0 | 0% | 0.0 | 0.985 |
| `energia` | Numérica (float) | Intensidad percibida (0–1) | — | 0 | 0% | 0.0 | 1.0 |
| `tonalidad` | Categórica (int) | Tonalidad musical (0=C, 11=B) | 12 | 0 | 0% | 0 | 11 |
| `volumen_db` | Numérica (float) | Volumen promedio en dB | — | 0 | 0% | -49.53 | 4.53 |
| `modo` | Categórica (int) | Modalidad (0=menor, 1=mayor) | 2 | 0 | 0% | 0 | 1 |
| `presencia_habla` | Numérica (float) | Presencia de voz hablada (0–1) | — | 0 | 0% | 0.0 | 0.965 |
| `acusticidad` | Numérica (float) | Nivel acústico (0–1) | — | 0 | 0% | 0.0 | 0.996 |
| `instrumentalidad` | Numérica (float) | Nivel instrumental, sin voz (0–1) | — | 0 | 0% | 0.0 | 1.0 |
| `presencia_en_vivo` | Numérica (float) | Presencia de audiencia en vivo (0–1) | — | 0 | 0% | 0.0 | 1.0 |
| `positividad` | Numérica (float) | Positividad musical (0–1) | — | 0 | 0% | 0.0 | 0.995 |
| `tempo_bpm` | Numérica (float) | Velocidad en BPM | — | 0 | 0% | 0.0 | 243.37 |
| `compas` | Categórica (int) | Compás musical (0–5) | 5 | 0 | 0% | 0 | 5 |
| `genero_musical` | Categórica (str) | Género musical | 114 | 0 | 0% | — | — |

### Clasificación analítica

- **Target:** `popularidad`
- **Identificadores (excluir del modelo):** `id_cancion`, `artistas`, `nombre_album`, `nombre_cancion`, `Unnamed: 0`
- **Numéricas continuas predictoras:** `duracion_ms`, `bailabilidad`, `energia`, `volumen_db`, `presencia_habla`, `acusticidad`, `instrumentalidad`, `presencia_en_vivo`, `positividad`, `tempo_bpm`
- **Categóricas predictoras:** `contenido_explicito`, `tonalidad`, `modo`, `compas`, `genero_musical`

> **Nota metodológica:** `tonalidad`, `modo` y `compas` se almacenan como enteros pero son **códigos musicales**, no magnitudes. `tonalidad=7` (Sol) no es "más" que `tonalidad=3` (Re♭). Se tratan como categóricas en el pipeline.

---

## 7. Análisis exploratorio de datos (EDA)

> Referencia histórica: estas tablas y comentarios NO son resultados de la limpieza nueva. Sustituirlos por las salidas actuales del notebook antes de presentarlos.

### 7.1 Estadística descriptiva de variables numéricas

| Variable | Media | Mediana | Desv. Est. | Min | P25 | P75 | P90 | P99 | Max | Skewness | Curtosis |
|----------|-------|---------|-----------|-----|-----|-----|-----|-----|-----|----------|---------|
| `popularidad` | 33.24 | 35.0 | 22.31 | 0 | 17 | 50 | 63 | 80 | 100 | 0.05 | -0.93 |
| `duracion_ms` | 228,031 | 212,906 | 107,297 | 8,586 | 174,066 | 261,506 | 327,773 | 530,909 | 5,237,295 | 11.20 | 354.98 |
| `bailabilidad` | 0.567 | 0.580 | 0.174 | 0.0 | 0.456 | 0.695 | 0.784 | 0.901 | 0.985 | -0.40 | -0.18 |
| `energia` | 0.641 | 0.685 | 0.252 | 0.0 | 0.472 | 0.854 | 0.941 | 0.993 | 1.0 | -0.60 | -0.53 |
| `volumen_db` | -8.26 | -7.00 | 5.03 | -49.53 | -10.01 | -5.00 | -3.68 | -1.63 | 4.53 | -2.01 | 5.90 |
| `presencia_habla` | 0.085 | 0.049 | 0.106 | 0.0 | 0.036 | 0.084 | 0.176 | 0.507 | 0.965 | 4.65 | 28.82 |
| `acusticidad` | 0.315 | 0.169 | 0.333 | 0.0 | 0.017 | 0.597 | 0.867 | 0.992 | 0.996 | 0.73 | -0.95 |
| `instrumentalidad` | 0.156 | 0.0 | 0.310 | 0.0 | 0.0 | 0.049 | 0.833 | 0.955 | 1.0 | 1.73 | 1.27 |
| `presencia_en_vivo` | 0.214 | 0.132 | 0.190 | 0.0 | 0.098 | 0.273 | 0.427 | 0.946 | 1.0 | 2.11 | 4.38 |
| `positividad` | 0.474 | 0.464 | 0.259 | 0.0 | 0.260 | 0.683 | 0.840 | 0.965 | 0.995 | 0.12 | -1.03 |
| `tempo_bpm` | 122.15 | 122.02 | 29.98 | 0.0 | 99.22 | 140.07 | 166.12 | 194.0 | 243.37 | 0.23 | -0.11 |

#### Interpretación de cada variable

**`popularidad` (TARGET):**  
Media 33.24 y mediana 35 están cercanas, indicando distribución aproximadamente simétrica (skewness ≈ 0.05). Sin embargo, hay una fuerte concentración bimodal: el 14.1% de canciones tiene popularidad = 0 (canciones inactivas o sin reproducciones registradas), y solo el 4.8% supera 70. La curtosis negativa (-0.93) confirma una distribución aplanada, sin pico pronunciado. **Implicación para ML:** un modelo de regresión debe convivir con esta bimodalidad y el gran bloque de ceros. Conservar popularidad cero: es un valor de la variable objetivo, no se imputa en esta limpieza.

**`duracion_ms`:**  
La distribución tiene skewness extremo (11.20) y curtosis de 354.98, evidenciando outliers muy severos por la derecha. La canción más larga dura 87.3 minutos (5,237,295 ms), que es probablemente un audio especial (audiolibro, compilado, etc.). La mediana de 3.55 minutos es representativa de lo que un oyente promedio esperaría. **Decisión:** conservar los valores extremos con bandera; no eliminar sin regla de negocio clara.

**`presencia_habla`:**  
Skewness de 4.65 y curtosis de 28.82 indican concentración extrema en valores bajos (mediana 0.049). La mayoría de canciones son música, no spoken word. Valores > 0.66 indican podcasts o rap denso. **Implicación:** esta variable requiere transformación logarítmica para modelos lineales.

**`instrumentalidad`:**  
El 34% de las canciones tiene `instrumentalidad == 0` exacto, y la mediana es 0. Distribución muy sesgada a la derecha. La mayoría de canciones tiene vocales. **Implicación:** es la variable con mayor correlación (negativa) con popularidad (-0.095), lo que es lógico: canciones sin voz son menos populares en streaming generalista.

**`volumen_db`:**  
Distribuida alrededor de -7 dB (mediana). El rango es amplio (-49.53 a +4.53 dB). Los 90 valores positivos son físicamente posibles pero inusuales en música normalizada; pueden corresponder a grabaciones con compresión agresiva. No se eliminan.

**`positividad`:**  
Distribución aproximadamente uniforme (curtosis -1.03), sin pico dominante. Las canciones del dataset cubren todo el espectro de positividad musical. El dataset es representativo emocionalmente.

---

### 7.2 Distribuciones y formas

**Variables con distribución aproximadamente normal:** `bailabilidad`, `tempo_bpm`, `positividad`.

**Variables con sesgo positivo fuerte (cola derecha):** `presencia_habla`, `instrumentalidad`, `presencia_en_vivo`, `duracion_ms`. Para modelos lineales, estas variables se beneficiarían de transformación logarítmica (log1p).

**Variables con sesgo negativo (cola izquierda):** `energia`, `volumen_db`.

**Multimodalidad detectada:** `acusticidad` muestra dos concentraciones: una cerca de 0 (electrónica/rock) y otra cerca de 1 (acústica/clásica). Esto refleja la diversidad de géneros del dataset.

**`popularidad` bimodal:** el pico en 0 (14.1% de canciones) y la distribución principal centrada en 35 crean una bimodalidad que el modelo de regresión debe manejar.

---

### 7.3 Variables categóricas

**`genero_musical`:**
- 114 géneros únicos
- Distribución perfectamente balanceada: **1,000 canciones por género** exactamente
- Esto es artificial: el mercado real no tiene la misma cantidad de canciones en todos los géneros

**Top 5 géneros más populares (por media de popularidad):**
1. `pop-film`: media 59.28, mediana 60
2. `k-pop`: media 56.95, mediana 60
3. `chill`: media 53.65, mediana 57
4. `sad`: media 52.38, mediana 54
5. `grunge`: media 49.59, mediana 55

**Bottom 5 géneros (menor popularidad media):**
1. `iranian`: media 2.21, mediana 0
2. `romance`: media 3.24, mediana 0
3. `latin`: media 8.30, mediana 0
4. `detroit-techno`: media 11.17, mediana 8
5. `chicago-house`: media 12.34, mediana 10

**Interpretación:** Los géneros con mediana 0 (`iranian`, `romance`, `latin`, `jazz`, `classical`) sugieren que la mayoría de canciones de esos géneros no tienen reproducciones registradas en la plataforma, o que el dataset capturó canciones de catálogo antiguo con popularidad desactualizada.

**`contenido_explicito`:**
- False: 104,252 canciones (91.4%)
- True: 9,747 canciones (8.6%)
- Popularidad media: False = 32.94, True = 36.45
- El contenido explícito tiene ligeramente mayor popularidad media, pero la diferencia (3.5 puntos) es pequeña y no implica causalidad. Las canciones explícitas incluyen mucho rap y pop urbano, géneros con alta exposición.

**`modo`:**
- Mayor (1): 72,681 canciones (63.7%)
- Menor (0): 41,318 canciones (36.3%)
- El modo mayor es más frecuente, consistente con la predominancia de música pop/electrónica.

**`compas`:**
- 4/4 (valor 4): 101,842 canciones (89.3%) — dominante en música occidental
- 3/4 (valor 3): 9,195 canciones (8.1%)
- 163 canciones con compas=0 (irregular o no detectado)

---

### 7.4 Correlaciones entre variables

| Par de variables | Correlación | Interpretación |
|-----------------|------------|----------------|
| `energia` ↔ `volumen_db` | **+0.762** | Alta: canciones más enérgicas son más sonoras |
| `energia` ↔ `acusticidad` | **-0.734** | Alta negativa: energía eléctrica vs. sonido acústico |
| `volumen_db` ↔ `acusticidad` | **-0.590** | Moderada negativa |
| `bailabilidad` ↔ `positividad` | **+0.477** | Canciones bailables tienden a ser más alegres |
| `instrumentalidad` ↔ `positividad` | **-0.324** | Instrumental ≠ alegría |
| `volumen_db` ↔ `instrumentalidad` | **-0.433** | |

**Correlaciones con `popularidad` (todas débiles):**

| Variable | Correlación con popularidad |
|----------|--------------------------|
| `instrumentalidad` | **-0.0951** (más fuerte) |
| `volumen_db` | +0.0504 |
| `presencia_habla` | -0.0449 |
| `positividad` | -0.0405 |
| `bailabilidad` | +0.0354 |
| `acusticidad` | -0.0255 |
| `tempo_bpm` | +0.0132 |
| `duracion_ms` | -0.0071 |
| `presencia_en_vivo` | -0.0054 |
| `energia` | +0.0011 |

**Conclusión crítica:** Las correlaciones lineales entre features de audio y popularidad son **todas menores a 0.10 en valor absoluto**. Esto implica que la popularidad no puede explicarse con relaciones lineales simples entre atributos de audio. Los factores más probables que determinan popularidad incluyen: exposición mediática, promoción, artista, sello discográfico, número de playlists que incluyen la canción — ninguno de los cuales está en el dataset.

**Implicación para ML:** se esperan modelos con R² bajo si solo se usan features de audio. El género (`genero_musical`) puede mejorar el modelo dado que las diferencias entre géneros son considerables.

**Alerta de multicolinealidad:** `energia` y `volumen_db` tienen correlación 0.762. En modelos lineales, esta redundancia puede inflar varianza de coeficientes. En modelos de árbol no es un problema, pero reduce la interpretabilidad de importancia de features.

---

### 7.5 Análisis de relaciones categóricas

**Popularidad por contenido_explicito:**

Las canciones explícitas tienen popularidad media ligeramente superior (36.45 vs. 32.94). Esta diferencia no es causal: el contenido explícito se asocia a géneros (rap, reggaeton, trap) que reciben alta exposición en plataformas de streaming.

**Popularidad por género (Top 10 vs Bottom 10):**

El rango entre el género más popular (`pop-film`, media 59.28) y el menos popular (`iranian`, media 2.21) es de 57 puntos. Esta diferencia entre grupos es **mucho más grande** que cualquier correlación individual con features de audio, lo que sugiere que `genero_musical` será la variable predictora más potente en el modelo.

---

## 8. Calidad de datos

### 8.1 Resumen de calidad

La revisión distingue nulos explícitos, símbolos «?», cadenas vacías y ceros
sujetos a una regla de limpieza. `auditoria_faltantes.csv` reporta cada tipo antes
de eliminar filas. `balance_limpieza.csv` informa el efecto efectivo de las reglas.

### 8.2 Valores faltantes e incidencias reportadas

| Variables | Incidencia informada | Acción implementada |
|---|---|---|
| `artistas`, `nombre_cancion` | 1 nulo explícito en cada variable | Eliminar la fila afectada |
| `nombre_album` | 1 nulo explícito y 20 «?» | `DESCONOCIDO` en filas conservadas |
| `tempo_bpm`, `bailabilidad`, `energia` | 157 ceros reportados en este grupo; verificar por variable | Eliminar si al menos una vale cero |
| `compas` | 163 ceros | Moda de entrenamiento en las filas conservadas |
| `duracion_ms` | 1 cero | Eliminar la fila |

Los conteos son información aportada para esta corrección, no mediciones nuevas.
El número de imputaciones puede ser menor que el de incidencias iniciales porque
algunas filas ya se eliminan por otros motivos. No sumar celdas o conteos por
variable como si fueran filas distintas.

---

### 8.3 Duplicados conceptuales (id_cancion repetidos)

El dataset tiene 114,000 filas pero solo **89,741 id_cancion únicos**. La diferencia (24.259) cuenta filas adicionales, no el número de canciones repetidas; este último se recalcula con las frecuencias por ID.

**Ejemplo:** una canción de pop que también aparece en la categoría "dance-pop" y "electropop" tendrá tres filas con el mismo `id_cancion` pero distinto `genero_musical`.

**Esto no es un error:** es la estructura intencional del dataset (114 géneros × 1,000 canciones cada uno).

**Riesgo de ML:** si la misma canción aparece en train y en test, el modelo "recuerda" esa canción, inflando artificialmente las métricas. **Solución:** usar `GroupShuffleSplit` agrupando por `id_cancion`, garantizando que ningún `id_cancion` aparezca simultáneamente en train y test.

### 8.4 Outliers detectados por IQR

> Conteos del informe previo; el notebook vuelve a calcular IQR después de la nueva limpieza.

| Variable | Límite Inf. | Límite Sup. | Outliers IQR | % | Decisión |
|----------|------------|------------|-------------|---|----------|
| `instrumentalidad` | -0.074 | 0.123 | 25,246 | 22.1% | **Conservar** — distribución bimodal válida |
| `presencia_habla` | -0.037 | 0.157 | 13,211 | 11.6% | **Conservar** — rap/spoken word son valores válidos |
| `presencia_en_vivo` | -0.164 | 0.536 | 8,642 | 7.6% | **Conservar** — grabaciones en vivo son reales |
| `volumen_db` | -17.53 | +2.51 | 6,173 | 5.4% | **Conservar** — rango de dB es válido musicalmente |
| `duracion_ms` | 42,906 | 392,666 | 5,616 | 4.9% | **Conservar con bandera** — revisar extremo superior |
| `popularidad` | -32.5 | +99.5 | 2 | 0.002% | **Conservar** — son valores 100, el máximo posible |
| `tempo_bpm` | 37.94 | 201.35 | 617 | 0.5% | **Conservar** — BPM extremos son válidos (drum&bass, lento) |
| `bailabilidad` | 0.098 | 1.053 | 620 | 0.5% | **Conservar** — ninguno fuera del rango 0–1 |

**Criterio general:** el criterio IQR es sensible a distribuciones asimétricas y bimodales. En este dataset, la mayoría de "outliers" detectados son valores legítimos que reflejan la diversidad musical. **No se elimina ningún outlier de forma automática.**

**Excepción identificada:** la canción de 87.3 minutos merece revisión manual. Una canción de esa duración puede ser un compilado, meditación guiada o audiolibro. No se elimina, pero se registra como caso extremo.

---

## 9. Tratamiento de anomalías y valores faltantes

### 9.1 Reglas aplicadas

| Acción | Condición | Justificación |
|---|---|---|
| Renombrar al cargar | 20 columnas originales | Lectura consistente en español |
| Eliminar índice artificial | `Unnamed: 0`, si existe | No aporta información |
| Eliminar fila | `artistas` o `nombre_cancion` nulos, vacíos o «?» | No inventar metadatos |
| Imputar `DESCONOCIDO` | `nombre_album` nulo, vacío o «?» | Preservar audio útil sin afirmar que sea un single |
| Eliminar fila | Cero en cualquiera de `tempo_bpm`, `bailabilidad`, `energia` | Criterio solicitado para este proyecto |
| Eliminar fila | Nulo explícito en esas tres métricas | No se usa KNN en esta versión |
| Eliminar fila | `duracion_ms` cero, negativa o nula | Duración no utilizable |
| Imputar moda | `compas` cero o nulo | Categoría observada, sin fracciones |
| Crear `duracion_min` | `duracion_ms / 60000` | Lectura de duración en minutos |
| Reportar IQR | Valores extremos restantes | No eliminarlos automáticamente |

**Alcance de los ceros:** bailabilidad y energía incluyen cero en sus escalas.
Su eliminación es una decisión del proyecto, no una prueba de dato faltante.
Se conservan ceros de otras variables, por ejemplo popularidad, instrumentalidad,
tonalidad y modo. [Definiciones de audio de Spotify](https://developer.spotify.com/documentation/web-api/reference/get-audio-features).

**Compás:** se reserva la partición por canción antes de imputar. La moda se
calcula con valores positivos enteros observados solo en entrenamiento; no se
fija arbitrariamente en 4. Ante empate se elige el menor valor. Si no hay compases
observados en entrenamiento, se detiene la ejecución con un mensaje claro.
[Regla de moda de SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).

`KNNImputer` promedia valores vecinos y podría producir fracciones; no se usa
directamente para esta categoría. Esto no significa que sea imposible diseñar
otro método basado en vecinos que preserve categorías.

### 9.2 Auditoría y resultado

Las máscaras de eliminación se combinan con OR: una fila con varios problemas se
elimina una sola vez. Después se imputan álbum y compás únicamente en las filas
conservadas. Se exportan el diagnóstico inicial, los motivos de descarte, las
filas descartadas, el balance final y la asignación entrenamiento/prueba.

El total final no está fijado en 113.999 ni se calcula restando 157 + 163 + 1:
depende de las coincidencias reales y debe obtenerse al ejecutar el notebook.

---

## 10. Preparación y transformación de datos

### 10.1 Bases de trabajo

- `df_crudo`: CSV con nombres españoles, sin eliminar filas.
- `df_base`: filas conservadas, álbum imputado con constante y compás cero marcado
  como NaN; origen del pipeline y de `spotify_base_modelo.csv`.
- `df_limpio`: compás completado con la moda de entrenamiento y duración en minutos;
  origen de tablas, gráficos y `spotify_clean.csv`.

`popularidad` es el objetivo. Se excluyen `id_cancion`, `artistas`, `nombre_album`,
`nombre_cancion` y la duración en minutos duplicada. Las diez variables de audio
son numéricas; contenido explícito, tonalidad, modo, compás y género se codifican
como categorías. El compás mantiene un valor entero antes de la codificación.

### 10.2 Partición y ajuste

La función `limpiar_spotify` reserva una sola partición con `GroupShuffleSplit`,
`random_state=42` y `test_size=0.20`, agrupando por `id_cancion`. El 20% corresponde
a grupos: las proporciones exactas de filas se calculan, no se presuponen.

La sección 15 reutiliza esos índices. El pipeline recibe `df_base` y contiene:

| Grupo | Transformación |
|---|---|
| Audio numérico | `StandardScaler` |
| Compás | `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder` |
| Otras categorías | `OneHotEncoder(handle_unknown='ignore')` |

Se ejecuta `fit_transform` exclusivamente en entrenamiento y `transform` en prueba.
Se comprueba que no haya IDs compartidos y que las matrices no tengan nulos ni
infinitos. La moda del pipeline debe coincidir con la usada para el EDA.

### 10.3 Escalamiento

`StandardScaler` centra y escala usando media y desviación estándar de
entrenamiento. **Es sensible a outliers**; no debe presentarse como un método
robusto. Su elección se puede comparar con otras alternativas en EP2.

### 10.4 Uso en EP2

Para validación cruzada, partir de `df_base` / `spotify_base_modelo.csv` y colocar
el preprocesador y el estimador dentro de un pipeline ajustado en cada pliegue
con grupos. No reutilizar la imputación ya aprendida de `spotify_clean.csv` para
una nueva partición. En nuevos datos, aplicar las mismas reglas deterministas
(incluido compás cero → NaN) antes de transformar con el pipeline entrenado.

El EDA de todo el conjunto es descriptivo. La selección de variables, modelos e
hiperparámetros se realiza con entrenamiento, sin usar el conjunto de prueba.

---

## 11. Variables relevantes para Machine Learning

### 11.1 Para regresión (predicción de `popularidad` como número)

| Variable | Relevancia estimada | Razonamiento |
|----------|-------------------|--------------|
| `genero_musical` | 🔴 Alta | Diferencia de 57 puntos entre géneros más/menos populares |
| `instrumentalidad` | 🟡 Baja-media | Correlación -0.095, la más fuerte de audio |
| `volumen_db` | 🟡 Baja | Correlación +0.050 |
| `presencia_habla` | 🟡 Baja | Correlación -0.045 |
| `contenido_explicito` | 🟡 Baja | Diferencia de 3.5 puntos entre grupos |
| `bailabilidad` | 🟡 Baja | Correlación +0.035 |
| `duracion_ms` | 🟢 Muy baja | Correlación -0.007 |
| `energia` | 🟢 Muy baja | Correlación +0.001 |

### 11.2 Para clasificación (canción popular/no popular)

Si se definiera una variable binaria (ej: `popular = popularidad >= 50`), el balance de clases sería:
- No popular (< 50): 74,632 canciones (65.5%)
- Popular (≥ 50): 39,367 canciones (34.5%)

El desbalance es moderado (ratio ~2:1), manejable con `class_weight='balanced'` o SMOTE.

**Justificación para clasificación binaria:** si el negocio necesita solo una decisión (¿priorizar o no priorizar esta canción?), la clasificación binaria es más directamente accionable que la regresión. Sin embargo, se pierde granularidad (una canción con 51 y una con 95 se tratarían igual).

### 11.3 Variables a excluir del modelo

| Variable | Razón de exclusión |
|----------|-------------------|
| `id_cancion` | Identificador; incluirlo provoca memorización |
| `artistas` | Alta cardinalidad (31,437 valores); artistas en test pueden no estar en train |
| `nombre_album` | Alta cardinalidad; mismo problema |
| `nombre_cancion` | Alta cardinalidad; mismo problema |
| `Unnamed: 0` | Índice artificial |
| `duracion_min` | Derivada de `duracion_ms`; redundante |

---

## 12. Sesgos, ética y privacidad

### 12.1 Sesgos identificados

#### Sesgo de muestreo balanceado artificial

El dataset tiene exactamente 1,000 canciones por género (114 géneros). Esto no representa la distribución real del mercado musical: el mercado tiene órdenes de magnitud más canciones de pop que de grindcore. Un modelo entrenado en este dataset puede subestimar la popularidad real en géneros masivos y sobreestimar en nichos.

**Tipo:** sesgo observado y medible.

#### Sesgo de retroalimentación (feedback loop)

`popularidad` en Spotify refleja reproducciones recientes. Si el modelo predice alta popularidad → se incluye en más playlists algorítmicas → más reproducciones → mayor popularidad real. Esto crea un ciclo que favorece a canciones ya populares.

**Tipo:** sesgo potencial en el uso del modelo; debe investigarse en etapa de deployment.

#### Sesgo de exposición por género

Los géneros con mediana de popularidad = 0 (`iranian`, `romance`, `latin`, `jazz`) no necesariamente son de menor calidad: pueden ser géneros con menor exposición en el algoritmo de Spotify, que mayoritariamente opera en inglés y favorece el pop anglosajón.

**Tipo:** sesgo potencial. Requiere investigación adicional comparando con métricas externas al dataset.

#### Sesgo por artista

El dataset incluye 31,437 artistas. Artistas con mayor cantidad de álbumes tienen más canciones y más probabilidad de que alguna sea muy popular, sesgando el análisis de popularidad promedio por artista.

**Tipo:** sesgo observado.

#### Sesgo de temporalidad

`popularidad` es un indicador dinámico. Una canción que era popular en 2020 puede tener hoy popularidad baja. Sin fecha de extracción del CSV, no es posible saber qué tan actualizados están los valores.

**Tipo:** limitación metodológica que puede inducir sesgo.

### 12.2 Estándares de privacidad

**Datos de usuarios:** el CSV no contiene información de usuarios (historiales de reproducción, perfiles, datos personales). El riesgo de privacidad individual es bajo.

**Datos de artistas:** los nombres de artistas son información pública. No hay información sensible de personas naturales en el dataset.

**Si se incorporaran datos de usuarios en el futuro:**
- Aplicar minimización de datos (solo lo necesario)
- Pseudonimización de identificadores de usuario
- Obtener consentimiento explícito o verificar base legal (GDPR/Ley 19.628 Chile)
- Implementar controles de acceso y cifrado
- Establecer plazos de retención y eliminación

### 12.3 Consideraciones éticas

**¿Qué implica predecir popularidad?**

Si el modelo se usa para decidir qué canciones promueve Spotify, puede crear una profecía autocumplida: las canciones predichas como populares reciben más promoción → se vuelven más populares → validan el modelo. Esto:
- Favorece a artistas ya establecidos
- Dificulta el descubrimiento de artistas emergentes de géneros no anglosajones
- Puede homogeneizar la oferta musical

**Recomendaciones éticas:**
- No usar el modelo como único criterio de decisión editorial
- Evaluar el desempeño del modelo por género (¿funciona igual para pop que para jazz?)
- Mantener supervisión humana en decisiones que afectan a artistas
- Evitar comunicar la predicción como medida de calidad artística
- Monitorear el impacto del modelo en la diversidad del catálogo

---

## 13. Hallazgos clave

La nueva versión hace verificable el tratamiento de faltantes y mantiene los
nombres en español a lo largo de la preparación y los gráficos. El balance
registra filas únicas descartadas y cantidades realmente imputadas.

Los valores históricos de correlación, popularidad por género y frecuencias deben
recalcularse. No se concluye que una variable sea la mejor predictora ni que el R²
será bajo sin evaluar un modelo. El análisis bivariado es evidencia descriptiva.

## 14. Conclusiones

La implementación queda preparada para procesar el CSV original y generar salidas
actualizadas. No se entrenó un modelo ni se certificaron resultados reales en esta
revisión porque el CSV no fue adjuntado.

Se debe revisar el sesgo de eliminar canciones con bailabilidad o energía cero y
medir la pérdida de representación por género. La imputación del compás conserva
categorías enteras y utiliza únicamente entrenamiento. Los identificadores
repetidos se agrupan en la partición, pero no se eliminan como duplicados.

Antes de presentar, regenerar tablas y gráficos y reemplazar las cifras históricas
de este informe. Para EP2, comparar contra una predicción constante basada en la
media de entrenamiento y evaluar MAE, RMSE, R² y resultados por género.

---

## 15. Trabajo futuro

| Fase | Acción |
|------|--------|
| **Modelamiento** | Entrenar Ridge Regression, Random Forest Regressor y Gradient Boosting contra baseline de media constante |
| **Evaluación** | Reportar MAE, RMSE y R² globales y por género |
| **Clustering** | Aplicar K-Means con k=3–8 sobre features de audio normalizados; evaluar con silueta |
| **Features avanzados** | Incorporar artista como embedding o target encoding |
| **Actualización de datos** | Extraer datos actualizados de la API de Spotify con fecha registrada |
| **Transformaciones** | Explorar log1p para presencia_habla, instrumentalidad y presencia_en_vivo en modelos lineales |
| **Bias auditing** | Evaluar desempeño del modelo por género y reportar diferencias |
| **Deployment** | Construir función de scoring reproducible; monitorear drift mensual |

---

## 16. Análisis exploratorio avanzado

> Referencia del análisis anterior. Recalcular todos los números y revisar sus interpretaciones; no son resultados de esta actualización.

> Esta sección va más allá de los requisitos de la EP1. Explora el dataset como lo haría un Data Scientist contratado para descubrir patrones no evidentes.

### 16.1 Anatomía de las canciones más populares (popularidad ≥ 70)

Del total de 113,999 canciones, solo **5,472 (4.8%)** tienen popularidad ≥ 70. Estas canciones forman una categoría de élite. Sus características promedio vs. el resto:

| Feature | Canciones populares (≥70) | Resto |
|---------|--------------------------|-------|
| `bailabilidad` | 0.632 (alta) | 0.561 |
| `energia` | 0.651 | 0.640 |
| `volumen_db` | -6.48 | -8.35 (más sonoras) |
| `presencia_habla` | 0.091 | 0.084 |
| `acusticidad` | 0.251 | 0.320 (menos acústicas) |
| `instrumentalidad` | 0.030 | 0.162 (casi todas tienen voz) |
| `positividad` | 0.419 | 0.477 (menos "alegres") |
| `tempo_bpm` | 124.3 | 122.1 |

**Patrón de canciones populares:** más bailables, más sonoras, casi sin instrumentalidad, menos acústicas, y sorprendentemente con menor positividad (menos "alegres" que el promedio). Esto sugiere que la popularidad en streaming no se asocia a "felicidad musical" sino a energía y presencia vocal.

### 16.2 Paradoja del positividad

Contra la intuición común, las canciones más populares tienen **menor positividad** (positividad musical) que el promedio. Esto puede explicarse porque los géneros más populares del dataset (k-pop, pop-film, sad, emo) incluyen géneros "melancólicos" con alta base de fans comprometidos.

### 16.3 Bimodalidad de acusticidad

`acusticidad` tiene dos concentraciones claras: cerca de 0 (electrónica/rock) y cerca de 0.9 (acústica/clásica). Esta bimodalidad refleja dos mundos musicales distintos en el mismo dataset. Los modelos de ML deberían manejar esto, pero un modelo global puede mezclarse con patrones contradictorios.

### 16.4 Tratamiento de tempo cero en esta versión

Las filas con `tempo_bpm == 0` se eliminan según la regla del proyecto, junto con
las que tengan bailabilidad o energía cero. El diagnóstico conserva sus conteos
originales y las filas descartadas para revisar su efecto. El valor cero por sí
solo no permite concluir por qué falló una medición ni cuál es el género.

---

### 16.5 Variables redundantes detectadas

**Par 1:** `energia` y `volumen_db` (r=0.762) — Prácticamente miden lo mismo en canciones modernas con normalización de volumen. Para modelos lineales, usar ambas es redundante.

**Par 2:** `bailabilidad` y `positividad` (r=0.477) — Correlación moderada: canciones bailables tienden a ser más alegres. No son intercambiables, pero aportan información parcialmente solapada.

**Recomendación:** en modelos con regularización (Ridge, Lasso), esto no es un problema crítico. En análisis de importancia de features, interpretar con cuidado.

### 16.6 Géneros con distribución de popularidad bimodal

Géneros como `jazz`, `classical`, `latin` y `romance` tienen **mediana = 0** pero **media > 0**. Esto indica que la mayoría de canciones de esos géneros tiene popularidad = 0, pero un pequeño número tiene popularidad alta. Son géneros de élite con una distribución muy desigual: pocas canciones muy famosas y miles oscuras.

### 16.7 Hipótesis de negocio generadas

1. **H1:** Las canciones con `instrumentalidad < 0.01` (voz presente) tienen popularidad sistemáticamente mayor. *(Verificable con t-test).*
2. **H2:** La combinación de alto `bailabilidad` + alta `energia` + baja `acusticidad` predice el cuartil superior de popularidad mejor que cualquier feature individual.
3. **H3:** El género explica más varianza en popularidad que todos los features de audio combinados.
4. **H4:** Comparar popularidad y género de las filas descartadas por ceros usando `df_crudo`; no se puede evaluar esta hipótesis en `df_limpio`, donde ya fueron retiradas.
5. **H5:** La duración óptima para maximizar popularidad está entre 2.5 y 4 minutos.

---

## 17. Auditoría contra la rúbrica

| Indicador | Peso | Evidencia | Estado | Observaciones |
|-----------|-----:|-----------|--------|---------------|
| **IE1:** Identificación de fuentes de datos y herramientas colaborativas | 10% | Sección 4: fuente primaria documentada (CSV, origen Kaggle), herramientas (Python, pandas, Jupyter, Git/GitHub, Colab), limitaciones explicitadas (sin fecha de extracción, sin versión de API) | Implementado; contrastar con rúbrica y ejecución | Justificación completa de cada herramienta |
| **IE2:** Manipulación y preparación de datos en Python | 30% | Sección 10: renombrado, auditoría de nulos y ceros, descarte por reglas, imputación de álbum y compás, partición por canción y pipeline sin ajuste sobre prueba | Implementado; contrastar con rúbrica y ejecución | Pipeline reproducible, organizado y sin leakage |
| **IE3:** Análisis exploratorio y calidad de los datos | 40% | Secciones 7–9: estadística descriptiva completa con skewness y curtosis, histogramas, frecuencias de categóricas, outliers IQR con interpretación, correlaciones, comparaciones por género y contenido_explicito, calidad de datos tabulada, decisiones justificadas | Implementado; contrastar con rúbrica y ejecución | EDA completo con interpretación de cada hallazgo |
| **IE4:** Evaluación de sesgos, ética y privacidad | 20% | Sección 12: 5 sesgos identificados (muestreo, feedback loop, exposición, artista, temporalidad), privacidad analizada, medidas de mitigación concretas, diferenciación entre sesgo observado y potencial | Implementado; contrastar con rúbrica y ejecución | Análisis profundo y diferenciado |

**¿El proyecto alcanzaría "Muy buen desempeño"?** Sí, si el notebook se ejecuta correctamente y se presentan los resultados con claridad en la defensa. Los 4 indicadores están cubiertos en profundidad y con justificación técnica.

---

## 18. Preguntas de defensa oral

> Las respuestas con cifras del análisis previo deben actualizarse con la nueva ejecución.

### IE1 — Fuentes y herramientas

**P1. ¿Por qué usaron Jupyter Notebook y no un script Python normal?**  
Jupyter integra código, visualizaciones y texto explicativo en un solo documento, lo que facilita la reproducibilidad y la comunicación de resultados. Para EDA es preferible porque permite ver resultados célula a célula.

**P2. ¿Qué limitaciones tiene la fuente de datos?**  
No tiene fecha de extracción ni versión de API de Spotify. Esto impide saber qué tan actualizados son los valores de `popularidad`, que varía con el tiempo.

**P3. ¿Para qué usarían GitHub en este proyecto?**  
Control de versiones del código y del notebook, revisión de cambios entre colaboradores, y trazabilidad del trabajo grupal. Permite que cualquier persona pueda reproducir el análisis clonando el repositorio.

### IE2 — Manipulación de datos

**P4. ¿Por qué eliminaron la columna `Unnamed: 0`?**  
Es un índice artificial generado al exportar el CSV con `df.to_csv()` sin `index=False`. No aporta información analítica y puede confundirse con un feature si se deja.

**P5. ¿Qué es data leakage y cómo lo previenen?**  
Data leakage ocurre cuando información del conjunto de test "contamina" el entrenamiento, inflando artificialmente las métricas. En este dataset, una misma canción puede aparecer en múltiples géneros; si cae en train y en test, el modelo "la recuerda". Se previene con `GroupShuffleSplit` agrupando por `id_cancion`.

**P6. ¿Por qué usaron StandardScaler y no MinMaxScaler?**  
StandardScaler centra y escala usando entrenamiento. Es sensible a valores extremos; no es correcto afirmar que sea robusto. La elección del escalador se evaluará junto con el modelo en EP2.

**P7. ¿Cuándo se ajusta el StandardScaler, con train o con todo el dataset?**  
Solo con los datos de train. Si se ajusta con todo el dataset incluyendo test, se está usando información del test para preparar los datos de entrenamiento, lo cual es leakage.

**P8. ¿Por qué no usaron Label Encoding para `genero_musical`?**  
Label Encoding asigna valores numéricos ordinales (0, 1, 2...) a los géneros, implicando que hay un orden entre ellos (género 0 < género 1). Los géneros no tienen orden natural. OneHotEncoder crea una columna binaria por género, sin imponer orden.

### IE3 — EDA y calidad

**P9. ¿Qué encontraron en la distribución de `popularidad`?**  
Distribución aproximadamente simétrica (skewness ≈ 0.05) pero con bimodalidad: el 14.1% de canciones tiene popularidad = 0 (posiblemente canciones inactivas o catálogo sin reproducciones), y el grueso se distribuye entre 10 y 70. Solo el 4.8% supera 70.

**P10. ¿Cuál es la correlación más alta entre los features de audio y `popularidad`?**  
`instrumentalidad` con -0.095. Esto indica que canciones con más instrumentalidad (sin voz) tienden a ser menos populares. Pero es una correlación débil: solo explica menos del 1% de la varianza de popularidad.

**P11. ¿Por qué no eliminaron los outliers de `instrumentalidad`?**  
El IQR detectó 22.1% de outliers en esta variable, pero `instrumentalidad` tiene una distribución bimodal legítima: muchas canciones tienen exactamente 0 (todas tienen voz) y otras tienen valores altos (son puramente instrumentales). Eliminar los "outliers" significaría eliminar todas las canciones instrumentales, lo cual no tiene sentido.

**P12. ¿Qué significa que `energia` y `volumen_db` tengan correlación 0.762?**  
Que son altamente redundantes: canciones más enérgicas son sistemáticamente más sonoras. En modelos lineales, incluir ambas puede inflar los errores estándar de los coeficientes (multicolinealidad). En modelos de árbol no es un problema, pero la importancia de features puede distribuirse arbitrariamente entre ellas.

**P13. ¿Por qué el género con menor popularidad tiene mediana = 0?**  
Géneros como `iranian`, `romance` y `latin` tienen mediana = 0, lo que significa que más del 50% de sus canciones tienen popularidad = 0. Esto puede indicar: canciones de catálogo antiguo con poca actividad reciente, géneros poco representados en el algoritmo de recomendación de Spotify, o simplemente canciones que en la fecha de extracción no tenían reproducciones significativas.

**P14. ¿Qué es skewness y por qué importa para ML?**  
Skewness mide la asimetría de una distribución. Valores positivos indican cola derecha larga. En ML, distribuciones muy asimétricas (presencia_habla con skewness 4.65) pueden perjudicar a modelos lineales que asumen distribuciones más simétricas. Se recomienda aplicar transformación log1p para linearizar estas variables.

### IE4 — Sesgos y ética

**P15. ¿Qué es un feedback loop y por qué es un riesgo ético?**  
Un feedback loop ocurre cuando el resultado del modelo influye en los datos futuros. Si el modelo predice alta popularidad para una canción → Spotify la promociona más → obtiene más reproducciones → su popularidad real sube → "valida" el modelo. Esto favorece a canciones y artistas ya conocidos, dificultando el descubrimiento de nuevos talentos.

**P16. ¿El dataset tiene problemas de privacidad?**  
En su forma actual, no. No contiene datos de usuarios ni información personal identificable. Los nombres de artistas son información pública. El riesgo de privacidad sería si se incorporaran historiales de reproducción individuales.

**P17. ¿Qué sesgo tiene el muestreo del dataset?**  
El dataset tiene exactamente 1,000 canciones por género (114 géneros), lo que es un muestreo artificialmente balanceado. El mercado real no es balanceado: hay miles de millones de canciones de pop y miles de black metal. Un modelo entrenado aquí puede generalizar incorrectamente a distribuciones reales.

**P18. ¿Cómo distinguen entre sesgo observado y sesgo potencial?**  
Sesgo observado es el que se puede medir con los datos disponibles (ej: género más popular en el dataset es pop-film). Sesgo potencial es el que sospechamos pero no podemos demostrar con estos datos (ej: que el algoritmo de Spotify favorece artistas anglosajones). Es importante distinguirlos para no afirmar como hecho lo que es hipótesis.

**P19. ¿Por qué es importante evaluar el modelo por género?**  
Si el modelo tiene buen desempeño global pero falla en géneros específicos (ej: predice bien para pop pero no para música latinoamericana), su uso puede perjudicar sistemáticamente a artistas de esos géneros. Reportar métricas por subgrupo es una práctica de equidad algorítmica.

**P20. ¿Qué harían diferente si tuvieran que mejorar el dataset?**  
Registrar la fecha de extracción y versión de API de Spotify; obtener una muestra proporcional al mercado real en lugar del muestreo balanceado artificial; incorporar variables externas como número de playlists que incluyen la canción, fecha de lanzamiento y presencia en redes sociales del artista; y actualizar los valores de `popularidad` periódicamente para tener una señal temporal.

---

## Estructura del proyecto

| Ruta | Contenido |
|---|---|
| `data/raw/Spotify_Tracks_Dataset.csv` | Datos originales, necesarios para ejecutar |
| `EP1_Spotify_ML_EDA_Preparacion_COMPLETO.ipynb` | Notebook actualizado; también admite ubicación en `notebooks/` |
| `README_Spotify_ML.md` | Informe y reglas actualizadas |
| `requirements.txt` | Dependencias |
| `data/processed/spotify_clean.csv` | Datos para EDA con compás imputado |
| `data/processed/spotify_base_modelo.csv` | Base sin imputación estadística del compás, para ML |
| `data/processed/auditoria_faltantes.csv` | Diagnóstico inicial por variable |
| `data/processed/resumen_eliminacion.csv` | Conteos por criterio, pueden coincidir |
| `data/processed/balance_limpieza.csv` | Filas únicas eliminadas e imputaciones |
| `data/processed/filas_descartadas.csv` | Registros originales descartados y motivos |
| `data/processed/particion_modelo.csv` | Asignación por canción a entrenamiento/prueba |
| `data/processed/descriptive_statistics.csv` | Estadísticas recalculadas |
| `data/processed/outliers_iqr_summary.csv` | Resumen IQR recalculado |
| `data/processed/genre_popularity_stats.csv` | Popularidad por género recalculada |
| `data/processed/variables_modelo.csv` | Nombres después de codificar |
| `images/01_histogramas_numericas.png` | Histogramas con nombres españoles |
| `images/02_variables_categoricas.png` | Frecuencias por categoría |
| `images/03_matriz_correlacion.png` | Correlaciones |
| `images/04_popularidad_por_genero.png` | Comparación de géneros |
| `images/05_popularidad_explicit.png` | Popularidad y contenido explícito |

Las salidas de datos y gráficos se generan al ejecutar el notebook; no se incluyen
salidas simuladas en esta entrega.

---

*Proyecto desarrollado para la asignatura Machine Learning (MLY1101), Duoc UC, 2026.*


### Preguntas sobre la nueva limpieza

**¿Por qué no usar SINGLE para un álbum faltante?** Porque el campo vacío no
demuestra el tipo de lanzamiento. DESCONOCIDO expresa lo que realmente sabemos.

**¿Por qué no sumar los conteos de incidencias?** Una fila puede tener artista
nulo, duración cero y varias métricas cero simultáneamente. La unión de máscaras
la cuenta una sola vez.

**¿Por qué la moda del compás se calcula en entrenamiento?** Para que la
distribución del conjunto de prueba no influya en la preparación del modelo.

**¿Todo cero es un dato faltante?** No. La eliminación está limitada a las
variables indicadas por el proyecto; se conservan, por ejemplo, popularidad cero
y modo menor (0).
