# Proyecto final Machine Learning
<p align="justify">
Este es el proyecto final de nuestro bootcamp de Machine Learning, donde demostramos las habilidades y conocimientos adquiridos a lo largo de nuestros estudios. A lo largo de este bootcamp, hemos estudiado diferentes modelos basados en proyectos de diferentes áreas y tipos. Ahora es el momento de crear nuestro propio proyecto utilizando el algoritmo que creemos que se adapta mejor a nuestro problema.

Tendremos que encontrar un conjunto de datos adecuado para trabajar, procesarlo, entrenar un modelo y, finalmente, ponerlo a disposición para su consumo.
</p>
 *"Hard work always beats talent when talent doesn't work hard"* - Tim Notke

## 👥  Credits

**Team Members:**
> - Ineta Keryte
> - Anthonny Maldonado
> - Guillermo Mansanta

**Academy:** 
> - [4Geeks Academy](https://4geeksacademy.com/us/index) 
> - **Bootcamp:** Spain-DS-17 
> - **Mentor:** [Ing. Héctor Chocobar Torrejón](https://github.com/hchocobar/)
> - **Teacher Assitant:** [Beatriz Solana Ros](https://github.com/mezcolantriz)

## 🎯 Objetivo del proyecto
<p align="justify">
- El objetivo de este proyecto es diseñar y entrenar un modelo de Machine Learning aplicado a la gestion farmaceutica, capaz de proyectar la demanda futura de cada producto de la farmacia para el año 2026, utilizando como base el histórico de ventas del año 2025. El modelo busca incorporar variables claves como la estacionalidad, el producto, el rubro, la presentación del producto y los patrones de consumo de los clientes, para estimar con mayor precisión cuántas unidades será necesario disponer en stock en cada período.

De esta manera, se apunta a transformar la gestión de inventario en un proceso proactivo y basado en datos, que permita optimizar los niveles de stock, reducir pérdidas por vencimientos, evitar quiebres de productos esenciales y mejorar la rentabilidad general del comercio.
</p>

El objetivo de este proyecto es desarrollar una solución completa de Machine Learning de extremo a extremo que incluya:
- Adquisición y procesamiento de datos
- Análisis exploratorio de datos (EDA)
- Desarrollo y optimización de modelos
- Desarrollo de aplicaciones Web
- Resolución de problemas del mundo real a través de técnicas de ML

## 🚀 Introducción al proyecto
<p align="justify">
- Este proyecto tiene como objetivo aplicar técnicas de Machine Learning para mejorar la forma en que una farmacia gestiona su stock. A partir del análisis de las ventas de un año completo (2025) de una farmacia ubicada en la provincia de Buenos Aires, se busca entender cómo se comporta la demanda de los productos y usar esa información para planificar mejor el inventario del año 2026.

  La idea principal es pasar de una gestión basada solo en la experiencia a una gestión basada en datos, que permita anticiparse a las necesidades de los clientes, evitar faltantes de productos importantes y reducir el exceso de mercadería en un contexto económico cambiante. Todo el enfoque está pensado desde la realidad del negocio farmacéutico y el comportamiento de consumo de las personas.*
</p>

### Nuestro problema
<p align="justify">
- En el contexto macroeconómico argentino, atravesado por inflación, inestabilidad en los precios y restricciones en el acceso al financiamiento, la gestión de inventarios se convierte en un factor crítico para la sostenibilidad de cualquier farmacia. La falta de una planificación de stock basada en criterios técnicos y analíticos impacta directamente tanto en la rentabilidad del negocio como en la calidad del servicio prestado a la comunidad.

Desde la perspectiva comercial, una mala política de inventarios genera una utilización ineficiente del capital de trabajo, con recursos financieros inmovilizados en mercadería de baja rotación o con riesgo de vencimiento. Esto incrementa los costos operativos, deteriora el flujo de caja y limita la capacidad de negociación con droguerías y laboratorios, afectando condiciones de pago, descuentos y líneas de crédito.

Desde la perspectiva del servicio farmacéutico, los errores de planificación derivan en quiebres de stock de medicamentos esenciales, demoras en la atención, pérdida de continuidad en tratamientos y disminución de la confianza de los pacientes y clientes. La farmacia deja de ser percibida como un punto de referencia sanitario confiable y pasa a ser vista como un comercio reactivo e ineficiente.

En conjunto, la ausencia de una gestión profesional del inventario compromete simultáneamente la competitividad económica del negocio y su rol social como prestador de un servicio de salud.

👉 Problema real: la farmacia no cuenta con una metodología objetiva para anticipar la demanda futura de sus productos.

🎯 Objetivo: desarrollar un modelo predictivo que permita estimar el stock óptimo por producto para el año 2026, en función del comportamiento histórico de ventas, estacionalidad, tipo de producto y patrones de consumo.
</p>



## ➖ DATASET
<p align="justify">
- El dataset contiene información de registro de ventas durante el año 2025 de una farmacia situada en Argentina, en la provincia de Buenos Aires. El registro se corresponde a los datos de los tickets de venta generados durante todo el año, los días que el comercio estuvo abierto. Teniendo en cuenta que el comercio trabaja de Lunes a Sábados de 8hs a 20 hs, es decir, 12 hs por día, se han generado un total de datos tal que nuestro dataset contiene:
</p>

🟤 117.415 filas

🟤 20 columnas, con variables categoricas y numericas como:   ['Fecha', 'Tipo Mov.', 'Fac. Tipo', 'Fac. Suc.', 'Fac. Nun.','Fisc. Numero', 'Tipo Pago', 'Cant.', 'Precio', 'Producto', 'Sub. Total', 'Rubro', 'Cobertura', 'Ajustes', 'Desc. Adic.', 'Total. Cliente', 'IVA', 'Tasa Iva', 'Total Gravado', 'Total sin Gravar'] 

🟤 Más de 7.500 productos distintos

🟤 Rubro farmacia (medicamentos, insumos médicos, suplementos, vitaminas, salud preventiva) y perfumería y cuidado personal (cremas, protectores, higiene). 

- Se trata de un conjunto de datos reales, lo que implica la presencia de ruido, valores inconsistentes, formatos heterogéneos y registros incompletos, características habituales en fuentes operativas del sector farmacéutico. Esta naturaleza del dataset representó un desafío significativo durante la etapa de data cleaning, ya que fue necesario aplicar múltiples técnicas de depuración, normalización y validación para garantizar la calidad de los datos antes de avanzar con el análisis y el modelado.




## ➖ METODOLOGÍA 

🔸 *Importación de librerías y cargado de dataset*

🔸 *Análisis inicial de estructura de dataframe, tipo de variables y calidad*

🔸 *Limpieza de datos nulos, duplicados, vacíos y outliers*

🔸 *Renombrado y orden de columnas*

🔸 *Transformaciones de columnas: nuevas columnas y asignaciones grupales*

🔸 *Análisis Exploratorio (EDA)*

🔸 *Visualizaciones*

🔸 *Guardado de resultados y observaciones del análisis*

🔸 *Normalización, encoding de datos y separación train/test*

🔸 *Creación y prueba de distintos modelos*

🔸 *Entrenamiento de modelo*

🔸 *Hiperparametrización del modelo*

🔸 *Calculo de precisión y errores*

🔸 *Optimización de EDA*

🔸 *Nuevos entrenamientos de modelo y optimización del modelo elegido*

🔸 *Creación y uso de url con Streamlit*

🔸 *Prueba real como usuario externo*


## 📝 FASES DE PROYECTO

### 🔽 Paso 1: Definición de nuestro problema

 Actualmente, la farmacia gestiona su stock principalmente a partir de la experiencia del personal, la intuición comercial y el análisis manual de ventas pasadas. Si bien este enfoque puede funcionar en escenarios estables, resulta insuficiente en un contexto dinámico y volátil como el argentino, donde los hábitos de consumo, los precios y la disponibilidad de productos cambian de forma constante.

 La ausencia de una metodología analítica y sistematizada para prever la demanda futura genera decisiones reactivas en lugar de estratégicas; esto genera:

❌ Quiebres de stock en productos críticos

❌ Sobrestock en productos de baja rotación

❌ Uso ineficiente del capital de trabajo
 
 - En definitiva, la farmacia no dispone hoy de una herramienta objetiva, basada en datos, que le permita anticiparse a las necesidades reales de sus pacientes y clientes.*


### 🔽 Paso 2: Adquisición y carga de datos

- Los datos provienen de un software real utilizado en farmacias argentinas:
✔️ El sistema exporta archivos en formato Excel encriptado. Se realizó un proceso previo de desencriptado y conversión a CSV.
✔️ Luego se cargaron en Python usando Pandas creando un Dataframe.

- En esta etapa se inspeccionaron columnas, verificaron tipos de datos y se evaluó la calidad inicial (nulos, duplicados).

👉 Resultado: un dataset crudo listo para limpieza y análisis.

### 🔽 Paso 3: Almacenamiento de datos

- Los datos fueron almacenados en una base de datos SQLite para facilitar su acceso, seguridad y reutilización.

- Desde Python se ejecutaron consultas SQL (SELECT, GROUP BY, WHERE) para generar vistas preliminares que sirvieron como base para el análisis exploratorio.

### 🔽 Paso 4: Realización de análisis descriptivo

En esta etapa se trabajó con estadísticas básicas y descriptivas para entender las principales variables del dataset:

✔️ Frecuencia de productos.
✔️ Distribuciones de ventas.
✔️ Medidas de tendencia central (media, mediana, moda).
✔️ Dispersión (desvío estándar) y comportamientos atípicos.


### 🔽 Paso 5: Full EDA

Análisis Exploratorio de Datos (EDA)

- Limpieza: eliminación de duplicados y nulos, normalización de texto (sin acentos, minúsculas, sin símbolos).
- Transformación: conversión de fechas, creación de variables (año, mes, día).
- Exploración: investigación, consultas al personal de farmacia, productos más vendidos, patrones por mes, comparaciones por categorías.
- Visualizaiones: gráficos de barras, histogramas y análisis temporal.

### 🔽 Paso 6: Creación de modelo y optimización de parámetros

🔸 Selección del modelo
- Inicialmente se eligió Random Forest por intuición y buen desempeño esperado.
- El objetivo era establecer métricas base para luego comparar con otros modelos.

🔸 Estrategia de partición (Train/Test)
- División inicial: 80% Train / 20% Test.
- Como los datos están organizados por semanas, una división aleatoria podía dejar productos en Test que no existieran en Train.
- Se realizó el split por bloques de semanas, garantizando que cada semana (en Train o Test) contenga todos los productos evaluados.

🔸 Primer modelo sin hiperparámetros (Random Forest)
- RMSE: 4.95
- R² Test: 69%
- R² Train: 96%
  - Se detectó overfitting, esperado en Random Forest sin ajuste fino.

🔸 Optimización de hiperparámetros (Random Forest)
- Primero se aplicó RandomizedSearchCV.
- Luego se afinó con GridSearchCV usando los mejores valores encontrados.
- - Modelo optimizado:
  -  R² Train: 85%
  -  R² Test: 70%
  -  RMSE: 4.87

🔸 Prueba de otros modelos
- Se evaluaron modelos alternativos.
- Se seleccionó CatBoost como candidato principal.

✔️ Modelo final (CatBoost ajustado)
- R² Test: 72%
- RMSE: 4.70
- Mejora respecto a Random Forest en capacidad de generalización.



### 🔽 Paso 7: Visualización e interacción con el modelo - Streamlit
- Flujo de la aplicación de Streamlit  

🔸 Carga de datos en tiempo real
   - El usuario sube archivos CSV con ventas recientes. La app ejecuta automáticamente procesos de:
   - Limpieza de datos
   - Normalización de texto
   - Asignación de categorías mediante un diccionario inteligente (JSON) desarrollado durante el EDA  

🔸 Actualización inteligente del histórico
- La aplicación detecta si los datos cargados son nuevos o duplicados. En función de eso:
- Actualiza el histórico de ventas
- Recalcula variables temporales (lags)
- Garantiza que las predicciones usen siempre la información más reciente


🔸Generación de predicciones (Forecasting)
- La app permite dos modos de análisis:
    - Modo Individual: selección de un producto y predicción de demanda a 1, 2, 4 u 8 semanas
    - Modo Global: ranking de productos con mayor demanda proyectada por grupo o categoría


🔸 Visualización interactiva de resultados
- Los resultados se muestran de forma clara mediante:
- Gráficos de series temporales
- Tablas comparativas por producto, grupo o categoría


🔸Exportación de resultados
- El usuario puede descargar las predicciones en formato CSV para:
- Integrarlas en sistemas de gestión internos
- Facilitar decisiones de compra y stock



## 📁 Estructura de proyecto

```
sp-ml-20-final-project-g2/
├── 📁 data/                # Raw and processed datasets
│    ├── 📁 interin/        # For intermediate data that has been transformed.
│    ├── 📁 processed/      # For the final data to be used for modeling.
│    ├── 📁 raw/            # For raw data without any processing.
├── 📁 database/            # SQL scripts and database configs
├── 📁 docs/                # Documentation and presentation materials
├── 📁 models/              # Trained model artifacts
     ├──model-pharmacy
├── 📁 notebooks/           # Jupyter notebooks for EDA and analysis
├── 📁 src/                 # Source code modules
     ├──EDA.ipynb
├── 📁 webapp/              # Flask/Streamlit application
```

## 🛠️ Tecnologías utilizadas

- Excel
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SQLite
- Randomforest
- XGBoost
- Streamlit

## 📊 Resultados

*[To be updated with our model performance and insights]*

## 🌐 Live Demo

*[Link to be added when the web application is deployed]*

## 🧠 Valor del Proyecto

✔️ Conecta datos reales con problemas reales
✔️ Tiene pipeline profesional (CSV → DB → EDA)
✔️ Apunta a solución de negocio
