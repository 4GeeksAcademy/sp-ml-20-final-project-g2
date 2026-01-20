# Proyecto final Machine Learning

Este es el proyecto final de nuestro bootcamp de Machine Learning, donde demostramos las habilidades y conocimientos adquiridos a lo largo de nuestros estudios. A lo largo de este bootcamp, hemos estudiado diferentes modelos basados en proyectos de diferentes áreas y tipos. Ahora es el momento de crear nuestro propio proyecto utilizando el algoritmo que creemos que se adapta mejor a nuestro problema.

Tendremos que encontrar un conjunto de datos adecuado para trabajar, procesarlo, entrenar un modelo y, finalmente, ponerlo a disposición para su consumo.

> *"Hard work always beats talent when talent doesn't work hard"* - Tim Notke

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

  La idea principal es pasar de una gestión basada solo en la experiencia a una gestión basada en datos, que permita anticiparse a las necesidades de los clientes, evitar faltantes de productos importantes y reducir el exceso de mercadería en un contexto económico cambiante. Todo el enfoque está pensado desde la realidad del negocio farmacéutico y el comportamiento de consumo de las personas.]*
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

### Dataset
<p align="justify">
- El dataset contiene información de registro de ventas durante el año 2025 de una farmacia situada en Argentina, en la provincia de Buenos Aires. El registro se corresponde a los datos de los tickets de venta generados durante todo el año, los días que el comercio estuvo abierto. Teniendo en cuenta que el comercio trabaja de Lunes a Sábados de 8hs a 20 hs, es decir, 12 hs por día, se han generado un total de datos tal que nuestro dataset contiene:
</p>

- 117.415 filas
- 20 columnas, con variables categoricas y numericas como:   ['Fecha', 'Tipo Mov.', 'Fac. Tipo', 'Fac. Suc.', 'Fac. Nun.','Fisc. Numero', 'Tipo Pago', 'Cant.', 'Precio', 'Producto', 'Sub. Total', 'Rubro', 'Cobertura', 'Ajustes', 'Desc. Adic.', 'Total. Cliente', 'IVA', 'Tasa Iva', 'Total Gravado', 'Total sin Gravar'] 

- Más de 7.500 productos distintos
- Rubro farmacia (medicamentos, insumos médicos, suplementos, vitaminas, salud preventiva) y perfumería y cuidado personal (cremas, protectores, higiene). 

- Se trata de un conjunto de datos reales, lo que implica la presencia de ruido, valores inconsistentes, formatos heterogéneos y registros incompletos, características habituales en fuentes operativas del sector farmacéutico. Esta naturaleza del dataset representó un desafío significativo durante la etapa de data cleaning, ya que fue necesario aplicar múltiples técnicas de depuración, normalización y validación para garantizar la calidad de los datos antes de avanzar con el análisis y el modelado.


### Methodology
*To be defined - we will document our chosen approach and algorithms*

### Results
*To be updated with our findings and model performance*

## 📝 Project Phases

### Step 1: Problem Definition

 Actualmente, la farmacia gestiona su stock principalmente a partir de la experiencia del personal, la intuición comercial y el análisis manual de ventas pasadas. Si bien este enfoque puede funcionar en escenarios estables, resulta insuficiente en un contexto dinámico y volátil como el argentino, donde los hábitos de consumo, los precios y la disponibilidad de productos cambian de forma constante.

 La ausencia de una metodología analítica y sistematizada para prever la demanda futura genera decisiones reactivas en lugar de estratégicas. Esto se traduce en quiebres de stock en productos críticos para la atención sanitaria, sobrestock de artículos de baja rotación y una utilización ineficiente del capital de trabajo. En definitiva, la farmacia no dispone hoy de una herramienta objetiva, basada en datos, que le permita anticiparse a las necesidades reales de sus pacientes y clientes.*


### Step 2: Acquiring and Loading the Data Set
Since in the real world data does not usually arrive in a flat csv file, this data must be acquired by one of the following ways:
- Extracting data from some web page or portal using web scraping techniques
- Exploitation of a public database using SQL language (the database must support this language)
- Exploitation of a public API to obtain data

Once you have the data, you must store it in a CSV document and load it into Python using Pandas.

**NOTE:** Depending on the dataset and the case study to be explored, datasets downloaded by other means could be evaluated and accepted.

### Step 3: Store the Information
A widely used practice is to store the data, especially if they are massive, in a database for quick access to them. From all the databases we have studied, choose the one most compatible with your data and store it there. Then, perform queries using Python (with pure SQL code or using the wrappers we have studied in the course) to use the different statements: SELECT, JOIN, INSERT.... These queries must provide a value to start the analysis on the data prior to the statistics and EDA.

It is important to understand that in the real world we do not only have CSV as an ally to store data, since it is easier to lose a flat file like CSV than a database with its connections and data models inside. Security is also a critical and important factor for storing your data there, since a CSV does not provide any protection mechanism that other technologies do.

### Step 4: Perform a Descriptive Analysis
The raw data stored in a database can be a great and very valuable source of information. Before we begin to simplify and exploit them with EDA, we must know their fundamental statistical measures: means, modes, distributions, deviations, etcetera. Analyze the descriptive statistical variables of each of the predictors of the data set and theorize about the distribution that each of them follows.

Use hypothesis tests if you consider it necessary.

### Step 5: Perform a Full EDA
This step is vital to ensure that we keep the variables that are strictly necessary and eliminate those that are not relevant or do not provide information. Use the example Notebook we worked on and adapt it to this use case.

Make sure to conveniently divide the data set into train and test as we have seen in previous lessons.

### Step 6: Build the Model and Optimize It
Once you have your data ready, decide which model fits best and train it. If in doubt, try using several of the ones you have already studied. Select the one that best fits the data.

Remember that the hyperparameter optimization step is very important to explore and achieve the best version of the model.

### Step 7: Deploy the Model
Create a Machine Learning web application using your saved model. You can use Flask, Streamlit or any other tool you know. Use Heroku, Render or another cloud computing platform of your choice to deploy your web application and share it with the world. Remember that the application is going to be the gateway to potential users or customers, and you have to take care of even the smallest detail.

## 📁 Project Structure

```
ml-project-repo/
├── 📁 data/                # Raw and processed datasets
│    ├── 📁 interin/        # For intermediate data that has been transformed.
│    ├── 📁 processed/      # For the final data to be used for modeling.
│    ├── 📁 raw/            # For raw data without any processing.
├── 📁 database/            # SQL scripts and database configs
├── 📁 docs/                # Documentation and presentation materials
├── 📁 models/              # Trained model artifacts
├── 📁 notebooks/           # Jupyter notebooks for EDA and analysis
├── 📁 src/                 # Source code modules
├── 📁 webapp/              # Flask/Streamlit application
```

## 🛠️ Technologies Used

*[To be updated as we select our tech stack]*

## 📊 Results

*[To be updated with our model performance and insights]*

## 🌐 Live Demo

*[Link to be added when the web application is deployed]*
