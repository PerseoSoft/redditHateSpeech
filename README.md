Caracterización de discurso de odio en r/argentina

---

Índice
- [Vistazo rápido](#vistazo-rápido)
  - [Flujo de datos generados](#flujo-de-datos-generados)
- [Introducción](#introducción)
  - [Discurso de odio](#discurso-de-odio)
  - [reddit](#reddit)
  - [r/argentina](#rargentina)
- [Obtención de datos](#obtención-de-datos)
- [Pre-procesamiento](#pre-procesamiento)
- [Embeddings](#embeddings)
- [Entrenamiento de detector de odio](#entrenamiento-de-detector-de-odio)
- [Aplicación del modelo a los comentarios](#aplicación-del-modelo-a-los-comentarios)
- [Análisis de resultados](#análisis-de-resultados)
- [Conclusiones](#conclusiones)
- [Trabajos futuros](#trabajos-futuros)
- [Texto no asignado](#texto-no-asignado)
- [Backlog](#backlog)
- [Fuentes consultadas para el trabajo](#fuentes-consultadas-para-el-trabajo)
  - [Discursos de odio](#discursos-de-odio)
  - [reddit API](#reddit-api)
  - [Procesamiento de lenguaje natural](#procesamiento-de-lenguaje-natural)
  - [Clustering](#clustering)
  - [Trabajos relacionados](#trabajos-relacionados)


## Vistazo rápido

El presente repo contiene el código correspondiente al proyecto final de la materia [Minería de datos para texto](https://sites.google.com/unc.edu.ar/textmining2021/), a cargo de Laura Alonso i Alemany.

Objetivo del proyecto: Caracterizar discursos de odio dentro de la comunidad de [reddit Argentina](https://reddit.com/r/argentina). Esto es, detectarlos y encontrar sub-lenguajes de odio en los mismos.

Para realizar esto, se llevó a cabo un proceso consistente en 5 etapas, como se muestra en la siguiente figura:

![pipeline_reddit](/misc/workflow.drawio.png)


Cada etapa tiene su correspondiente notebook:

1. Obtención del conjunto de comentarios de a través de la API de Reddit ([notebook](https://github.com/PerseoSoft/redditHateSpeech/blob/main/src/1_pipeline_download_reddit_comments.ipynb)).
   
2. Pre-procesamiento del mismo ([notebook](https://github.com/PerseoSoft/redditHateSpeech/blob/main/src/2_pipeline_preprocessing.ipynb)).

3. Aplicación de embeddings y categorización en clústers (notebook [LDA](https://github.com/PerseoSoft/redditHateSpeech/blob/main/src/3a_pipeline_lda.ipynb) [Word2Vec](https://github.com/PerseoSoft/redditHateSpeech/blob/main/src/3b_pipeline_embedding_word2vec.ipynb) [FastText](https://github.com/PerseoSoft/redditHateSpeech/blob/main/src/3c_pipeline_embedding_fasttext.ipynb)).

4. Entrenamiento de un modelo de detección de odio y extracción de palabras de odio en cada dataset ([notebook](https://github.com/PerseoSoft/redditHateSpeech/blob/main/src/4_detect_hate_speech.ipynb)).
Para realizar el entrenamiento de los modelos, es necesario contar con los datasets respectivos de cada competencia (Hateval, DETOXIS, MeOffendMex) que se desee entrenar.

5. Uso del modelo para predecir los comentarios recolectados ([notebook](https://github.com/PerseoSoft/redditHateSpeech/blob/main/src/5_pipeline_hate_speech.ipynb)).

6. Combinación de dicho modelo con las categorías encontradas para encontrar correlaciones ([notebook](https://github.com/PerseoSoft/redditHateSpeech/blob/main/src/6_pipeline_result.ipynb)).

**Este informe y proyecto estan en proceso 🚧🔨, todavía sujetos a cambios, correcciones, y mejoras**


### Flujo de datos generados

Los distintos notebooks forman un pipeline en el cuál cada uno utiliza los datos generados por el anterior. Se listan cada una de las entradas:

1. Obtención de comentarios. 
    - Archivos de entrada: N/A. 
    - Archivo de salida: *docs/reddit_data.csv*: CSV que contiene los comentarios de reddit descargados

2. Pre-procesamiento del dataset.
    - Archivos de entrada: *docs/reddit_data.csv*.
    - Archivos de salida: *docs/preprocessing_reddit_data.csv*: CSV con los comentarios pre-procesados.
   

3. Embeddings y clustering.
    - Archivos de entrada: *docs/preprocessing_reddit_data.csv*.
    - Archivos de salida: 
      - *docs/reddit_data_<método>.csv*, donde *<método>* puede ser 'lda', o 'word2vec', 'fasttext'. Cada uno de estos archivos toma el dataset pre-procesado y le agrega el número de clúster al que pertenecería cada comentario, según su cercanía.
      - *docs/models/<model>.model*, el modelo entrenado. Puede ser 'word2vec', o 'fasttext'. 
      - *docs/models/<model>_kmeans.model*, el modelo de k-means entrenado usando los embeddings de <model> (para 'word2vec' y 'fasttext').


4. Entrenamiento y selección del modelo.
   - Archivos de entrada: *docs/hateval2019/hateval2019_es_train.csv*, *docs/detoxis_data/train.csv*, y *docs/MeOffendEs/mx-train-data-non-contextual.csv*. Estos archivos requieren la descarga previa manual de cada dataset.
   - Archivos de salida: para cada dataset, se guarda:
     - Palabras de odio de cada modelo: *docs/palabras_odio.csv*.
     - Vectorizador: *docs/models/<dataset>_vectorizer.pkl* donde *<dataset>* es hateval, detoxis, o meoffendmex.
     - Modelo entrenado: *docs/models/<dataset>_<iniciales_modelo>_model.pkl* donde *<iniciales_modelo>* es 'lr', 'rf', o 'nb'.
   - Archivos de salida (de prueba): Predicciones: *docs/test/reddit_<dataset>_hate_comments.csv*, uno para cada <dataset>: 'hateval', 'detoxis', 'meoffendmex'.
   
5. Aplicación del modelo en comentarios de reddit. 
   - Archivos de entrada: *docs/reddit_data_<método>.csv*.
   - Archivos de salida:
     - *docs/reddit_data_hate_speech.csv* - CSV que toma  **TODO**
6. Análisis de resultados.
   - Archivos de entrada: *docs/reddit_data_hate_speech.csv*
   - Archivos de salida: N/A.
## Introducción

### Discurso de odio

Hay varias posturas sobre lo que es discurso de odio, en general se coincide en que es un discurso que:

1. Apunta a un grupo o individuo, basado en algún aspecto como su orientación sexual, religión, nacionalidad, etc.
2. Busca humillar, discriminar o propagar el odio/hostilidad hacia ese grupo.
3. Tiene una intención deliberada.

Su manifestación en Internet, además:
1. Puede motivar formas de agresión en línea.
2. Permite propagar el discurso de odio con velocidad.
3. Permite que el discurso se mantenga y comparta con facilidad.
4. Facilita la generación de cámaras de eco.
5. Al estar en servidores privados, permite que ciertas empresas intenten eludir su control, y usarlos para mantener sus usuarios interactuando con el servicio.

### r/argentina

[Reddit](https://www.reddit.com/) es  una red social de “comunidades” creadas por usuarios. En este proyecto, nos centramos en [reddit argentina](https://www.reddit.com/r/argentina/).

En la siguiente imagen podemos ver la estructura general de un post en reddit:

![](misc/reddit.png)

En cada comunidad sus miembros hacen posts, y cada post puede ser comentado generando debate.

Su aspecto distintivo, es que cada post o comentario recibe votos, con el objetivo de que aquellos posts o comentarios que más aportan aparezcan encima de los que no. También se pueden premiar a aquellos destacados.

## Obtención de datos

Para la obtención de los datos se utilizó un *wrapper* de la API de reddit, llamado [praw](https://praw.readthedocs.io/en/stable/index.html), a partir del cuál descargamos comentarios de diferentes *post* del *subreddit* argentina, así como las respuestas de los comentarios.
Los comentarios en reddit pueden ser *link* o pueden ser solo textos. Filtramos solamente los comentarios que tengan textos. A la vez solo se consideraron comentarios que tuvieran como mínimo cierta cantidad de caracteres.

De cada comentario que se guardó de reddit, se obtuvieron los siguientes datos:
- *id*: identificador de reddit. Se guardó por cuestiones de trazabilidad.
- *comment_parent_id*: identificador del *post* o comentario al cuál responde el comentario actual en caso que corresponda. Se guardó por cuestiones de trazabilidad.
- *flair*: es el tipo de comentario etiquetado por reddit, por ejemplo, política, economía, humor, etc.
- *comms_num*: número de respuestas que recibió el comentaio.
- *score*: es un puntaje que los usuarios le dan al comentario.

## Pre-procesamiento

El pre-procesamiento consistió en:

- Eliminar emojis, urls, comillas, caracteres especiales, puntuaciones.
- Aplicar tokenización: en cada comentario, el token era la palabra.
- Conversión a minúscula.
- Eliminación de stopwords utilizando spaCy.
- Lematización utilizando spaCy.
- Construir bigramas y trigramas.

## Embeddings

Para poder detectar las subcomunidades dentro de reddit comenzamos utilizando LDA. Sin embargo, los resultados que obtuvimos no fueron satisfactorios, ya que a la hora de realizar un análisis de los tópicos identificados por el modelo, encontramos poca cohesión entre los temas.

A raíz de esto, probamos con *word embedding* donde obtuvimos resultados que captan mucho mejor la semántica de la información. El proceso que llevamos a cabo en *word embedding* para obtener las subcomunidades fue:

1. Generar una representación vectorial de los comentarios: se mapearon los comentarios a partir de palabras en vectores numéricos.
2. Aplicamos un algoritmo de *clustering*, particularmente *k-means*, donde las características que se pasaron son los vectores numéricos obtenidos en el paso anterior.

Utilizamos dos técnicas de *word embedding*: primero usamos Word2Vec y luego FastText.

A continuación mostramos algunos comentarios que fueron agrupados a través de las diferentes técnicas aplicadas. Un evento particular que sucedió durante la descarga de estos datos en reddit fue el debate de la ley de etiquetados en Argentina. Vamos a comparar las subcomunidades obtenidos en cada técnica analizando particularmente la subcomunidad referida a este evento.

### Embedding con LDA

En la siguiente imagen se pueden observar algunos de los tópicos identificados por LDA.

![](misc/embedding_1.png)

El tópico número 91, **piedra - etiqueta - pan - mira**, incluye comentarios sobre la tratativa de la ley de etiquetado y temas que tienen que ver con la comida en general. Algunos comentarios son:

1. Me alegro mucho, seguro muy feliz todos por el reencuentro. Igual te recomiendo que no coma directo de la lata, pasale a un platito o comedero. Entiendo que a veces ni te dan tiempo.
2. Todo mi secundario el desayuno fue un fantoche triple y una lata de coca.  Y sólo gastaba 2$. Qué buenos tiempos.
3. La manteca no hace mal. Es muy difícil comer exceso de grasas para tu cuerpo en comparación con lo fácil que es atiborrarte con azúcar y carbohidratos. Esos son los verdaderos enemigos
4. Y con etiquetas que te dicen cuánta grasa tiene un kilo de bayonesa
5. Alta banfest se van a mandar los mods con este thread. Despedite de tu cuenta, maquinola, denunciado

### Embedding con Word2Vec

En la siguiente imagen se pueden observar algunas de las subcominidades identificados por Word2Vec.

![](misc/embedding_2.png)

El *cluster* número 94, **ley - etiquetado - proyecto**, incluye comentarios sobre la tratativa de la ley de etiquetado y temas que tienen que ver con las leyes en general. Algunos comentarios son:

1. Una prueba mas de la ley de oferta y demanda
2. Con la nueva ley no le podés regalar leche entera o un alfajor a un comedor, decir comida basura en un país donde el 50% de los chicos no hacen toda las comidas es lo más clasista que existe.
3. Recuerden la ley de alquileres.... Fué sancionada con un beso muy fuerte de los K, PRO y demás muchachos...
4. No entiendo cómo hay tanta gente en contra de una ley que no te cambia un carajo tu vida. Es la ley más anodina que sacó el Kirchnerismo en toda su historia creo
5. Pero hay leyes contra la violencia de genero! Como paso esto!!!1!?
6. No existe tal cosa en Argentina. Existe el Estado de Sitio, pero no se asemeja para nada a una ley marcial.. El concepto de ley marcial como tal, desapareció en el 94 con la nueva Constitución.

### Embedding con FastText

En la siguiente imagen se pueden observar algunas de las subcominidades identificados por FastText.

![](misc/embedding_3.png)

Como se puede ver en el cluster **jaja - jajaja - jajajar - jajajaja - jajaj**, FastText identifica mejor las alteraciones que pueden suceder dentro de una palabra.

El *cluster* número 113, **ley - etiquetado - votar**, incluye comentarios sobre la tratativa de la ley de etiquetado y temas que tienen que ver con las leyes en general. Algunos comentarios son:

1. Feriado con fines turísticos. Ley 27.399
2. ajajaja como los cagaron a los primeros. como siempre la ley aplica a todos por igual /s
3. El sticker en Chile fue durante la transición de la ley. Imagínate tener productos fabricados y tener que cambiar la envoltura a todos para que cumplan la ley
4. Gracias gloriosa ley de regulación de alimentos, ahora se que desayunar coca cola con surtidos bagleys esta mal
5. Eso y que la ley va a prohibir vender dulces y gaseosas en los colegios, y usar imágenes de famosos en los envases.
6. Eso está por la ley Micaela no?. Tipo esta clase de capacitaciones no?
7. y ahora Lipovetzky reconoce lo de la ley de alquileres


## Entrenamiento de detector de odio

## Aplicación del modelo a los comentarios

## Análisis de resultados


## Conclusiones

## Trabajos futuros
## Fuentes consultadas para el trabajo


### Discursos de odio

- https://en.wikipedia.org/wiki/Hate_speech
- https://www.rightsforpeace.org/hate-speech
- https://fsi.stanford.edu/news/reddit-hate-speech
- https://variety.com/2020/digital/news/reddit-bans-hate-speech-groups-removes-2000-subreddits-donald-trump-1234692898
- https://www.reddithelp.com/hc/en-us/articles/360045715951-Promoting-Hate-Based-on-Identity-or-Vulnerability

### reddit API

- https://www.jcchouinard.com/reddit-api/


### Procesamiento de lenguaje natural

- Foundations of Statistical Natural Language Processing - Manning & Schütze (1999)
- https://spacy.io
- https://radimrehurek.com/gensim/
- https://www.nltk.org
- https://www.baeldung.com/cs/ml-word2vec-topic-modeling
- https://www.kdnuggets.com/2018/04/robust-word2vec-models-gensim.html
- https://adrian-rdz.github.io/NLP_word2vec/
- https://towardsdatascience.com/applying-machine-learning-to-classify-an-unsupervised-text-document-e7bb6265f52
- https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
- https://www.roelpeters.be/calculating-mutual-information-in-python/

### Clustering

- https://towardsdatascience.com/k-means-clustering-8e1e64c1561c
- https://paperperweek.wordpress.com/2018/04/09/best-ways-to-cluster-word2vec/
- https://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/
- https://medium.com/@rohithramesh1991/unsupervised-text-clustering-using-natural-language-processing-nlp-1a8bc18b048d
- https://xplordat.com/2018/12/14/want-to-cluster-text-try-custom-word-embeddings/
- https://towardsdatascience.com/clustering-with-more-than-two-features-try-this-to-explain-your-findings-b053007d680a

### Trabajos relacionados

- https://github.com/jfreddypuentes/spanlp
- https://medium.com/ml2vec/using-word2vec-to-analyze-reddit-comments-28945d8cee57
- https://www.kaggle.com/szymonjanowski/internet-articles-data-with-users-engagement
- https://towardsdatascience.com/religion-on-twitter-5f7b84062304
- https://becominghuman.ai/detecting-gender-based-hate-speech-in-spanish-with-natural-language-processing-cdbba6ec2f8b
- https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/
