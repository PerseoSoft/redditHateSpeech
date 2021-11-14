# Caracterización de discurso de odio en r/argentina
## Introducción

El presente repo contiene el código correspondiente al proyecto final de la materia [Minería de datos para texto](https://sites.google.com/unc.edu.ar/textmining2021/), a cargo de Laura Alonso i Alemany.

Objetivo del proyecto: Caracterizar discursos de odio dentro de la comunidad de [Reddit Argentina](https://reddit.com/r/argentina). Esto es, detectarlos y encontrar sub-lenguajes de odio en los mismos.

Para realizar esto, se llevó a cabo un proceso consistente en 5 etapas, como se muestra en la siguiente figura:

![pipeline_reddit](/misc/workflow.drawio.png)


Cada etapa tiene su correspondiente notebook:

1. Obtención del conjunto de comentarios de a través de la API de Reddit ([notebook](https://github.com/EvaVillarrealGuzman/redditHateSpeech/blob/main/src/1_pipeline_download_reddit_comments.ipynb)).
   
2. Pre-procesamiento del mismo ([notebook](https://github.com/EvaVillarrealGuzman/redditHateSpeech/blob/main/src/2_pipeline_preprocessing.ipynb)).

3. Aplicación de embeddings y categorización en clústers (notebook [LDA](https://github.com/EvaVillarrealGuzman/redditHateSpeech/blob/main/src/3a_pipeline_lda.ipynb) [Word2Vec](https://github.com/EvaVillarrealGuzman/redditHateSpeech/blob/main/src/3b_pipeline_embedding_word2vec.ipynb) [FastText](https://github.com/EvaVillarrealGuzman/redditHateSpeech/blob/main/src/3c_pipeline_embedding_fasttext.ipynb)).

4. Entrenamiento de un modelo de detección de odio y extracción de palabras de odio en cada dataset ([notebook](https://github.com/EvaVillarrealGuzman/redditHateSpeech/blob/main/src/4_detect_hate_speech.ipynb)).
Para realizar el entrenamiento de los modelos, es necesario contar con los datasets respectivos de cada competencia (Hateval, DETOXIS, MeOffendMex) que se desee entrenar.

5. Uso del modelo para predecir los comentarios recolectados ([notebook](https://github.com/EvaVillarrealGuzman/redditHateSpeech/blob/main/src/5_pipeline_hate_speech.ipynb)).

6. Combinación de dicho modelo con las categorías encontradas para encontrar correlaciones ([notebook](https://github.com/EvaVillarrealGuzman/redditHateSpeech/blob/main/src/6_pipeline_result.ipynb)).

**Este informe y proyecto estan en proceso 🚧🔨, todavía sujetos a cambios, correcciones, y mejoras**

- https://github.com/jfreddypuentes/spanlp
- https://becominghuman.ai/detecting-gender-based-hate-speech-in-spanish-with-natural-language-processing-cdbba6ec2f8b
- https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/
- https://www.jcchouinard.com/reddit-api/
- https://towardsdatascience.com/religion-on-twitter-5f7b84062304
- https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
- https://towardsdatascience.com/applying-machine-learning-to-classify-an-unsupervised-text-document-e7bb6265f52
- https://medium.com/@rohithramesh1991/unsupervised-text-clustering-using-natural-language-processing-nlp-1a8bc18b048d
- https://xplordat.com/2018/12/14/want-to-cluster-text-try-custom-word-embeddings/
- https://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/
- https://adrian-rdz.github.io/NLP_word2vec/
- https://paperperweek.wordpress.com/2018/04/09/best-ways-to-cluster-word2vec/
- https://www.kdnuggets.com/2018/04/robust-word2vec-models-gensim.html
- https://www.baeldung.com/cs/ml-word2vec-topic-modeling
- https://www.kaggle.com/szymonjanowski/internet-articles-data-with-users-engagement
- https://medium.com/ml2vec/using-word2vec-to-analyze-reddit-comments-28945d8cee57
- https://towardsdatascience.com/clustering-with-more-than-two-features-try-this-to-explain-your-findings-b053007d680a
- https://towardsdatascience.com/k-means-clustering-8e1e64c1561c
- https://towardsdatascience.com/fit-vs-predict-vs-fit-predict-in-python-scikit-learn-f15a34a8d39f