import string
import re

import spacy

import emoji


def give_emoji_free_text(text):
    """
    Removes emoji's from tweets
    Accepts:
        Text (tweets)
    Returns:
        Text (emoji free tweets)
    """
    emoji_list = [c for c in text if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text


def url_free_text(text):
    '''
    Cleans text from urls
    '''
    text = re.sub(r'http\S+', '', text)
    return text


def email_free_text(text):
    '''
    Cleans text from emails
    '''
    text = re.sub('\S*@\S*\s?', '', text)
    return text


def quotes_free_text(text):
    '''
    Cleans text from quotes
    '''
    text = re.sub("\'", "", text)
    return text


def get_lemmas(text):
    '''Used to lemmatize the processed tweets'''
    lemmas = []

    doc = nlp(text)

    # Something goes here :P
    for token in doc:
        if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN', 'PROPN']):
            lemmas.append(token.lemma_)

    return lemmas


# Tokenizer function
def tokenize(text):
    """
    Parses a string into a list of semantic units (words)
    Args:
        text (str): The string that the function will tokenize.
    Returns:
        list: tokens parsed out
    """
    # Removing url's
    pattern = r"http\S+"

    tokens = re.sub(pattern, "", text) # https://www.youtube.com/watch?v=O2onA4r5UaY
    tokens = re.sub('[^a-zA-Z 0-9]', '', text)
    tokens = re.sub('[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    tokens = re.sub('\w*\d\w*', '', text) # Remove words containing numbers
    tokens = re.sub('@*!*\$*', '', text) # Remove @ ! $
    tokens = tokens.strip(',') # TESTING THIS LINE
    tokens = tokens.strip('?') # TESTING THIS LINE
    tokens = tokens.strip('!') # TESTING THIS LINE
    tokens = tokens.strip("'") # TESTING THIS LINE
    tokens = tokens.strip(".") # TESTING THIS LINE

    tokens = tokens.lower().split() # Make text lowercase and split it

    return tokens


def preprocess_corpus(corpus):
    preprocessed_corpus = []

    for idx, line in enumerate(corpus):

        line = give_emoji_free_text(line)
        line = url_free_text(line)
        line = email_free_text(line)
        line = quotes_free_text(line)
        line = tokenize(line)

        preprocessed_corpus.append(' '.join(line))
        
    return preprocessed_corpus


nlp = spacy.load('es_core_news_sm', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
