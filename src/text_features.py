#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
from collections import Counter


# ── pré-processamento ──────────────────────────────────────────────────────────

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)       # remove tags HTML
    text = re.sub(r"[^a-z\s]", "", text)   # remove pontuação e números
    return text


def build_vocab(texts, max_words=10000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    most_common = counter.most_common(max_words)
    word_index = {word: i for i, (word, _) in enumerate(most_common)}
    return word_index

def truncate_text(text, max_words=120, min_words=80):
    if not isinstance(text, str):
        return None
    words = text.split()
    if len(words) >= min_words:
        return ' '.join(words[:max_words])
    return None


# ── Bag of Words (one-hot) ─────────────────────────────────────────────────────

def vectorize_text(text, word_index, max_words):
    vector = np.zeros(max_words, dtype=np.float32)
    for word in set(text.split()):
        if word in word_index:
            vector[word_index[word]] = 1
    return vector


def texts_to_bow(texts, word_index, max_words):
    return np.array([vectorize_text(t, word_index, max_words) for t in texts])


# ── TF-IDF  ────────────────────────────────────────────────────────────

class TFIDFVectorizer:

    def __init__(self, max_words=10000, ngram_range=(1, 1)):
        self.max_words = max_words
        self.ngram_range = ngram_range
        self.word_index = {}
        self.idf = None

    def _get_ngrams(self, tokens):
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            ngrams.extend([' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        return ngrams

    def fit(self, texts):
        df_counter = Counter()
        for text in texts:
            tokens = text.split()
            df_counter.update(set(self._get_ngrams(tokens)))

        most_common = df_counter.most_common(self.max_words)
        self.word_index = {word: i for i, (word, _) in enumerate(most_common)}

        n_docs = len(texts)
        df = np.array([df_counter[w] for w in self.word_index], dtype=np.float32)
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), len(self.word_index)), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.split()
            ngrams = self._get_ngrams(tokens)
            tf_counter = Counter(ngrams)
            total = len(ngrams)
            for ngram, count in tf_counter.items():
                if ngram in self.word_index:
                    tf = count / (total + 1e-10)
                    j = self.word_index[ngram]
                    X[i, j] = tf * self.idf[j]
        # normalização L2 por linha
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + 1e-10)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


# ── character n-gram TF-IDF ────────────────────────────────────────────────────

def _char_clean(text):
    text = re.sub(r"<.*?>", "", text)
    return text.lower()


class CharNgramVectorizer:
    """TF-IDF sobre character n-grams (ex.: 3-5 chars) do texto minimamente limpo.

    Captura padrões de estilo a nível sublexical: sufixos, pontuação integrada,
    transições, espaçamento.
    """

    def __init__(self, max_features=10000, ngram_range=(3, 5)):
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.vocab        = {}
        self.idf          = None

    def _get_char_ngrams(self, text):
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            ngrams.extend([text[i:i+n] for i in range(len(text) - n + 1)])
        return ngrams

    def fit(self, texts):
        df_counter = Counter()
        for text in texts:
            t = _char_clean(text)
            df_counter.update(set(self._get_char_ngrams(t)))
        most_common = df_counter.most_common(self.max_features)
        self.vocab = {ng: i for i, (ng, _) in enumerate(most_common)}
        n_docs = len(texts)
        df = np.array([df_counter[ng] for ng in self.vocab], dtype=np.float32)
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1.0
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            t = _char_clean(text)
            ngrams = self._get_char_ngrams(t)
            tf_counter = Counter(ngrams)
            total = max(len(ngrams), 1)
            for ng, count in tf_counter.items():
                if ng in self.vocab:
                    tf = count / total
                    X[i, self.vocab[ng]] = tf * self.idf[self.vocab[ng]]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + 1e-10)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


# ── vectorizador combinado (TF-IDF palavra + char n-grams) ─────────────────────

STYLE_DIM = 0

class CombinedVectorizer:
    """TF-IDF de palavras (1,2)-grams + TF-IDF de character (3,5)-grams.

    Os char n-grams capturam padrões de estilo sub-lexical sem o bias
    das features estilométricas numéricas.
    """

    def __init__(self, max_words=15000, ngram_range=(1, 2),
                 max_chars=10000, char_range=(3, 5)):
        self.tfidf    = TFIDFVectorizer(max_words=max_words, ngram_range=ngram_range)
        self.char_vec = CharNgramVectorizer(max_features=max_chars, ngram_range=char_range)
        self.word_index = {}

    def fit_transform(self, raw_texts):
        clean = [clean_text(t) for t in raw_texts]
        X_word = self.tfidf.fit_transform(clean)
        self.word_index = self.tfidf.word_index
        X_char = self.char_vec.fit_transform(raw_texts)
        return np.hstack([X_word, X_char])

    def transform(self, raw_texts):
        clean = [clean_text(t) for t in raw_texts]
        X_word = self.tfidf.transform(clean)
        X_char = self.char_vec.transform(raw_texts)
        return np.hstack([X_word, X_char])


# ── labels ─────────────────────────────────────────────────────────────────────

CLASS_NAMES = ['google',
               'anthropic',
               'meta',
               'openai',
               'human']

def encode_labels(labels, class_names=CLASS_NAMES):
    label2idx = {c: i for i, c in enumerate(class_names)}
    return np.array([label2idx[l.lower()] for l in labels], dtype=np.int32)

def labels_to_onehot(labels_idx, n_classes=5):
    return np.eye(n_classes, dtype=np.float32)[labels_idx]

def decode_labels(indices, class_names=CLASS_NAMES):
    return [class_names[i] for i in indices]