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

    def __init__(self, max_words=10000):
        self.max_words = max_words
        self.word_index = {}
        self.idf = None

    def fit(self, texts):
        df_counter = Counter()
        for text in texts:
            df_counter.update(set(text.split()))

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
            tf_counter = Counter(tokens)
            for word, count in tf_counter.items():
                if word in self.word_index:
                    tf = count / (len(tokens) + 1e-10)
                    j = self.word_index[word]
                    X[i, j] = tf * self.idf[j]
        # normalização L2 por linha
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + 1e-10)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


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