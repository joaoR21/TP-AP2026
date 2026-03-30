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

# ── features estilométricas ────────────────────────────────────────────────────

def _stylometric_single(text):
    """Extrai ~20 features estilométricas numéricas de um texto."""
    words = text.split()
    n_chars = len(text)
    n_words = max(len(words), 1)
    n_sents = max(text.count('.') + text.count('!') + text.count('?'), 1)

    unique_words = set(w.lower() for w in words)
    ttr = len(unique_words) / n_words                     # type-token ratio

    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    std_word_len = np.std([len(w) for w in words]) if words else 0
    avg_sent_len = n_words / n_sents

    # rácios de caracteres
    n_alpha = sum(c.isalpha() for c in text)
    n_digit = sum(c.isdigit() for c in text)
    n_upper = sum(c.isupper() for c in text)
    n_space = sum(c.isspace() for c in text)

    # pontuação específica
    n_comma = text.count(',')
    n_semicolon = text.count(';')
    n_colon = text.count(':')
    n_paren = text.count('(') + text.count(')')
    n_dash = text.count('-') + text.count('–') + text.count('—')
    n_quote = text.count('"') + text.count("'") + text.count('"') + text.count('"')

    # caracteres especiais / unicode (típicos de certos LLMs)
    n_special = sum(1 for c in text if not c.isascii())

    feats = [
        n_chars,                            # 0  comprimento total
        n_words,                            # 1  num palavras
        avg_word_len,                       # 2  comp médio palavra
        std_word_len,                       # 3  std comp palavra
        avg_sent_len,                       # 4  comp médio frase (em palavras)
        ttr,                                # 5  type-token ratio
        n_comma / n_words,                  # 6  vírgulas por palavra
        n_semicolon / n_words,              # 7  ponto-vírgulas por palavra
        n_colon / n_words,                  # 8  dois-pontos por palavra
        n_paren / n_words,                  # 9  parênteses por palavra
        n_dash / n_words,                   # 10 travessões por palavra
        n_quote / n_words,                  # 11 aspas por palavra
        n_upper / max(n_alpha, 1),          # 12 ratio maiúsculas
        n_digit / max(n_chars, 1),          # 13 ratio dígitos
        n_space / max(n_chars, 1),          # 14 ratio espaços
        n_special / max(n_chars, 1),        # 15 ratio caracteres não-ASCII
        n_sents,                            # 16 num frases
        len(unique_words) / max(n_sents, 1),# 17 palavras únicas por frase
    ]
    return np.array(feats, dtype=np.float32)


def extract_stylometric_features(texts):
    """Extrai features estilométricas para uma lista de textos. Retorna array normalizado."""
    X = np.array([_stylometric_single(t) for t in texts])
    # z-score normalização por coluna
    means = X.mean(axis=0, keepdims=True)
    stds  = X.std(axis=0, keepdims=True) + 1e-10
    return (X - means) / stds, means, stds


def apply_stylometric_features(texts, means, stds):
    """Aplica features estilométricas com estatísticas de treino."""
    X = np.array([_stylometric_single(t) for t in texts])
    return (X - means) / stds


# ── vectorizador combinado v2 (TF-IDF + char n-grams + estilometria) ──────────

class CombinedVectorizerV2:
    """TF-IDF palavras + char n-grams + features estilométricas."""

    def __init__(self, max_words=15000, ngram_range=(1, 2),
                 max_chars=10000, char_range=(3, 5)):
        self.tfidf    = TFIDFVectorizer(max_words=max_words, ngram_range=ngram_range)
        self.char_vec = CharNgramVectorizer(max_features=max_chars, ngram_range=char_range)
        self.word_index = {}
        self.style_means = None
        self.style_stds  = None

    def fit_transform(self, raw_texts):
        clean = [clean_text(t) for t in raw_texts]
        X_word = self.tfidf.fit_transform(clean)
        self.word_index = self.tfidf.word_index
        X_char = self.char_vec.fit_transform(raw_texts)
        X_style, self.style_means, self.style_stds = extract_stylometric_features(raw_texts)
        return np.hstack([X_word, X_char, X_style])

    def transform(self, raw_texts):
        clean = [clean_text(t) for t in raw_texts]
        X_word = self.tfidf.transform(clean)
        X_char = self.char_vec.transform(raw_texts)
        X_style = apply_stylometric_features(raw_texts, self.style_means, self.style_stds)
        return np.hstack([X_word, X_char, X_style])

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