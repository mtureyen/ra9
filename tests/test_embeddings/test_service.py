"""Tests for the embedding service."""

import math


def test_embed_text_returns_1024_dim_vector(embedding_service):
    vec = embedding_service.embed_text("hello world")
    assert len(vec) == 1024
    assert all(isinstance(v, float) for v in vec)


def test_embed_text_is_normalized(embedding_service):
    vec = embedding_service.embed_text("test input")
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 0.01


def test_batch_embed_returns_correct_count(embedding_service):
    vecs = embedding_service.batch_embed(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == 1024 for v in vecs)


def test_batch_embed_empty_list(embedding_service):
    assert embedding_service.batch_embed([]) == []


def test_similar_texts_have_high_cosine(embedding_service):
    v1 = embedding_service.embed_text("the cat sat on the mat")
    v2 = embedding_service.embed_text("a cat was sitting on a mat")
    cosine = sum(a * b for a, b in zip(v1, v2))
    assert cosine > 0.8


def test_dissimilar_texts_have_low_cosine(embedding_service):
    v1 = embedding_service.embed_text("quantum physics equations")
    v2 = embedding_service.embed_text("chocolate cake recipe")
    cosine = sum(a * b for a, b in zip(v1, v2))
    assert cosine < 0.5


def test_dim_property(embedding_service):
    assert embedding_service.dim == 1024
