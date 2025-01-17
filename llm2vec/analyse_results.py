"""In this file we compute the cosine similarity of the embedding previously
obtained and store in a json file.
We then plot the cosine similarity matrices."""

import json
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def compute_cosine_similarities_numpy(
    list1: List[List[float]], list2: List[List[float]]
) -> np.ndarray:
    """
    Compute cosine similarity between two lists of vectors using NumPy.

    Args:
        list1: First list of vectors (serialized tensors)
        list2: Second list of vectors (serialized tensors)

    Returns:
        np.ndarray: Matrix of cosine similarities
    """
    # Convert to numpy arrays
    arr1 = np.array(list1, dtype=np.float32)
    arr2 = np.array(list2, dtype=np.float32)

    norm1 = np.linalg.norm(arr1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(arr2, axis=1, keepdims=True)
    arr1_normalized = arr1 / norm1
    arr2_normalized = arr2 / norm2

    # Compute similarity matrix
    return np.dot(arr1_normalized, arr2_normalized.T)


if __name__ == "__main__":
    with open("output/llm2vec/embeddings.json", encoding="utf-8") as f:
        results = json.load(f)

    orig_q_reps = np.array(results["Llama_vanilla_queries"])
    orig_d_reps = np.array(results["Llama_vanilla_documents"])

    # LLM2Vec model
    l2v_q_reps = np.array(results["Llama_llm2vec_queries"])
    l2v_d_reps = np.array(results["Llama_llm2vec_documents"])

    sbert_q_reps = np.array(results["SBERT_queries"])
    sbert_d_reps = np.array(results["SBERT_documents"])

    original_model_similarities = compute_cosine_similarities_numpy(
        orig_q_reps, orig_d_reps
    )
    llm2vec_model_similarities = compute_cosine_similarities_numpy(
        l2v_q_reps, l2v_d_reps
    )
    sBERT_model_similarities = compute_cosine_similarities_numpy(
        sbert_q_reps, sbert_d_reps
    )
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    vmin = min(
        original_model_similarities.min().min(),
        llm2vec_model_similarities.min().min(),
        sBERT_model_similarities.min().min(),
    )
    vmax = max(
        original_model_similarities.max().max(),
        llm2vec_model_similarities.max().max(),
        sBERT_model_similarities.max().max(),
    )

    sns.heatmap(
        original_model_similarities,
        ax=ax1,
        cmap="rocket",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Similarity"},
    )
    sns.heatmap(
        llm2vec_model_similarities,
        ax=ax2,
        cmap="rocket",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Similarity"},
    )
    sns.heatmap(
        sBERT_model_similarities,
        ax=ax3,
        cmap="rocket",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Similarity"},
    )

    ax1.set_title("Original Model Similarities", pad=10)
    ax2.set_title("LLM2Vec Model Similarities", pad=10)
    ax3.set_title("SBERT Model Similarities", pad=10)

    plt.tight_layout()
    plt.show()
