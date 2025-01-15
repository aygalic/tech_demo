import json
import numpy as np
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt

def compute_cosine_similarities_numpy(list1: List[List[float]], 
                                    list2: List[List[float]]) -> np.ndarray:
    """
    Compute cosine similarity between two lists of vectors using NumPy.
    Optimized for M1 Mac using built-in Accelerate framework.
    
    Args:
        list1: First list of vectors (serialized tensors)
        list2: Second list of vectors (serialized tensors)
        
    Returns:
        np.ndarray: Matrix of cosine similarities
    """
    # Convert to numpy arrays
    arr1 = np.array(list1, dtype=np.float32)
    arr2 = np.array(list2, dtype=np.float32)
    
    # Compute norms
    norm1 = np.linalg.norm(arr1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(arr2, axis=1, keepdims=True)
    
    # Normalize
    arr1_normalized = arr1 / norm1
    arr2_normalized = arr2 / norm2
    
    # Compute similarity matrix
    return np.dot(arr1_normalized, arr2_normalized.T)

# Example usage
if __name__ == "__main__":
    # Example data

    results = None

    with open('output.json', encoding='utf-8') as f:
        results = json.load(f)

    orig_q_reps = np.array(results["Llama_vanilla_queries"])
    orig_d_reps = np.array(results["Llama_vanilla_documents"])
    
    # LLM2Vec model
    l2v_q_reps = np.array(results["Llama_llm2vec_queries"])
    l2v_d_reps = np.array(results["Llama_llm2vec_documents"])


    sbert_q_reps = np.array(results["SBERT_queries"])
    sbert_d_reps = np.array(results["SBERT_documents"])


    # similarities from original model
    print("\nOriginal Model Similarities:")
    original_model_similarities = compute_cosine_similarities_numpy(orig_q_reps, orig_d_reps)
    print(original_model_similarities)

    print("\nLLM2Vec Model Similarities:")
    llm2vec_model_similarities = compute_cosine_similarities_numpy(l2v_q_reps, l2v_d_reps)
    print(llm2vec_model_similarities)

    print("\nsBERT Model Similarities:")
    sBERT_model_similarities = compute_cosine_similarities_numpy(sbert_q_reps, sbert_d_reps)
    print(sBERT_model_similarities)

    sns.heatmap(original_model_similarities)
    plt.show()
    sns.heatmap(llm2vec_model_similarities)*
    plt.show()
    sns.heatmap(sBERT_model_similarities)
    plt.show()