import json

import torch
from src.llm2vec.model_factory import load_original_model, load_sbert_model, load_llm2vec_model
from src.llm2vec.custom_llm2vec import CustomLLM2Vec

def get_data(
    json_path: str = "assets/llm2vec/ressources.json",
) -> tuple[list[str], list[str], list[str]]:
    """Retrieve experiment data from json file

    Parameters
    ----------
    json_path : str, optional
        Path of the json file to load data from,
        by default 'assets/llm2vec/ressources.json'

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Tuple of list containing queries, poetry documents and science documents
    """

    with open(json_path, encoding="utf-8") as f:
        ressources = json.load(f)

    instruct = ressources["instruction"]
    queries = [instruct + query for query in ressources["queries"]]
    poetry_documents = ressources["documents"]["poetry_documents"]
    science_documents = ressources["documents"]["science_documents"]
    return queries, poetry_documents, science_documents


def compare_models():
    # Load all models
    original_model, original_tokenizer = load_original_model()
    llm2vec_model, llm2vec_tokenizer = load_llm2vec_model()
    sbert_model = load_sbert_model()

    # Create wrappers for LLM models
    original_wrapper = CustomLLM2Vec(
        original_model, original_tokenizer, pooling_mode="mean", max_length=512
    )
    llm2vec_wrapper = CustomLLM2Vec(
        llm2vec_model, llm2vec_tokenizer, pooling_mode="mean", max_length=512
    )

    # Prepare documents

    queries, poetry_documents, science_documents = get_data()

    selected_docs = poetry_documents[:1] + science_documents[:1]
    selected_docs = poetry_documents + science_documents

    print("Computing embeddings...")

    # Original model embeddings
    orig_q_reps = original_wrapper.encode(queries)
    orig_d_reps = original_wrapper.encode(selected_docs)

    # LLM2Vec model embeddings
    l2v_q_reps = llm2vec_wrapper.encode(queries)
    l2v_d_reps = llm2vec_wrapper.encode(selected_docs)

    # SBERT embeddings
    sbert_q_reps = torch.tensor(sbert_model.encode(queries))
    sbert_d_reps = torch.tensor(sbert_model.encode(selected_docs))

    results = {
        "Llama_vanilla_queries": orig_q_reps.cpu().numpy().tolist(),
        "Llama_vanilla_documents": orig_d_reps.cpu().numpy().tolist(),
        "Llama_llm2vec_queries": l2v_q_reps.cpu().numpy().tolist(),
        "Llama_llm2vec_documents": l2v_d_reps.cpu().numpy().tolist(),
        "SBERT_queries": sbert_q_reps.cpu().numpy().tolist(),
        "SBERT_documents": sbert_d_reps.cpu().numpy().tolist(),
    }

    with open("output/llm2vec/embeddings.json", "w+", encoding="utf8") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    compare_models()
