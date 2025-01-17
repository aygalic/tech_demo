import json

import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from llm2vec import LLM2Vec

ORIGINAL_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLM2VEC_MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
LLM2VEC_LORA_MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"
SBERT_MODEL_ID = (
    "sentence-transformers/all-mpnet-base-v2"  # One of the best performing SBERT models
)


class CustomLLM2Vec(LLM2Vec):
    """Custom LLM2Vec class that handles CausalLM outputs"""

    def forward(self, sentence_feature):
        outputs = self.model(
            input_ids=sentence_feature["input_ids"],
            attention_mask=sentence_feature["attention_mask"],
            output_hidden_states=True,  # Make sure to get hidden states
        )
        # For CausalLM models, use hidden_states instead of last_hidden_state
        if hasattr(outputs, "hidden_states"):
            last_hidden = outputs.hidden_states[-1]
        else:
            last_hidden = outputs.last_hidden_state

        return self.get_pooling(sentence_feature, last_hidden)


def load_original_model():
    """Load the original Llama model"""
    tokenizer = AutoTokenizer.from_pretrained(LLM2VEC_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        output_hidden_states=True,  # Make sure this is enabled
    )
    return model, tokenizer


def load_llm2vec_model():
    """Load the LLM2Vec-enhanced model"""
    tokenizer = AutoTokenizer.from_pretrained(LLM2VEC_MODEL_ID)
    config = AutoConfig.from_pretrained(LLM2VEC_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        LLM2VEC_MODEL_ID,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    # Load MNTP weights
    model = PeftModel.from_pretrained(
        model,
        LLM2VEC_MODEL_ID,
    )
    model = model.merge_and_unload()

    # Load supervised weights
    model = PeftModel.from_pretrained(model, LLM2VEC_LORA_MODEL_ID)
    return model, tokenizer


def load_sbert_model():
    """Load SentenceBERT model"""
    return SentenceTransformer(SBERT_MODEL_ID)


def prepare_sbert_input(queries):
    """Prepare input for SBERT (removes instruction prefix)"""
    # SBERT doesn't use instructions, so we'll just use the queries
    return [q[1] if isinstance(q, list) else q for q in queries]


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
    breakpoint()

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
    sbert_queries = prepare_sbert_input(queries)
    sbert_q_reps = torch.tensor(sbert_model.encode(sbert_queries))
    sbert_d_reps = torch.tensor(sbert_model.encode(selected_docs))

    results = {
        "Llama_vanilla_queries": orig_q_reps.cpu().numpy().tolist(),
        "Llama_vanilla_documents": orig_d_reps.cpu().numpy().tolist(),
        "Llama_llm2vec_queries": l2v_q_reps.cpu().numpy().tolist(),
        "Llama_llm2vec_documents": l2v_d_reps.cpu().numpy().tolist(),
        "SBERT_queries": sbert_q_reps.cpu().numpy().tolist(),
        "SBERT_documents": sbert_d_reps.cpu().numpy().tolist(),
    }

    with open("output.json", "w+", encoding="utf8") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    compare_models()
