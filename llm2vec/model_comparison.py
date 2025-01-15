import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from llm2vec import LLM2Vec

from ressources import queries, poetry_documents, science_documents
import json



ORIGINAL_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLM2VEC_MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
LLM2VEC_LORA_MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"


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


def compare_models():
    # Load both models
    original_model, original_tokenizer = load_original_model()
    llm2vec_model, llm2vec_tokenizer = load_llm2vec_model()

    # Create wrappers using CustomLLM2Vec
    original_wrapper = CustomLLM2Vec(
        original_model, original_tokenizer, pooling_mode="mean", max_length=512
    )
    llm2vec_wrapper = CustomLLM2Vec(
        llm2vec_model, llm2vec_tokenizer, pooling_mode="mean", max_length=512
    )

    # Get embeddings from both models
    print("Computing embeddings...")

    # Original model
    orig_q_reps = original_wrapper.encode(queries)
    orig_d_reps = original_wrapper.encode(poetry_documents[:1] + science_documents[:1])

    # LLM2Vec model
    l2v_q_reps = llm2vec_wrapper.encode(queries)
    l2v_d_reps = llm2vec_wrapper.encode(poetry_documents[:1] + science_documents[:1])

    results = {
        "Llama_vanilla_queries" : orig_q_reps.cpu().numpy().tolist(),
        "Llama_vanilla_documents" : orig_d_reps.cpu().numpy().tolist(),
        "Llama_llm2vec_queries" : l2v_q_reps.cpu().numpy().tolist(),
        "Llama_llm2vec_documents" : l2v_d_reps.cpu().numpy().tolist(),
    }

    with open('output.json', 'w+', encoding ='utf8') as json_file:
        json.dump(results, json_file)

if __name__ == "__main__":
    compare_models()
