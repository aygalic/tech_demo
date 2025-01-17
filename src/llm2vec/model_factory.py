
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


ORIGINAL_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLM2VEC_MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
LLM2VEC_LORA_MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"
SBERT_MODEL_ID = ("sentence-transformers/all-mpnet-base-v2")

    

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
