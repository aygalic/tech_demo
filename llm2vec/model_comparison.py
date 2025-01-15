from llm2vec import LLM2Vec
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from peft import PeftModel

ORIGINAL_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLM2VEC_MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
LLM2VEC_LORA_MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"

class CustomLLM2Vec(LLM2Vec):
    """Custom LLM2Vec class that handles CausalLM outputs"""
    def forward(self, sentence_feature):
        outputs = self.model(
            input_ids=sentence_feature["input_ids"],
            attention_mask=sentence_feature["attention_mask"],
            output_hidden_states=True  # Make sure to get hidden states
        )
        # For CausalLM models, use hidden_states instead of last_hidden_state
        if hasattr(outputs, 'hidden_states'):
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
        output_hidden_states=True  # Make sure this is enabled
    )
    return model, tokenizer

def load_llm2vec_model():
    """Load the LLM2Vec-enhanced model"""
    tokenizer = AutoTokenizer.from_pretrained(
        LLM2VEC_MODEL_ID
    )
    config = AutoConfig.from_pretrained(
        LLM2VEC_MODEL_ID, 
        trust_remote_code=True
    )
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
    model = PeftModel.from_pretrained(
        model, 
        LLM2VEC_LORA_MODEL_ID
    )
    return model, tokenizer

def compare_models():
    # Load both models
    original_model, original_tokenizer = load_original_model()
    llm2vec_model, llm2vec_tokenizer = load_llm2vec_model()
    
    # Create wrappers using CustomLLM2Vec
    original_wrapper = CustomLLM2Vec(original_model, original_tokenizer, pooling_mode="mean", max_length=512)
    llm2vec_wrapper = CustomLLM2Vec(llm2vec_model, llm2vec_tokenizer, pooling_mode="mean", max_length=512)
    
    # Test data
    instruction = "Given a web search query, retrieve relevant passages that answer the query:"
    queries = [
        [instruction, "how much protein should a female eat"],
        [instruction, "summit define"],
        [instruction, "What is a cat"],
    ]
    
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
        "A cat is feline",
        "A cat is just like a dog"
    ]
    
    # Get embeddings from both models
    print("Computing embeddings...")
    
    # Original model
    breakpoint()
    orig_q_reps = original_wrapper.encode(queries)
    orig_d_reps = original_wrapper.encode(documents)
    
    # LLM2Vec model
    l2v_q_reps = llm2vec_wrapper.encode(queries)
    l2v_d_reps = llm2vec_wrapper.encode(documents)
    
    # Compute similarities for both models
    def compute_similarities(q_reps, d_reps):
        q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
        return torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))
    
    print("\nOriginal Model Similarities:")
    print(compute_similarities(orig_q_reps, orig_d_reps))
    
    print("\nLLM2Vec Model Similarities:")
    print(compute_similarities(l2v_q_reps, l2v_d_reps))

if __name__ == "__main__":
    compare_models()