import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from llm2vec import LLM2Vec
from sentence_transformers import SentenceTransformer
from mteb import MTEB
import numpy as np
import json
from tqdm import tqdm


# Keep your existing model IDs and CustomLLM2Vec class
ORIGINAL_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLM2VEC_MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
LLM2VEC_LORA_MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"
SBERT_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"  # One of the best performing SBERT models

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
        torch_dtype=torch.float16,
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
        torch_dtype=torch.float16,
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


class SentenceTransformerWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = 512  # You can adjust this
        
    def encode(self, sentences, batch_size=32, 
               convert_to_numpy=True, 
               device=None, normalize_embeddings=False, **kwargs):
        """
        Encode sentences to embeddings
        """
        # Handle single sentence
        if isinstance(sentences, str):
            sentences = [sentences]
            
        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            # Move to device if specified
            if device:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                self.model = self.model.to(device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True
                )
                
                # Get the last hidden states
                if hasattr(outputs, 'hidden_states'):
                    last_hidden = outputs.hidden_states[-1]
                else:
                    last_hidden = outputs.last_hidden_state
                    
                # Mean pooling
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                sentence_embeddings = torch.sum(last_hidden * attention_mask, 1) / torch.sum(attention_mask, 1)
                
                if normalize_embeddings:
                    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                
                if convert_to_numpy:
                    sentence_embeddings = sentence_embeddings.cpu().numpy()
                    
                embeddings.extend(sentence_embeddings)
                
        if convert_to_numpy:
            return np.array(embeddings)
        return torch.stack(embeddings)

    def get_sentence_embedding_dimension(self):
        """Return the dimension of the sentence embeddings"""
        return self.model.config.hidden_size

    def get_max_seq_length(self):
        """Return the maximum sequence length"""
        return self.max_seq_length

def evaluate_model(model_wrapper, model_name):
    """Run MTEB evaluation"""
    
    evaluation = MTEB(tasks=["ArXivHierarchicalClusteringS2S"])
    #evaluation = MTEB(tasks=["EmotionClassification"])
    evaluation = MTEB(tasks=["TwentyNewsgroupsClustering"])

    #breakpoint()
    results = evaluation.run(model_wrapper,  output_folder=f"results/{model_name}")
    return results



def benchmark_models():
    # Load all models
    print("Loading models...")
    
    # Original Llama
    original_model, original_tokenizer = load_original_model()
    original_wrapper = SentenceTransformerWrapper(original_model, original_tokenizer)
    
    # LLM2Vec
    llm2vec_model, llm2vec_tokenizer = load_llm2vec_model()
    llm2vec_wrapper = SentenceTransformerWrapper(llm2vec_model, llm2vec_tokenizer)
    
    # SBERT
    sbert_model = load_sbert_model()

    # Run evaluations
    print("Running MTEB evaluations...")
    
    models = {
        "sbert": sbert_model,
        "llama_vanilla": original_wrapper,
        "llama_llm2vec": llm2vec_wrapper,
    }
    
    results = {}
    for model_name, wrapper in models.items():
        print(f"\nEvaluating {model_name}...")
        results[model_name] = evaluate_model(wrapper, model_name)
    
    # Save results
    with open('llm2vec/mteb_results.json', 'w+', encoding = "utf-8") as f:
        json.dump(results, f)
    
    return results

if __name__ == "__main__":
    results = benchmark_models()
    # Print summary metrics
    for model_name, model_results in results.items():
        print(f"\nResults for {model_name}:")
        for task_type, scores in model_results.items():
            if isinstance(scores, dict):
                avg_score = np.mean([v for v in scores.values() if isinstance(v, (int, float))])
                print(f"{task_type}: {avg_score:.2f}")