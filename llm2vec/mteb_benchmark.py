"""Proof of concept on the feasability of an LLM2Vec -> MTEB pipeline"""

from mteb import MTEB
import numpy as np
import json

from src.llm2vec.model_factory import load_original_model, load_sbert_model, load_llm2vec_model
from src.llm2vec.sentence_transformer_wrapper import SentenceTransformerWrapper


def evaluate_model(model, model_name: str, tasks: list[str] | None = None) -> list:
    """Run MTEB evaluation

    Parameters
    ----------
    model : _type_ #FIXME
        Model to evaluate
    model_name : str
        Name of the model
    tasks : list[str], optional
        Benchmark subtask, if None is provided, the name of all tasks will be
        output in the console, by default None

    Returns
    -------
    list[TaskResult]
        Benchmark results for the provided model
    """
    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(model,  output_folder=f"results/{model_name}")
    return results

def benchmark_models() -> dict[str: list]:
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
    tasks = ["TwentyNewsgroupsClustering"]
    models = {
        "sbert": sbert_model,
        "llama_vanilla": original_wrapper,
        "llama_llm2vec": llm2vec_wrapper,
    }
    
    results = {}
    for model_name, wrapper in models.items():
        print(f"\nEvaluating {model_name}...")
        results[model_name] = evaluate_model(wrapper, model_name, tasks)
    
    # Save results
    with open('llm2vec/mteb_results.json', 'w+', encoding = "utf-8") as f:
        json.dump(results, f)
    
    return results

if __name__ == "__main__":
    bechmark_results = benchmark_models()
    # Print summary metrics
    for model_name, model_results in bechmark_results.items():
        print(f"\nResults for {model_name}:")
        for task_type, scores in model_results.items():
            if isinstance(scores, dict):
                avg_score = np.mean([v for v in scores.values() if isinstance(v, (int, float))])
                print(f"{task_type}: {avg_score:.2f}")