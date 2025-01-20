"""Wrapper for LLM2Vec, used to provide support for MTEB embeddings"""
from llm2vec import LLM2Vec

class CustomLLM2Vec(LLM2Vec):
    """Custom LLM2Vec class that handles CausalLM outputs"""

    def forward(self, sentence_feature: dict):
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
