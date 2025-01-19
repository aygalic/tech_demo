"""Sentense-transformer wrapper"""

from typing import Any, Literal

import numpy as np
import torch
from tqdm import tqdm


class SentenceTransformerWrapper:
    """Wrapper mimicking the sentense-transformers class from hugging face.

    Parameters
    ----------
    model : torch.nn.Module
        Model to wrap
    tokenizer : Any
        Associated tokenizer
    """

    def __init__(self, model: torch.nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = 512  # You can adjust this

    def encode(
        self,
        sentences: Any, #FIXME
        batch_size: int=32,
        convert_to_numpy: bool = True,
        device: Literal["mps", "cpu", "cuda", None] = None,
        normalize_embeddings: bool=False,
        **kwargs,
    ):
        """
        Encode sentences to embeddings
        """
        # Handle single sentence
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i : i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )

            # Move to device if specified
            if device:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                self.model = self.model.to(device)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

                # Get the last hidden states
                if hasattr(outputs, "hidden_states"):
                    last_hidden = outputs.hidden_states[-1]
                else:
                    last_hidden = outputs.last_hidden_state

                # Mean pooling
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                sentence_embeddings = torch.sum(
                    last_hidden * attention_mask, 1
                ) / torch.sum(attention_mask, 1)

                if normalize_embeddings:
                    sentence_embeddings = torch.nn.functional.normalize(
                        sentence_embeddings, p=2, dim=1
                    )

                if convert_to_numpy:
                    sentence_embeddings = sentence_embeddings.cpu().numpy()

                embeddings.extend(sentence_embeddings)

        if convert_to_numpy:
            return np.array(embeddings)
        return torch.stack(embeddings)

    def get_sentence_embedding_dimension(self) -> int:
        """Return the dimension of the sentence embeddings"""
        return self.model.config.hidden_size

    def get_max_seq_length(self) -> int:
        """Return the maximum sequence length"""
        return self.max_seq_length
