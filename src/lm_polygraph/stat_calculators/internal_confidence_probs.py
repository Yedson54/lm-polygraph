from typing import Dict, List, Tuple

import numpy as np
import torch

from transformers import PreTrainedModel
from lm_polygraph.utils.model import WhiteboxModel

from .stat_calculator import StatCalculator


def compute_positional_attention(
    max_len: int, center_idx: int, w: float = 1.0, s: float = 1.0
) -> np.ndarray:
    r"""
    Computes a delta vector using the formula in the image, given a center index.

    Args:
        max_len (int): Total length of the sequence.
        center_idx (int): The center index i for which to compute delta[i,j] across j.
        w (float): Locality control parameter (w > 0). Higher values concentrate weight more at the center.
        s (float): Symmetry control parameter. s=1 is symmetric; s>1 decays faster to the right of center.

    Returns:
        np.ndarray: A 1D array of length `max_len` representing $\delta_i$ over $j$.
    """
    deltas = np.zeros(max_len)
    for j in range(max_len):
        l2 = (center_idx - j) ** 2
        alpha = -s * w * l2 if center_idx <= j else -w * l2
        deltas[j] = np.exp(alpha)
    return deltas / deltas.sum()


def get_yes_no_token_ids(
    tokenizer, yes_token: str = "Yes", no_token: str = "No"
) -> Tuple[List[int], List[int]]:
    """
    Helper to get token IDs for 'Yes' and 'No'.

    Returns:
        tuple: A tuple of two lists, each containing the token ID for 'Yes' and 'No'.
    Raises:
        ValueError: If 'Yes' or 'No' tokenize to multiple tokens.
    """
    yes_ids = tokenizer.encode(yes_token, add_special_tokens=False)
    no_ids = tokenizer.encode(no_token, add_special_tokens=False)
    if len(yes_ids) != 1:
        raise ValueError(f"Token '{yes_token}' tokenizes to multiple tokens: {yes_ids}")
    if len(no_ids) != 1:
        raise ValueError(f"Token '{no_token}' tokenizes to multiple tokens: {no_ids}")
    return [yes_ids[0]], [no_ids[0]]


def _get_lm_head_weight(model: PreTrainedModel) -> torch.Tensor:
    """
    Attempts to get the unembedding matrix weight from the model.
    """
    if hasattr(model, "lm_head"):
        return model.lm_head.weight
    elif hasattr(model, "embed_out"):
        return model.embed_out.weight
    else:
        raise ValueError("Could not find lm_head or embed_out in the model")


class InternalConfidenceProbsCalculator(StatCalculator):
    """
    Calculates the binary-renormalized probability of 'Yes' over 'No' for the last K tokens
    across all layers of the model, using a self-evaluation prompt.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return ["ic_pyes_matrix"], ["input_texts"]

    def __init__(
        self,
        yes_token_ids: List[int],
        no_token_ids: List[int],
        last_k_tokens: int = 6,
        prompt_template: str = "Respond only with Yes or No to indicate whether you are capable of answering the following question. {q}",
    ):
        """
        Parameters:
            yes_token_ids (List[int]): Token ID(s) for 'Yes'. Must be a single token ID in the list.
            no_token_ids (List[int]): Token ID(s) for 'No'. Must be a single token ID in the list.
            last_k_tokens (int): Number of suffix tokens to consider for the IC matrix.
            prompt_template (str): The prompt template to use. Must contain '{q}' for the question.
        """
        super().__init__()
        self.yes_token_ids = yes_token_ids
        self.no_token_ids = no_token_ids
        self.last_k_tokens = last_k_tokens
        self.prompt_template = prompt_template

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the internal confidence probabilities.
        Adapated from Chen et al. (2026) [https://arxiv.org/pdf/2506.09669](https://arxiv.org/pdf/2506.09669)
        Code: [https://github.com/tigerchen52/query_level_uncertainty](https://github.com/tigerchen52/query_level_uncertainty)

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics (not used here).
            texts (List[str]): Input texts batch.
            model (WhiteboxModel): Model used for extraction.
            max_new_tokens (int): Not used.
        Returns:
            Dict[str, np.ndarray]: dictionary with 'ic_pyes_matrix' (List of [K, L] arrays).
        """
        input_texts = dependencies.get("input_texts", texts)

        # Format prompts
        prompts = [self.prompt_template.format(q=text) for text in input_texts]

        # Tokenize (using model.tokenize which handles chat templates if instruct=True)
        batch = model.tokenize(prompts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}

        # Get unembedding weights for Yes and No
        lm_head_weight = _get_lm_head_weight(model.model)
        W_yes = lm_head_weight[self.yes_token_ids[0]]
        W_no = lm_head_weight[self.no_token_ids[0]]

        ic_pyes_matrices = []

        with torch.no_grad():
            # Run forward pass to get hidden states
            out = model.model(**batch, output_hidden_states=True)

            # out.hidden_states is a tuple of (embedding, layer_1, ..., layer_L)
            # We skip layer 0 (embedding) and use 1..L
            hidden_states = out.hidden_states[1:]
            L = len(hidden_states)

            # Each hidden state is [batch_size, seq_len, hidden_size]
            # We want to stack them to [batch_size, seq_len, L, hidden_size]
            # But we only need the last K tokens

            for b in range(len(prompts)):
                # Handle varying sequence lengths if padding is used, but batch tokenize pads to max length
                # We need to find the actual sequence length for this batch item
                if "attention_mask" in batch:
                    seq_len = batch["attention_mask"][b].sum().item()
                else:
                    seq_len = batch["input_ids"].shape[1]

                # Get the last K tokens for this sequence
                start_idx = max(0, seq_len - self.last_k_tokens)
                K_actual = seq_len - start_idx

                # [K_actual, L]
                p_yes_matrix = np.zeros((K_actual, L), dtype=np.float32)

                for layer_idx in range(L):
                    # [K_actual, hidden_size]
                    h = hidden_states[layer_idx][b, start_idx:seq_len, :]

                    # [K_actual]
                    logit_yes = torch.matmul(h, W_yes)
                    logit_no = torch.matmul(h, W_no)

                    # Softmax over Yes/No
                    # p_yes = exp(logit_yes) / (exp(logit_yes) + exp(logit_no))
                    # Numerically stable:
                    max_logit = torch.max(logit_yes, logit_no)
                    exp_yes = torch.exp(logit_yes - max_logit)
                    exp_no = torch.exp(logit_no - max_logit)
                    p_yes = exp_yes / (exp_yes + exp_no)

                    p_yes_matrix[:, layer_idx] = p_yes.cpu().numpy()

                ic_pyes_matrices.append(p_yes_matrix)

        return {"ic_pyes_matrix": ic_pyes_matrices}
