from lm_polygraph.stat_calculators.internal_confidence_probs import (
    InternalConfidenceProbsCalculator,
)
from lm_polygraph.stat_calculators.internal_confidence_probs import get_yes_no_token_ids
import logging

log = logging.getLogger("lm_polygraph")


def load_stat_calculator(config, builder):
    model = builder.model
    if model is None:
        raise ValueError("Model is required to build InternalConfidenceProbsCalculator")

    try:
        yes_token_ids, no_token_ids = get_yes_no_token_ids(
            model.tokenizer,
            yes_token=config.get("yes_token", "Yes"),
            no_token=config.get("no_token", "No"),
        )
    except Exception as e:
        log.warning(
            f"Failed to resolve Yes/No token IDs automatically: {e}. Falling back to default Llama-3 tokens."
        )
        yes_token_ids = [7566]
        no_token_ids = [2360]

    return InternalConfidenceProbsCalculator(
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
        last_k_tokens=config.get("last_k_tokens", 6),
        prompt_template=config.get(
            "prompt_template",
            "Respond only with Yes or No to indicate whether you are capable of answering the following question. {q}",
        ),
    )
