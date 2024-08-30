import json
import torch
from copy import deepcopy
from model_utils import get_model
from eval_utils.prompt_utils import get_prompt_template
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class SafetyLock(
    nn.Module,
    PyTorchModelHubMixin,
    # optionally, you can add metadata which gets pushed to the model card
    repo_url="WestlakeNLP/SafetyLock_llama_3_8B",
    license="mit",
):
    def __init__(self, activate):
        super().__init__()
        self.param = activate
    def forward(self, x):
        return x



def create_meta_safetylock(model, tokenizer, prompt_template_style, dataset_path='', num_heads_to_intervene=24,last_token_for_get_activate=5):
    """
    Create Meta-SafetyLock from the original model.

    :param model: The original language model
    :param tokenizer: Tokenizer for processing text
    :param prompt_template_style: Style of the prompt template
    :param use_safe_dataset: Whether to use the safe dataset
    :param num_heads_to_intervene: Number of attention heads to intervene
    :return: Meta-SafetyLock (activation values)
    """


    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)[:100]  # Use only the first 100 samples

    # Get or compute activations for all heads

    all_head_activations = model.preprocess_activate_dataset(dataset, get_prompt_template(prompt_template_style),last_token_for_get_activate=last_token_for_get_activate)

    # Prepare data for intervention
    labels = []
    head_activations_for_intervention = []

    for i, activation in enumerate(all_head_activations):
        if i % (last_token_for_get_activate*2) < last_token_for_get_activate:
            labels.append(1)  # Safe sample
        else:
            labels.append(0)  # Unsafe sample
        head_activations_for_intervention.append(deepcopy(activation))

    # Get Meta-SafetyLock (activation values)
    meta_safetylock = model.get_activations(
        deepcopy(head_activations_for_intervention),
        labels,
        num_to_intervene=num_heads_to_intervene
    )

    return meta_safetylock

if __name__ == '__main__':
    # Usage example
    from transformers import LlamaForCausalLM
    model,tokenizer = get_model('/zhuminjun/model/ai-modelscope/mistral-nemo-instruct-2407',use_bit_4=True)
    meta_safetylock = create_meta_safetylock(model, tokenizer, dataset_path='../data/safe_dataset.json',prompt_template_style='none',num_heads_to_intervene=24)
    torch.save(meta_safetylock,
               '/zhuminjun/PAlign/LLMs-Finetuning-Safety-main/llama2/CODE/Meta_safetylock_weight/Mistral_Nemo_12B.safelock')