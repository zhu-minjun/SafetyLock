import os
import sys
import time
import csv
import torch
import json
from tqdm import trange
from typing import List
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
# Import custom modules (assuming they are in the current directory)
from llama_iti import get_model
from eval_utils.model_utils import load_peft_model
from eval_utils.prompt_utils import apply_prompt_template, get_prompt_template
from eval_utils.ppl_utils import PPL_Calculator
from eval_utils.bpe import load_subword_nmt_table, BpeOnlineTokenizer


def question_read(text_file):
    """Read questions from a CSV file."""
    dataset = []
    with open(text_file, "r") as file:
        data = list(csv.reader(file, delimiter=","))
    for row in data:
        dataset.append(row[0])
    return dataset


def set_model(model, prompt_template_style, use_safe_dataset=True, alpha=4, K=24, model_name=None):
    """Set up the model with safety interventions."""
    # Load dataset
    if use_safe_dataset:
        dataset = json.load(open('../safe_dataset.json', encoding='utf-8'))[:100]
    else:
        dataset = json.load(open('../hhrlhf_dataset.json', encoding='utf-8'))

    # Load or compute activations
    try:
        all_head_wise_activations = torch.load('../Meta_safetylock_weight/Meta-Llama-3-8B-Instruct.safelock')
    except:
        all_head_wise_activations = model.preprocess_activate_dataset(dataset,
                                                                      get_prompt_template(prompt_template_style))
        torch.save(all_head_wise_activations, '../Meta_safetylock_weight/Meta-Llama-3-8B-Instruct.safelock')

    # Process activations
    model.reset_all()
    labels = []
    head_wise_activations = []
    for i in range(len(all_head_wise_activations)):
        if i % 10 < 5:
            labels.append(1)
            head_wise_activations.append(all_head_wise_activations[i].copy())
        else:
            labels.append(0)
            head_wise_activations.append(all_head_wise_activations[i].copy())

    # Set model activations
    activate = model.get_activations(head_wise_activations, labels, num_to_intervene=K)
    model.reset_all()
    model.activate = activate
    model.set_activate(activate, alpha)

    return model


def main(
        model_name: str = None,
        peft_model: str = None,
        max_new_tokens: int = 512,
        seed: int = 42,
        use_fast_kernels: bool = False,
        output_file: str = 'path/to/output',
        intervention: bool = True,
        alpha: int = None,
        K: int = None,
        defense: str = None,
        model=None,
        tokenizer=None,
        attack=None,
        eval_files=None,
        **kwargs
):
    """Main function for model inference and evaluation."""

    # Set up the environment
    prompt_template_style = 'alpaca'
    prompt_file = 'path/to/prompt_file.csv'

    # Set random seeds
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    # Prepare tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Apply safety interventions if required
    if intervention:
        model = set_model(model, prompt_template_style, use_safe_dataset=True, alpha=alpha, K=K, model_name=model_name)

    model.model.eval()
    if peft_model:
        model = load_peft_model(model, peft_model)

    # Set up defense mechanisms
    if defense == 'PPL':
        ppl_calculator = PPL_Calculator(model='openai-community/gpt2-large')
    elif defense == 'Retokenization':
        merge_table_path = './eval_utils/subword_nmt.voc'
        merge_table = load_subword_nmt_table(merge_table_path)
        subword_nmt_tokenizer = BpeOnlineTokenizer(
            bpe_dropout_rate=0.2,
            merge_table=merge_table)

    # Enable fast kernels if required
    if use_fast_kernels:
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' before proceeding.")

    # Start timing
    time_start = time.time()

    def generate(model, tokenizer, dataset, defense=None):
        """Generate responses for the given dataset."""
        chats, chats_base = apply_prompt_template(prompt_template_style, dataset, tokenizer, defense=defense)
        tokens = [torch.tensor(chat).long().to("cuda:0") for chat in chats]

        with torch.no_grad():
            input_token_length = [token.shape[0] for token in tokens]
            results = []

            # Apply different defense mechanisms
            if defense == 'PPL':
                # Perplexity-based defense
                ppl_score = ppl_calculator.get_perplexity(dataset)
                outputs = model.generate(tokens, max_new_tokens=max_new_tokens)
                for idx, output in enumerate(outputs):
                    if ppl_score[idx] > 175.57:
                        output_text = "Sorry, I cannot answer your question."
                    else:
                        output_text = tokenizer.decode(output[input_token_length[idx]:]).split('<|eot_id|>')[0]
                    results.append(output_text)

            elif defense == "Paraphrase":
                # Paraphrase-based defense
                outputs = model.generate(tokens, max_new_tokens=max_new_tokens)
                step1_outputs = [tokenizer.decode(output[input_token_length[idx]:]).split('<|eot_id|>')[0] for
                                 idx, output in enumerate(outputs)]

                _, chats = apply_prompt_template(prompt_template_style, step1_outputs, tokenizer, defense=defense)
                tokens = [torch.tensor(chat).long().to("cuda:0") for chat in chats]
                input_token_length = [token.shape[0] for token in tokens]
                outputs = model.generate(tokens, max_new_tokens=max_new_tokens)

                results = [tokenizer.decode(output[input_token_length[idx]:]).split('<|eot_id|>')[0] for idx, output in
                           enumerate(outputs)]

            elif defense == 'Retokenization':
                # Retokenization-based defense
                user_prompt_retokenized = [
                    subword_nmt_tokenizer(data, sentinels=['', '</w>'], regime='end', bpe_symbol=' ') for data in
                    dataset]
                _, chats = apply_prompt_template(prompt_template_style, user_prompt_retokenized, tokenizer,
                                                 defense=defense)
                tokens = [torch.tensor(chat).long().to("cuda:0") for chat in chats]
                input_token_length = [token.shape[0] for token in tokens]
                outputs = model.generate(tokens, max_new_tokens=max_new_tokens)

                results = [tokenizer.decode(output[input_token_length[idx]:]).split('<|eot_id|>')[0] for idx, output in
                           enumerate(outputs)]

            elif defense in ['Self-Reminder', 'ICD']:
                # Self-Reminder or ICD defense
                outputs = model.generate(tokens, max_new_tokens=max_new_tokens)
                results = [tokenizer.decode(output[input_token_length[idx]:]).split('<|eot_id|>')[0] for idx, output in
                           enumerate(outputs)]

            elif defense == 'Self-Exam':
                # Self-Exam defense
                tokens = [torch.tensor(chat).long().to("cuda:0") for chat in chats_base]
                input_token_length = [token.shape[0] for token in tokens]
                outputs = model.generate(tokens, max_new_tokens=max_new_tokens)

                step1_outputs = [tokenizer.decode(output[input_token_length[idx]:]).split('<|eot_id|>')[0] for
                                 idx, output in enumerate(outputs)]

                chats, _ = apply_prompt_template(prompt_template_style, results, tokenizer, defense=defense)

                outputs = model.generate(tokens, max_new_tokens=20)
                step2_outputs = [tokenizer.decode(output[input_token_length[idx]:]).split('<|eot_id|>')[0] for
                                 idx, output in enumerate(outputs)]

                for idx, step2_output in enumerate(step2_outputs):
                    if "yes" in step2_output.lower():
                        output = "Sorry, I cannot answer your question."
                    elif "no" in step2_output.lower():
                        output = step1_outputs[idx]
                    else:
                        output = step1_outputs[idx]
                    results.append(output)

            else:
                # No defense
                tokens = [torch.tensor(chat).long().to("cuda:0") for chat in chats_base]
                input_token_length = [token.shape[0] for token in tokens]
                outputs = model.generate(tokens, max_new_tokens=max_new_tokens)

                results = [tokenizer.decode(output[input_token_length[idx]:]).split('<|eot_id|>')[0] for idx, output in
                           enumerate(outputs)]

        return results

    # Load dataset for attack evaluation
    from datasets import load_from_disk
    if attack:
        if attack in ["GCG", "AutoDAN", "PAIR"]:
            attack_prompts = load_dataset('flydust/SafeDecoding-Attackers', split="train")
            attack_prompts = attack_prompts.filter(lambda x: x['source'] == attack)
            attack_prompts = attack_prompts.filter(lambda x: x['target-model'] == "llama2")
        elif attack == "DeepInception":
            attack_prompts = load_dataset('flydust/SafeDecoding-Attackers', split="train")
            attack_prompts = attack_prompts.filter(lambda x: x['source'] == attack)
        else:
            raise ValueError("Invalid attacker name.")

        question_dataset = [i['prompt'] for i in attack_prompts]

        out = []

        with torch.no_grad():
            batch = 10 if defense == 'Retokenization' else 32
            for i in trange(0, len(question_dataset), batch):
                outputs = generate(model, tokenizer, question_dataset[i:i + batch], defense=defense)

                for idx, output_text in enumerate(outputs):
                    out.append({'prompt': question_dataset[i + idx], 'answer': output_text})
                    print('\n\n\n')
                    print('>>> sample - ' + str(i + idx))
                    print('prompt = ', question_dataset[i + idx])
                    print('answer = ', output_text)

        time_end = time.time()
        if output_file is not None:
            filename = f'{attack}_LOADIN_LLAMA_{model_name.split("/")[-1]}'
            if intervention:
                filename += f'_{alpha}_{K}'
            filename += f'_{str(defense)}_TIME_{str(time_end - time_start)}.json'

            with open(os.path.join(output_file, filename), 'w') as f:
                for li in out:
                    f.write(json.dumps(li))
                    f.write("\n")

    else:
        if "HEx-PHI" in eval_files:
            results = {}
            for p in range(1, 12):
                results[str(p)] = []
                question_dataset = question_read(eval_files + f'/category_{p}.csv')

                # Apply prompt template
                chats = apply_prompt_template(prompt_template_style, question_dataset, tokenizer)

                with torch.no_grad():
                    batch = 15
                    for i in trange(0, len(chats), batch):
                        tokens = [torch.tensor(chat).long().to("cuda:0") for chat in chats[i:i + batch]]
                        input_token_length = [token.shape[0] for token in tokens]
                        outputs = model.generate(tokens, max_new_tokens=max_new_tokens)

                        for idx, output in enumerate(outputs):
                            output_text = tokenizer.decode(output[input_token_length[idx]:]).split('<|eot_id|>')[0]

                            results[str(p)].append({'prompt': question_dataset[i + idx], 'answer': output_text})
                            print('\n\n\n')
                            print('>>> sample - ' + str(i + idx))
                            print('prompt = ', question_dataset[i + idx])
                            print('answer = ', output_text)

            if output_file is not None:
                if intervention:
                    with open(output_file + f'/HEx_PHI_{model_name.split("/")[-1]}_{alpha}_{K}.json',
                              'w') as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                else:
                    with open(output_file + f'/HEx_PHI_{model_name.split("/")[-1]}.json', 'w') as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)

        else:
            question_dataset = question_read(eval_files)

            # Apply prompt template
            chats = apply_prompt_template(prompt_template_style, question_dataset, tokenizer)

            out = []

            with torch.no_grad():
                batch = 15
                for i in trange(0, len(chats), batch):
                    tokens = [torch.tensor(chat).long().to("cuda:0") for chat in chats[i:i + batch]]
                    input_token_length = [token.shape[0] for token in tokens]
                    outputs = model.generate(tokens, max_new_tokens=max_new_tokens, eos_id=False)

                    for idx, output in enumerate(outputs):
                        output_text = tokenizer.decode(output[input_token_length[idx]:]).split('<|eot_id|>')[0]

                        out.append({'prompt': question_dataset[i + idx], 'answer': output_text})
                        print('\n\n\n')
                        print('>>> sample - ' + str(i + idx))
                        print('prompt = ', question_dataset[i + idx])
                        print('answer = ', output_text)

            if output_file is not None:
                if intervention:
                    with open(output_file + f'/advbench_LOADIN_LLAMA_{model_name.split("/")[-1]}_{alpha}_{K}.json',
                              'w') as f:
                        for li in out:
                            f.write(json.dumps(li))
                            f.write("\n")
                else:
                    with open(output_file + f'/advbench_LOADIN_LLAMA_{model_name.split("/")[-1]}.json', 'w') as f:
                        for li in out:
                            f.write(json.dumps(li))
                            f.write("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference with safety evaluation on language models.")
    parser.add_argument("--model", type=str, default="alpaca_7b-full-hf", help="Model name")
    parser.add_argument("--defense", type=str, default="Vanilla", choices=["Vanilla", "SafeLock", "PPL", "Retokenization", "Self-Reminder", "ICD", "Self-Exam"], help="Defense mechanism to use")
    parser.add_argument("--intervention", action="store_true", help="Whether to use intervention")
    parser.add_argument("--alpha", type=int, default=6, help="Alpha value for SafeLock")
    parser.add_argument("--K", type=int, default=24, help="K value for SafeLock")
    parser.add_argument("--output_dir", type=str, default="path/to/output", help="Output directory for results")
    parser.add_argument("--eval_files", type=str, default="path/to/eval_file", help="Path to evaluation file or directory, ['./eval_files/harmful_behaviors.csv','./eval_files/HEx_PHI']")
    parser.add_argument("--mode", type=str, choices=["Policy", "AdvBench"], default="AdvBench", help="Evaluation mode")

    args = parser.parse_args()

    model_name = f"path/to/finetuned_models/{args.model}"
    model, tokenizer = get_model(model_name)

    torch.cuda.reset_peak_memory_stats()

    main(model_name=model_name,
         intervention=args.intervention,
         model=model,
         tokenizer=tokenizer,
         defense=args.defense,
         output_file=args.output_dir,
         alpha=args.alpha if args.defense == "SafeLock" else None,
         K=args.K if args.defense == "SafeLock" else None,
         mode=args.mode,
         eval_files=args.eval_files)