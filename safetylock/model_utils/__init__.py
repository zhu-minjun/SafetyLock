# https://github.com/WENGSYX/ControlLM
# Authors: Yixuan Weng (wengsyx@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from einops import rearrange
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer,AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.rich import tqdm
from baukit import Trace, TraceDict


from copy import deepcopy

def get_model(model_name,use_bit_4=False):

    class SafetyLockLM:
        def __init__(self):

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name
            )
            self.config = AutoConfig.from_pretrained(model_name)
            if self.config.architectures[0] == 'MistralForCausalLM':
                from .modeling_mistral import MistralForCausalLM as ModelForCausalLM
            elif self.config.architectures[0] == 'LlamaForCausalLM':
                from .modeling_llama import LlamaForCausalLM as ModelForCausalLM
            else:
                print('SafetyLockLM not implemented yet for {}.'.format(self.config.architectures[0]))
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
            self.activate = None

            self.model_file = model_name

            if use_bit_4:
                model = ModelForCausalLM.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                )
                self.weight_cache = []
                for i, layer in enumerate(model.model.layers):
                    self.weight_cache.append(deepcopy(model.model.layers[i].self_attn.o_proj.weight).cuda())
                model = None
                torch.cuda.empty_cache()

                from transformers import BitsAndBytesConfig, pipeline
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                )
                self.model = ModelForCausalLM.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    device_map='auto'
                )
            else:
                self.model = ModelForCausalLM.from_pretrained(
                    model_name,
                ).half().cuda()

            self.model.eval()
            self.device = self.model.device

            self.bias_cache = []
            for i, layer in enumerate(self.model.model.layers):
                self.bias_cache.append(deepcopy(self.model.model.layers[i].self_attn.o_proj.bias))

        def generate(model,text,max_length=512,max_new_tokens=None,eos_id=False):

            tokenizer = model.tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            stop_id = tokenizer.sep_token_id
            pad_id = tokenizer.pad_token_id

            device = model.device
            input_ids = []
            for t in text:
                input_ids.append(t)
            min_prompt_len = min(len(t) for t in input_ids)
            max_prompt_len = max(len(t) for t in input_ids)

            if max_new_tokens:
                max_length = max_prompt_len + max_new_tokens
            tokens = torch.full((len(input_ids), max_length), pad_id, dtype=torch.long).to(device)
            for k, t in enumerate(input_ids):
                tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
            prev_pos = 0
            cur_pos = min_prompt_len - 1
            input_text_mask = tokens != pad_id
            eos_reached = torch.tensor([eos_id] * len(input_ids), device=device)
            past_key_values = None

            with torch.no_grad():
                for cur_pos_add in range(max_length):
                    cur_pos += 1
                    if prev_pos != 0:
                        prev_pos = cur_pos - 1
                    if tokens.shape[1] == cur_pos:
                        break
                    torch.cuda.empty_cache()

                    logits = model.model(tokens[:, prev_pos:cur_pos], use_cache=True,
                                         past_key_values=past_key_values)
                    next_token = torch.topk(logits['logits'][:, -1], 1, dim=-1)[1][:, -1]
                    next_token = next_token.reshape(-1)
                    next_token = torch.where(
                        input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                    )
                    tokens[:, cur_pos] = next_token
                    eos_reached |= (~input_text_mask[:, cur_pos]) & (
                            next_token == model.tokenizer.eos_token_id
                    )

                    if all(eos_reached):
                        break
                    prev_pos = cur_pos
                    past_key_values = logits["past_key_values"]
            return tokens

        def __call__(self, input_ids):
            with torch.no_grad():
                logits = self.model(input_ids)
                return logits

        def get_last_activations(self, layer):
            return self.model.model.layers[layer].activations

        def set_add_activations(self, layer, activations):
            activations = activations.half()
            self.model.model.layers[layer].add(activations)


        def reset_all(self):
            for i,layer in enumerate(self.model.model.layers):
                self.model.model.layers[i].self_attn.o_proj.bias = self.bias_cache[i]


        def get_activations(self, all_head_wise_activations, labels, num_to_intervene=48):
            def get_top_heads(separated_activations, separated_labels, num_layers, num_heads, num_to_intervene):

                probes, all_head_accs_np = train_probes(separated_activations,
                                                        separated_labels, num_layers=num_layers, num_heads=num_heads)
                all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads,2)
                all_head_accs_np = all_head_accs_np.mean(2)
                top_accs = np.argsort(all_head_accs_np.reshape(num_heads * num_layers))[::-1][:num_to_intervene]
                top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]

                return top_heads, probes

            def train_probes(separated_head_wise_activations, separated_labels,
                             num_layers, num_heads):

                all_head_accs = []
                probes = []

                train_idxs = np.arange(len(separated_labels))

                # pick a val set using numpy
                train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs) * (1 - 0.4)),
                                                  replace=False)
                val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

                all_X_train = np.array([separated_head_wise_activations[i] for i in train_set_idxs])
                all_X_val = np.array([separated_head_wise_activations[i] for i in val_set_idxs])
                y_train = np.array([separated_labels[i] for i in train_set_idxs])
                y_val = np.array([separated_labels[i] for i in val_set_idxs])

                for layer in tqdm(range(num_layers)):
                    for head in range(num_heads):
                        X_train = all_X_train[:, layer, head, :]
                        X_val = all_X_val[:, layer, head, :]

                        clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
                        y_pred = clf.predict(X_train)
                        y_val_pred = clf.predict(X_val)
                        all_head_accs.append([accuracy_score(y_val, y_val_pred),accuracy_score(y_train,y_pred)])
                        probes.append(clf)

                all_head_accs_np = np.array(all_head_accs)

                return probes, all_head_accs_np

            def flattened_idx_to_layer_head(flattened_idx, num_heads):
                return flattened_idx // num_heads, flattened_idx % num_heads

            def layer_head_to_flattened_idx(layer, head, num_heads):
                return layer * num_heads + head

            def get_interventions_dict(top_heads, probes, tuning_activations, num_heads,com_directions=None):

                interventions = {}
                for layer, head in top_heads:
                    interventions[f"model.layers.{layer}.self_attn.head_out"] = []
                for layer, head in top_heads:
                    if com_directions is not None:
                        direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
                    else:
                        direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
                    direction = direction / np.linalg.norm(direction)
                    activations = tuning_activations[:, layer, head, :]  # batch x 128
                    proj_vals = activations @ direction.T
                    proj_val_std = np.std(proj_vals)
                    interventions[f"model.layers.{layer}.self_attn.head_out"].append(
                        (head, direction.squeeze(), proj_val_std))
                for layer, head in top_heads:
                    interventions[f"model.layers.{layer}.self_attn.head_out"] = sorted(
                        interventions[f"model.layers.{layer}.self_attn.head_out"], key=lambda x: x[0])

                return interventions

            def get_com_directions(num_layers, num_heads,usable_head_wise_activations,
                                   usable_labels):

                com_directions = []

                for layer in range(num_layers):
                    for head in range(num_heads):
                       # usable_idxs = np.concatenate([train_set_idxs, val_set_idxs], axis=0)
                        #usable_head_wise_activations = np.concatenate(
                            #[separated_head_wise_activations[i][:, layer, head, :] for i in usable_idxs], axis=0)
                        #usable_labels = np.concatenate([separated_labels[i] for i in usable_idxs], axis=0)
                        usable_labels = np.array(usable_labels)
                        head_wise_activations = usable_head_wise_activations[:,layer, head, :]
                        true_mass_mean = np.mean(head_wise_activations[usable_labels == 1], axis=0)
                        false_mass_mean = np.mean(head_wise_activations[usable_labels == 0], axis=0)
                        com_directions.append(true_mass_mean - false_mass_mean)
                com_directions = np.array(com_directions)

                return com_directions

            num_layers = self.model.model.config.num_hidden_layers
            num_heads = self.model.model.config.num_attention_heads

            head_wise_activations = deepcopy(all_head_wise_activations)
            head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)
            tuning_activations = deepcopy(all_head_wise_activations)
            tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h=num_heads)

            top_heads, probes = get_top_heads(head_wise_activations, labels, num_layers, num_heads, num_to_intervene)

            com_directions = get_com_directions(num_layers, num_heads, head_wise_activations,
                                                labels)

            interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads,com_directions)

            return interventions

        def preprocess_activate_dataset(self,dataset,system_prompt="You are a helpful, honest and concise assistant.",last_token_for_get_activate=5):
            self.system_prompt = system_prompt

            def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output,use_base=False):
                if use_base:
                    return torch.tensor(tokenizer(system_prompt.format(instruction)+model_output)).unsqueeze(0)
                if type(instruction) == str:
                    if model_output:
                        con = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": model_output}
                        ]
                        return torch.tensor(tokenizer.apply_chat_template(con)[:-5]).unsqueeze(0)
                    else:
                        con = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": instruction},
                        ]
                        return torch.tensor(tokenizer.apply_chat_template(con)).unsqueeze(0)
                else:
                    con = instruction
                    con.append({'role':'assistant','content':model_output})
                    return torch.tensor(tokenizer.apply_chat_template(con)[:-5]).unsqueeze(0)

            def data_preprocess(dataset):
                all_prompts = []
                for i in range(len(dataset)):
                    question = dataset[i]['question']

                    pos_answer = dataset[i]['answer_matching_behavior']
                    pos_tokens = prompt_to_tokens(
                        self.tokenizer, self.system_prompt, question, pos_answer
                    )
                    all_prompts.append(pos_tokens)


                    neg_answer = dataset[i]['answer_not_matching_behavior']
                    neg_tokens = prompt_to_tokens(
                        self.tokenizer, self.system_prompt, question, neg_answer
                    )
                    all_prompts.append(neg_tokens)

                return all_prompts

            def get_llama_activations_bau(self, prompt):

                HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(self.model.config.num_hidden_layers)]
                MLPS = [f"model.layers.{i}.mlp" for i in range(self.model.config.num_hidden_layers)]

                with torch.no_grad():
                    prompt = [p.squeeze(0) for p in prompt]

                    len_num = [p.shape[0] for p in prompt]
                    max_len = max(len_num)

                    prompt = [torch.cat([p,torch.ones(max_len-p.shape[0])*self.tokenizer.pad_token_id], dim=0).to(torch.int) for p in prompt]
                    prompt = torch.stack(prompt)
                    prompt = prompt.to(self.model.device)
                    with TraceDict(self.model, HEADS + MLPS) as ret:
                        output = self.model(prompt, output_hidden_states=True)
                    hidden_states = output.hidden_states
                    hidden_states = torch.stack(hidden_states, dim=0).squeeze()
                    hidden_states = hidden_states.to(torch.float16).detach().cpu().numpy()
                    head_wise_hidden_states = [ret[head].output.squeeze().to(torch.float16).detach().cpu() for head in HEADS]
                    head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
                    mlp_wise_hidden_states = [ret[mlp].output.squeeze().to(torch.float16).detach().cpu() for mlp in MLPS]
                    mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()

                return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, len_num

            prompts = data_preprocess(dataset)

            all_layer_wise_activations = []
            all_head_wise_activations = []

            batch = 16
            for prompt_num in tqdm(range(0,len(prompts),batch)):
                layer_wise_activation, head_wise_activation, _, len_num = get_llama_activations_bau(self, prompts[prompt_num:prompt_num+batch])
                for index in range(len(len_num)):
                    for i in range(1,last_token_for_get_activate+1):
                        all_layer_wise_activations.append(layer_wise_activation[:, index, len_num[index]-i, :])
                        all_head_wise_activations.append(head_wise_activation[:, index, len_num[index]-i, :])

            return all_head_wise_activations

        def set_activate(self,interventions,alpha):
            num_layers = self.model.model.config.num_hidden_layers
            num_heads = self.model.model.config.num_attention_heads

            for head_out_name, list_int_vec in interventions.items():
                layer_no = int(head_out_name.split('.')[2])
                displacement = np.zeros((int(num_heads), int(self.model.model.config.hidden_size / num_heads)))
                for head_no, head_vec, std in list_int_vec:
                    displacement[head_no] = alpha * std * head_vec
                device = self.model.model.layers[layer_no].self_attn.o_proj.weight.device.index
                displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
                if '70B' in self.model_file.upper():
                    bias_tobe = F.linear(displacement.to(torch.bfloat16),self.weight_cache[layer_no]).to(
                        device)
                else:
                    bias_tobe = F.linear(displacement.to(torch.float16),
                                         self.model.model.layers[layer_no].self_attn.o_proj.weight).to(
                        device)
                self.model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)




    model = SafetyLockLM()
    model.reset_all()

    return model, model.tokenizer
