from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel  # 0.4.0
import torch
import requests
import json
import dotenv
from fireworks.client import Fireworks
import time
import warnings
import sklearn
import sys
from tqdm import tqdm
import pandas as pd
import argparse
from google import genai
from google.genai import types
import base64
from openai import OpenAI
import os
import anthropic

warnings.filterwarnings("ignore")

def load_local_model(args):
    if args.source != 'hf':
        return None, None

    model_name = args.base_model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.quant_bits == 4,
        load_in_8bit=args.quant_bits == 8,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model.model_parallel = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    # ensure a pad token exists and embeddings match
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"

    # resize model embedding layer if tokenizer length changed
    if len(tokenizer) != model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))

    # qwen special ids
    if args.base_model == 'qwen':
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|extra_0|>')

    if args.peft_model != "":
        model = PeftModel.from_pretrained(model, args.peft_model)

    model = model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def inference(args, inputs, max_new_token=60, delimiter="\n", model=None,
              tokenizer=None):

    config = dotenv.dotenv_values("../.env")
    temperature = args.temperature if hasattr(args, 'temperature') else 0.0

    if args.source == "together":
        together_api_key = args.together_api_key if hasattr(args, 'together_api_key') else config.get("TOGETHER_API_KEY")
        answer = []
        headers = {"Authorization": f"Bearer {together_api_key}", "Content-Type": "application/json"}
        url = "https://api.together.xyz/v1/chat/completions"
        model_name = args.model_name if hasattr(args, 'model_name') and args.model_name else args.base_model
        for x in inputs:
            payload = {"model": model_name, "max_tokens": max_new_token,
                       "messages": [{"role": "user", "content": x}],
                       "temperature": temperature}
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                answer.append(response.json()["choices"][0]["message"]["content"])
            else:
                answer.append("Error: Failed to get response from Together AI API")
        return answer

    elif args.source == "fireworks":
        answer = []
        for x in inputs:
            client = Fireworks(api_key=config.get("FIREWORKS_KEY"))
            response = client.chat.completions.create(
                model=args.base_model,
                max_tokens=max_new_token,
                messages=[{"role": "user", "content": x}],
                stream=False
            )
            answer.append(response.choices[0].message.content)
        return answer

    elif args.source == 'google':
        client = genai.Client(
            vertexai=True,
            project="1023064188719",
            location="us-central1",
        )
        answer = []
        for x in inputs:
            generate_content_config = types.GenerateContentConfig(
                temperature=args.temperature,
                top_p=1,
                max_output_tokens=max_new_token,
                response_modalities=["TEXT"],
            )
            response = client.models.generate_content(
                model=args.base_model,
                contents=x,
                config=generate_content_config,
            )
            answer.append(response.text)
        if args.if_print:
            print(answer)
        return answer

    elif args.source == 'openai':
        answer = []
        client = OpenAI()
        for x in inputs:
            response = client.chat.completions.create(
                model=args.model_name if hasattr(args, 'model_name') and args.model_name else "gpt-3.5-turbo",
                messages=[{"role": "user", "content": x}],
                max_tokens=max_new_token,
                temperature=temperature,
                stream=False
            )
            answer.append(response.choices[0].message.content)
        return answer

    elif args.source == 'anthropic':
        key = os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=key)
        answer = []
        for x in inputs:
            response = client.messages.create(
                model=args.model_name if hasattr(args, 'model_name') and args.model_name else "claude-3-sonnet-20240229",
                max_tokens=max_new_token,
                messages=[{"role": "user", "content": x}],
                temperature=temperature
            )
            answer.append(response.content[0].text)
        return answer

    elif args.source == 'deepseek':
        key = os.getenv("DEEPSEEK_API_KEY")
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        answer = []
        for x in inputs:
            response = client.chat.completions.create(
                model=args.model_name if hasattr(args, 'model_name') and args.model_name else "deepseek-chat",
                messages=[{"role": "user", "content": x}],
                max_tokens=max_new_token,
                temperature=temperature,
                stream=False
            )
            answer.append(response.choices[0].message.content)
        return answer

    else:  # Local
        if len(inputs) == 0:
            return []
        tokens = tokenizer(inputs, return_tensors='pt', padding=True, max_length=512,
                           return_token_type_ids=False)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
        res = model.generate(**tokens, max_new_tokens=max_new_token, eos_token_id=tokenizer.eos_token_id,
                             temperature=0.00001)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        out_text = ["".join(o.split(delimiter)[-1]).strip() for o in res_sentences]
        args.if_print and print(out_text)
        return out_text


def evaluate_accuracy(out, target, target_type_list):
    correct_count = 0
    response = []
    target_type_list_lower = [str(t).lower() for t in target_type_list]

    if len(out) != len(target):
        raise ValueError("Input lists 'out' and 'target' must have the same length.")

    for x, y in zip(out, target):
        x_str = str(x)
        y_str = str(y)
        x_lower = x_str.lower()
        y_lower = y_str.lower()
        found_labels_info = []
        for valid_label in target_type_list_lower:
            try:
                index = x_lower.find(valid_label)
                if index != -1:
                    found_labels_info.append({'label': valid_label, 'index': index})
            except AttributeError:
                continue
        is_current_prediction_correct = False
        if found_labels_info:
            found_labels_info.sort(key=lambda item: item['index'])
            first_occurred_label = found_labels_info[0]['label']
            if first_occurred_label == y_lower:
                is_current_prediction_correct = True
        if is_current_prediction_correct:
            correct_count += 1
            response.append(y)
        else:
            response.append(x)
    accuracy = correct_count / len(out) if len(out) > 0 else 0.0
    return accuracy, response


def process_batched(out_text_list, target_list):
    processed_out_text_list = []
    processed_target_list = []
    for out_text, target in zip(out_text_list, target_list):
        split_output = [x.strip().replace("\n", "") for x in out_text.split(',')]
        split_target = [x.strip().replace("\n", "") for x in target.split(',')]
        processed_target_list += split_target
        output_len = len(split_output)
        target_len = len(split_target)
        if output_len != target_len:
            if output_len > target_len:
                processed_out_text_list += split_output[:target_len]
            else:
                processed_out_text_list += split_output + [""] * (target_len - output_len)
        else:
            processed_out_text_list += split_output
    assert len(processed_out_text_list) == len(processed_target_list)
    return processed_out_text_list, processed_target_list
