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


def load_local_model(args):
    if args.source != 'hf':
        return None, None

    model_name = args.base_model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.quant_bits == 4,  # Load in 4-bit if quant_bits is 4
        load_in_8bit=args.quant_bits == 8,  # Load in 8-bit if quant_bits is 8
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model.model_parallel = True

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              torch_dtype=torch.bfloat16,
                                              trust_remote_code=True,
                                              device_map="auto"
                                              )

    tokenizer.padding_side = "left"
    if args.base_model == 'qwen':
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|extra_0|>')

    tokenizer.pad_token = tokenizer.eos_token
    # if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     model.resize_token_embeddings(len(tokenizer))

    print(f'pad: {tokenizer.pad_token_id}, eos: {tokenizer.eos_token_id}')
    # model.generation_config.pad_token_id = tokenizer.pad_token_id

    if args.peft_model != "":
        model = PeftModel.from_pretrained(model, args.peft_model)

    model = model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def inference(args: {}, inputs: [str], max_new_token=60, delimiter="\n", if_print_out=False, model=None,
              tokenizer=None):
    config = dotenv.dotenv_values("../.env")


    together_api_key = args.together_api_key if hasattr(args, 'together_api_key') else config.get("TOGETHER_API_KEY")

    temperature = args.temperature if hasattr(args, 'temperature') else 0.0

    if together_api_key:
        # Use Together API for all models when API key is provided
        answer = []
        headers = {
            "Authorization": f"Bearer {together_api_key}",
            "Content-Type": "application/json"
        }

        url = "https://api.together.xyz/v1/chat/completions"

        model_name = args.base_model

        if if_print_out:
            print(f"Using Together API with model: {model_name}, temperature: {temperature}")

        for x in inputs:
            payload = {
                "model": model_name,
                "max_tokens": max_new_token,
                "messages": [
                    {
                        "role": "user",
                        "content": x
                    }
                ],
                "temperature": temperature
            }

            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                response_json = response.json()
                answer.append(response_json["choices"][0]["message"]["content"])
            else:
                print(f"Error calling Together AI API: {response.status_code} - {response.text}")
                answer.append("Error: Failed to get response from Together AI API")

        return answer

    elif "fireworks" in args.base_model:  # Use Fireworks API
        answer = []
        for x in inputs:
            client = Fireworks(api_key=config["FIREWORKS_KEY"])
            response = client.chat.completions.create(
                model=args.base_model,
                max_tokens=max_new_token,
                messages=[
                    {
                        "role": "user",
                        "content": x
                    }
                ],
                stream=False
            )
            answer.append(response.choices[0].message.content)
            # print(answer)
        return answer

    elif args.source == 'google':

        client = genai.Client(
            vertexai=True,
            project="1023064188719",
            location="us-central1",
            # api_key=config["GOOGLE_KEY"],
        )
        answer = []
        for x in inputs:
            generate_content_config = types.GenerateContentConfig(
                temperature=args.temperature,
                top_p=0.95,
                max_output_tokens=max_new_token,
                response_modalities=["TEXT"],
            )

            response = client.models.generate_content(
                model=args.base_model,
                contents=x,
                config=generate_content_config,
            )
            # print(response.text)
            answer.append(response.text)
        return answer


    else:  # Local
        if len(inputs) == 0:
            return []

        tokens = tokenizer(inputs, return_tensors='pt', padding=True, max_length=512,
                           return_token_type_ids=False)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
        res = model.generate(**tokens, max_new_tokens=max_new_token, eos_token_id=tokenizer.eos_token_id,
                             temperature=0.8)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        out_text = ["".join(o.split(delimiter)[-1]).strip() for o in res_sentences]

        if_print_out and print(out_text)

        return out_text


def evaluate_accuracy(out, target):
    correct_count = 0
    response = []
    for x, y in zip(out, target):
        if y in x:
            correct_count += 1
            response.append(y)
        else:
            response.append(x)

    accuracy = correct_count / len(out)
    return accuracy, response


def process_batched(out_text_list, target_list):
    processed_out_text_list = []
    processed_target_list = []

    for out_text, target in zip(out_text_list, target_list):
        split_output = [x.strip().replace("\n", "") for x in out_text.split(',')]
        split_target = [x.strip().replace("\n", "") for x in target.split(',')]
        processed_target_list += (split_target)
        output_len = len(split_output)
        target_len = len(split_target)

        if output_len != target_len:
            if output_len > target_len:
                # Output is longer, truncate
                processed_out_text_list += (split_output[:target_len])
            else:
                # Target is longer, pad output with empty strings
                padding_needed = target_len - output_len
                processed_out_text_list += (split_output + [""] * padding_needed)
        else:
            # Lengths match, use output as is
            processed_out_text_list += (split_output)
    assert len(processed_out_text_list) == len(processed_target_list)
    return processed_out_text_list, processed_target_list
