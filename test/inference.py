from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel  # 0.4.0
import torch
import requests
import json
import dotenv
from fireworks.client import Fireworks


def inference(args: {}, inputs: [str], max_new_token=60, delimiter="\n", if_print_out=False):
    config = dotenv.dotenv_values("../.env")
    if "fireworks" in args.base_model:  # Use API
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

    else:  # Local

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
