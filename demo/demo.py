import gradio as gr
import dotenv
from fireworks.client import Fireworks

tagging_example = [
    ["Llama 3.1 8B (Finetuned)", "1.5",
     "47 Table of Contents DELL TECHNOLOGIES INC . NOTES TO CONDENSED CONSOLIDATED FINANCIAL STATEMENTS ( Continued ) ( unaudited ) NOTE 11 - INCOME AND OTHER TAXES For the three months ended August 2 , 2019 and August 3 , 2018 , the Company s effective income tax rates were 3912.6 % and 1.5 % , respectively , on pre - tax losses of $ 111 million and $ 468 million , respectively ."],
    ["Llama 3.1 8B (Finetuned)", "4.0",
     "As a result of this modification , the Company adjusted the value of its right - of - use asset and operating lease liability using an incremental borrowing rate of approximately 4.0 % "]
]

models = {"Llama 3.1 8B (Finetuned)": "llama-v3p1-8b-instruct",
          "Llama 3.1 8B (Base)": "accounts/fireworks/models/llama-v3p1-8b-instruct"}

def inference(inputs: str, max_new_token=60, delimiter="\n", if_print_out=False):
    config = dotenv.dotenv_values("../.env")

    client = Fireworks(api_key=config["FIREWORKS_KEY"])
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        max_tokens=max_new_token,
        messages=[
            {
                "role": "user",
                "content": inputs
            }
        ],
        stream=False
    )
    answer = (response.choices[0].message.content)
    # print(answer)
    return answer


def sentence_builder(model, keyword, sentence):
    prompt = f'''What is the appropriate XBRL US GAAP tag for "{keyword}" in the given sentence? Output the US GAAP tag only and nothing else. \n "{sentence}"\n'''

    answer = f"""Prompt: {prompt}
Output: {inference(prompt)}"""
    return answer


if __name__ == '__main__':
    with gr.Blocks() as tagging:
        gr.Markdown(
            "XBRL Tagging: The LLM will output a US GAAP tag for a given keyword in a sentence. Feel free to click any of the examples below to try them or enter your own.")
        gr.Interface(
            fn=sentence_builder,
            inputs=[
                gr.Dropdown(
                    ["Llama 3.1 8B (Finetuned)", "Llama 3.1 8B (Base)"], label="Model", info=""
                ),
                gr.Textbox(label="Keyword"),
                gr.Textbox(label="Sentence"),
            ],
            outputs="text",
            examples=tagging_example
        )

    extraction = gr.Interface(
        fn=sentence_builder,
        inputs=[
            gr.Dropdown(
                ["Llama 3.1 8B (Finetuned)", "Llama 3.1 8B (Base)"], label="Model", info=""
            ),
            gr.Textbox(label="Sentence"),
            gr.Textbox(label="Keyword"),

        ],
        outputs="text",
    )

    with gr.Blocks() as demo:
        gr.Markdown("# FinLoRA Demo")
        gr.TabbedInterface([tagging, extraction], ["XBRL Tagging", "XBRL Extraction"])

    demo.launch()
