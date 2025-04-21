from datasets import load_dataset
import datasets
import jsonlines
dic = {
    0: "negative",
    1: 'neutral',
    2: 'positive',
}


def make_label(x):
    if x < - 0.1:
        return "negative"
    elif x >= -0.1 and x < 0.1:
        return "neutral"
    elif x >= 0.1:
        return "positive"


def add_instructions(x):
    if "post" in x:
        return "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
    else:
        return "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."


def process_fpb():
    fpb_datasets = load_dataset("financial_phrasebank", "sentences_50agree")["train"]
    fpb_datasets = fpb_datasets.train_test_split(test_size=0.25, seed=42)['train'].to_pandas()
    fpb_datasets.columns = ["input", "output"]
    fpb_datasets["output"] = fpb_datasets["output"].apply(lambda x: dic[x])
    fpb_datasets[
        "instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    return datasets.Dataset.from_pandas(fpb_datasets)


def process_fiqa():
    fiqa_datasets = load_dataset('ChanceFocus/flare-fiqasa')['train'].to_pandas()
    fiqa_datasets["instruction"] = fiqa_datasets["query"].apply(add_instructions)
    fiqa_datasets = fiqa_datasets[['text', 'answer', "instruction"]]
    fiqa_datasets.columns = ["input", "output", "instruction"]
    fiqa_datasets = datasets.Dataset.from_pandas(fiqa_datasets)
    return fiqa_datasets


def process_tfns():
    social_media_dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')['train'].to_pandas()
    social_media_dataset['label'] = social_media_dataset['label'].apply(lambda x: dic[x])
    social_media_dataset[
        'instruction'] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
    social_media_dataset.columns = ['input', 'output', 'instruction']
    return datasets.Dataset.from_pandas(social_media_dataset)


def process_nwgi():
    finance_dataset = load_dataset('oliverwang15/news_with_gpt_instructions')['train'].to_pandas()
    finance_dataset['output'] = finance_dataset['label']
    finance_dataset["input"] = finance_dataset["news"]
    finance_dataset[
        "instruction"] = 'What is the sentiment of this news? Please choose an answer from {strong negative/moderately negative/mildly negative/neutral/mildly positive/moderately positive/strong positive}.'
    finance_dataset = finance_dataset[['input', 'output', 'instruction']]
    finance_dataset = datasets.Dataset.from_pandas(finance_dataset)
    return finance_dataset


def process_data(example):
    context = f"Instruction: {example['instruction']}\nInput: {example['input']}\nAnswer: "
    target = example['output']
    return {'context': context, 'target': target}


def process_save_data(all_data, file_name, split='train'):
    processed_data_normal = all_data.map(process_data)
    processed_data_normal = processed_data_normal.remove_columns(["input", "output", "instruction"])
    print("example:", processed_data_normal[0])

    with jsonlines.open(f"{split}/{file_name}.jsonl", 'w') as writer:
        writer.write_all(processed_data_normal)




if __name__ == '__main__':
    fpb_train = process_fpb()
    fiqa_train = process_fiqa()
    nwgi_train = process_nwgi()
    tfns_train = process_tfns()

    fpb_train = datasets.concatenate_datasets([fpb_train] * 6)
    fiqa_train = datasets.concatenate_datasets([fiqa_train] * 21)
    tfns_train = datasets.concatenate_datasets([tfns_train] * 2)

    print(fpb_train)
    print(fiqa_train)
    print(nwgi_train)
    print(tfns_train)
    sentiment_train_dataset = datasets.concatenate_datasets([fpb_train, fiqa_train, tfns_train, nwgi_train])
    print("final", sentiment_train_dataset)

    process_save_data(sentiment_train_dataset, "finlora_sentiment_train")

