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
    fpb_datasets = fpb_datasets.train_test_split(test_size=0.25, seed=42)
    train, test = fpb_datasets["train"].to_pandas(), fpb_datasets["test"].to_pandas()

    def change_prompt(data):
        data.columns = ["input", "output"]
        data["output"] = data["output"].apply(lambda x: dic[x])
        data[
            "instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
        data = datasets.Dataset.from_pandas(data)
        return data

    train = change_prompt(train)
    test = change_prompt(test)
    return train, test


def process_fiqa():
    fiqa = load_dataset('ChanceFocus/flare-fiqasa')
    train, test = fiqa['train'].to_pandas(), fiqa['test'].to_pandas()

    def change_prompt(data):
        data["instruction"] = data["query"].apply(add_instructions)
        data = data[['text', 'answer', "instruction"]]
        data.columns = ["input", "output", "instruction"]
        data = datasets.Dataset.from_pandas(data)
        return data

    train = change_prompt(train)
    test = change_prompt(test)
    return train, test


def process_tfns():
    tfns = load_dataset('zeroshot/twitter-financial-news-sentiment')
    train, test = tfns['train'].to_pandas(), tfns['validation'].to_pandas()

    def change_prompt(data):
        data['label'] = data['label'].apply(lambda x: dic[x])
        data[
            'instruction'] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
        data.columns = ['input', 'output', 'instruction']
        return datasets.Dataset.from_pandas(data)

    train = change_prompt(train)
    test = change_prompt(test)
    return train, test


def process_nwgi():
    nwgi = load_dataset('oliverwang15/news_with_gpt_instructions')
    train, test = nwgi['train'].to_pandas(), nwgi['test'].to_pandas()

    def change_prompt(data):
        data['output'] = data['label']
        data["input"] = data["news"]
        data[
            "instruction"] = 'What is the sentiment of this news? Please choose an answer from {strong negative/moderately negative/mildly negative/neutral/mildly positive/moderately positive/strong positive}.'
        data = data[['input', 'output', 'instruction']]
        data = datasets.Dataset.from_pandas(data)
        return data

    train = change_prompt(train)
    test = change_prompt(test)
    return train, test


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
    ner = load_dataset('FinGPT/fingpt-ner-cls')
    headline = load_dataset('FinGPT/fingpt-headline')

    fpb_train, fpb_test = process_fpb()
    fiqa_train, fiqa_test = process_fiqa()
    nwgi_train, nwgi_test = process_nwgi()
    tfns_train, tfns_test = process_tfns()
    ner_train, ner_test = ner['train'], ner['test']
    headline_train, headline_test = headline['train'], headline['test']

    # fpb_train = datasets.concatenate_datasets([fpb_train] * 6)
    # fiqa_train = datasets.concatenate_datasets([fiqa_train] * 21)
    # tfns_train = datasets.concatenate_datasets([tfns_train] * 2)

    print(fpb_train)
    print(fiqa_train)
    print(tfns_train)
    print(nwgi_train)
    # sentiment_train_dataset = datasets.concatenate_datasets([fpb_train, fiqa_train, tfns_train, nwgi_train])
    # print("final", sentiment_train_dataset)
    #
    # process_save_data(sentiment_train_dataset, "finlora_sentiment_train", split='train')
    # process_save_data(headline_train, "headline_train", split='train')
    # process_save_data(ner_train, "ner_train", split='train')
    #
    # process_save_data(fpb_test, "fpb_test", split='test')
    # process_save_data(fiqa_test, "fiqa_test", split='test')
    # process_save_data(nwgi_test, "nwgi_test", split='test')
    # process_save_data(tfns_test, "tfns_test", split='test')
    # process_save_data(headline_test, "headline_test", split='test')
    # process_save_data(ner_test, "ner_test", split='test')
    #
