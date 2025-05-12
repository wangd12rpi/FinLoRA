import os
import json
import glob
import re
import difflib
import tiktoken
import pandas as pd
from tqdm import tqdm

encoding = tiktoken.get_encoding("cl100k_base")
tokenize_function = encoding.encode

def count_tokens(text_string):
    return len(tokenize_function(text_string))

def extract_all_words(text_string):
    return re.findall(r'\b\w+\b', text_string.lower())

def words_are_similar(word_one, term_word):
    return difflib.SequenceMatcher(None, word_one, term_word).ratio() >= 0.8

xbrl_dataframe = pd.read_csv('XBRL Terminology.csv')
xbrl_terms_list = [str(term).lower() for term in xbrl_dataframe['Term'].dropna().unique()]

dataset_files = {
    'gpqa_main': 'gpqa_main.csv',
    'gsm8k': 'gsm8k.json',
    'human_eval_v2': 'human-eval-v2.jsonl',
    'mmlu_test': 'mmlu_test.json'
}

results_list = []
output_path = 'prompt_statistics.txt'
header_fields = ['dataset', 'average_prompt_tokens', 'total_prompt_tokens', 'average_term_frequency', 'overall_term_ratio']
with open(output_path, 'w') as out_file:
    out_file.write('\t'.join(header_fields) + '\n')

def analyze_prompt_collection(dataset_name, prompt_collection):
    prompt_count_value = 0
    total_token_count_value = 0
    total_word_count_value = 0
    total_matched_word_count_value = 0
    fractional_match_sum_value = 0
    for prompt_text_value in tqdm(prompt_collection, desc=dataset_name):
        prompt_count_value += 1
        current_token_count_value = count_tokens(prompt_text_value)
        total_token_count_value += current_token_count_value
        word_list_value = extract_all_words(prompt_text_value)
        word_count_current_value = len(word_list_value)
        total_word_count_value += word_count_current_value
        matched_word_count_current_value = 0
        prompt_lower = prompt_text_value.lower()
        for term_word in xbrl_terms_list:
            if ' ' in term_word:
                if re.search(r'\b' + re.escape(term_word) + r'\b', prompt_lower):
                    matched_word_count_current_value += len(term_word.split())
            else:
                for single_word in word_list_value:
                    if words_are_similar(single_word, term_word):
                        matched_word_count_current_value += 1
                        break
        total_matched_word_count_value += matched_word_count_current_value
        if word_count_current_value > 0:
            fractional_match_sum_value += matched_word_count_current_value / word_count_current_value
    if prompt_count_value > 0:
        average_token_length_value = total_token_count_value / prompt_count_value
        average_term_frequency_value = fractional_match_sum_value / prompt_count_value
    else:
        average_token_length_value = 0
        average_term_frequency_value = 0
    if total_word_count_value > 0:
        overall_term_ratio_value = total_matched_word_count_value / total_word_count_value
    else:
        overall_term_ratio_value = 0
    row = (dataset_name, average_token_length_value, total_token_count_value, average_term_frequency_value, overall_term_ratio_value)
    results_list.append(row)
    with open(output_path, 'a') as out_file:
        out_file.write('\t'.join(str(field) for field in row) + '\n')
        out_file.flush()

gpqa_dataframe = pd.read_csv(dataset_files['gpqa_main'])
gpqa_prompt_list = gpqa_dataframe['Question'].astype(str).tolist()
analyze_prompt_collection('gpqa_main', gpqa_prompt_list)

with open(dataset_files['gsm8k'], 'r') as gsm_file_handle:
    gsm8k_data_list = json.load(gsm_file_handle)
gsm8k_prompt_list = [dataset_record['question'] for dataset_record in gsm8k_data_list]
analyze_prompt_collection('gsm8k', gsm8k_prompt_list)

human_eval_prompt_list = []
with open(dataset_files['human_eval_v2'], 'r') as human_file_handle:
    for jsonl_line_value in tqdm(human_file_handle, desc='human_eval_v2'):
        human_record_dictionary = json.loads(jsonl_line_value)
        human_eval_prompt_list.append(human_record_dictionary['test'])
analyze_prompt_collection('human_eval_v2', human_eval_prompt_list)

with open(dataset_files['mmlu_test'], 'r') as mmlu_file_handle:
    mmlu_data_list = json.load(mmlu_file_handle)
mmlu_prompt_list = []
for mmlu_record in tqdm(mmlu_data_list, desc='mmlu_test'):
    question_text_string = mmlu_record['question']
    choices_text_string = ', '.join(mmlu_record['choices'])
    full_prompt_text_string = f"Answer this {question_text_string} with one of the following answer choices: {choices_text_string}"
    mmlu_prompt_list.append(full_prompt_text_string)
analyze_prompt_collection('mmlu_test', mmlu_prompt_list)

test_folder_directory = 'test'
for jsonl_file_path in glob.glob(os.path.join(test_folder_directory, '*.jsonl')):
    dataset_name_string = os.path.splitext(os.path.basename(jsonl_file_path))[0]
    test_prompt_list = []
    with open(jsonl_file_path, 'r') as test_file_handle:
        for jsonl_line_content in tqdm(test_file_handle, desc=dataset_name_string):
            test_record_dictionary = json.loads(jsonl_line_content)
            test_prompt_list.append(test_record_dictionary['context'])
    analyze_prompt_collection(dataset_name_string, test_prompt_list)
