import csv
import ast
import json


def create_batched_qa_dataset_from_csv_jsonl(csv_file_path, examples_per_batch, output_jsonl_path):
    qa_dataset = []
    all_possible_tags = set()

    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            sentence = row['sentence']
            numerals_tags_str = row['numerals-tags']

            try:
                numerals_tags_dict = ast.literal_eval(numerals_tags_str)
            except (SyntaxError, ValueError):
                print(f"Warning: Could not parse numerals-tags for sentence: {sentence}. Skipping row.")
                continue

            if not numerals_tags_dict:
                print(f"Warning: No tags found for sentence: {sentence}. Skipping row.")
                continue

            for tag, values in numerals_tags_dict.items():
                if values:  # Ensure there are values associated with the tag
                    numerical_value = values[0]  # Assuming we are interested in the first value if multiple exist.

                    question = f"What is the best us gaap tag for entity \"{numerical_value}\" in sentence: \"{sentence}\"?"
                    answer = tag
                    qa_dataset.append({'question': question, 'answer': answer})
                    all_possible_tags.add(tag)

    batched_dataset = []
    num_batches = (len(qa_dataset) + examples_per_batch - 1) // examples_per_batch

    possible_tags_prompt_section = ("You are XBRL expert. Choose the best XBRL US GAAP tag for each highlighted entity in "
                                    "the sentences below. Provide only the US GAAP tags, comma-separated, in the order of the sentences and highlighted entity. "
                                    "Provide nothing else\n") + ", ".join(
        sorted(list(all_possible_tags))) + "\n"

    possible_tags_prompt_section = possible_tags_prompt_section.replace("  ", "").replace("us-gaap:", "")
    print(len(all_possible_tags))
    for i in range(num_batches):
        start_index = i * examples_per_batch
        end_index = min((i + 1) * examples_per_batch, len(qa_dataset))
        batch_examples = qa_dataset[start_index:end_index]

        batch_qa_pairs = {'context': possible_tags_prompt_section, 'target': ""}
        for example in batch_examples:
            prompt = example['question'] + "\n"
            batch_qa_pairs['context'] += prompt
        batch_qa_pairs['context'] += "\nOutput US GAAP tags:"
        batch_qa_pairs['target'] = ",".join([x['answer'] for x in batch_examples]).replace("us-gaap:", "")
        batched_dataset.append(batch_qa_pairs)

    print(batched_dataset[0])
    # Write to JSONL file
    with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:

        for qa_pair in batched_dataset:
            json.dump(qa_pair, jsonl_file, ensure_ascii=False)
            jsonl_file.write('\n')  # Write each QA pair as a separate JSON object on a new line


if __name__ == '__main__':
    csv_file_path = 'test/test_sample.csv'  # Replace with the path to your CSV file
    output_jsonl_path = 'test/fnxl_test_batched.jsonl'  # Path for the output JSONL file
    examples_per_batch = 4  # Set the desired number of examples per batch

    create_batched_qa_dataset_from_csv_jsonl(csv_file_path, examples_per_batch, output_jsonl_path)

    print(f"Batched QA dataset saved to: {output_jsonl_path}")
