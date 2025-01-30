import xml.etree.ElementTree as ET
import re


def find_elements_by_context_ref(xml_file, context_id):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        matching_elements = []
        for element in root.iter():

            if element.get("contextRef") == context_id and "us-gaap" in element.tag:
                truncated_content = element.text[:100] if element.text else ""  # Truncate content
                element.text = truncated_content

                ele = ET.tostring(element, encoding="unicode").replace("ns0", "us-gaap")
                if "TextBlock" in ele or "style=" in ele:
                    continue

                ele = ele.replace('xmlns:us-gaap="http://fasb.org/us-gaap/2023"', "").replace(
                    f'contextRef="{context_id}"', "")
                ele = re.sub(r"</.*?>", "</>", ele)  # Remove closing tag text (to reduce token count)
                ele = re.sub(r"\w+=\".*?\"", "", ele)  # Remove attributes
                ele = re.sub(r"\s+", " ", ele)  # Remove consecutive spaces

                matching_elements.append(ele)

        return "\n".join(matching_elements)

    except FileNotFoundError:
        print(f"Error: XML file not found: {xml_file}")
        return ""


# replace the file name with xbrl raw text
def add_xml(qa_string, limit=1000000):
    if '<' not in qa_string or ',id:' not in qa_string:
        return qa_string

    # Extract information from the QA string
    start = qa_string.find("<") + 1
    end = qa_string.find(">")
    placeholder = qa_string[start:end]
    parts = placeholder.split(",id:")
    doc_path = "DowJones30/" + parts[0]

    context_id = parts[1]

    # Get the XML content using the custom grep function
    xml_content = find_elements_by_context_ref(doc_path, context_id)[:limit]

    # Replace the placeholder with the XML content
    new_qa_string = qa_string.replace(f"<{placeholder}>", xml_content + "\n\n")
    return new_qa_string


# %%
import json
from typing import List, Dict
from tqdm import tqdm
import re
import random
import os.path


def get_xbrl_dataset(data: List[Dict], example_q=None, example_a=None):
    """
    Saves entries with matching category1 or category2 in the format for fine-tuning.

    Args:
        data (List[Dict]): The input JSON data.
        category (str): The category name to match.
        output_file (str): The output file path.
    """

    results = {}
    for entry in tqdm(data):
        if (entry["doc_path"], entry["answer"], entry["contextID"][0]) in results.keys():
            continue

        question = entry["query"]
        question = re.sub(r"\(.*?\)", "", question)
        doc_path = entry["doc_path"]
        context_ids = entry["contextID"]

        if not os.path.isfile('DowJones30/' + doc_path):
            # print(f"missing file {doc_path}")
            continue

        example_qa = ""
        if example_q != None and example_a != None:
            example_qa = f"\nExample question: {example_q}\nExample answer: {example_a}"

        context = \
            f""""You are a knowledgeable XBRL assistant that can answer questions based on XML data. 
             You will be provided with a context extracted from an XBRL file and a question related to it. The example question can help you to learn the format of the answer.
             Your task is to analyze the XBRL context and provide an accurate and very concise answer to the question, DO NOT output xml, code, explanation or create new question.
            \nXBRL file:\n ```xml\n <{doc_path},id:{context_ids[0]}> ```\n
            {example_qa}
            \nQuestion: {question}
            \nAnswer:"""

        context_xml = add_xml(context)
        if len(context_xml) > 24000:
            continue

        target = entry["raw_answer"]
        # print(entry["answer"])
        # entry["doc_path"], entry["answer"], entry["contextID"][0]
        results[entry["doc_path"], entry["answer"], entry["contextID"][0]] = {"context": context_xml,
                                                                              "target": str(target),
                                                                              "doc_path": entry['doc_path']}

    print("final length", len(results))
    return list(results.values())


def gen_xbrl(cat, example_q, example_a):
    with open("../xbrl/data/XBRL.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        filtered_data = [entry for entry in data if entry['category1'] == cat or entry['category2'] == cat]
        all_doc_path = list(set([entry['doc_path'] for entry in filtered_data]))
        print(f"Total data size for this {cat}: {len(filtered_data)}, total number of filings {len(all_doc_path)}")
        random.shuffle(filtered_data)

        # train_data = filtered_data[split_size:]
        # train_data = train_data

        dataset = get_xbrl_dataset(filtered_data[:2500], example_q, example_a)
        dataset = dataset[:1500]
        test_data = []
        train_data = []
        random.shuffle(all_doc_path)
        for x in all_doc_path:
            portion = [entry for entry in dataset if entry["doc_path"] == x]
            if len(test_data) < 100:
                test_data += portion
            else:
                train_data += portion

        return train_data, test_data


# %%
tags_train, tags_test = gen_xbrl("xbrl_tags",
                                 example_q="What is the US GAAP XBRL tag for Cash and Cash Equivalents as reported by Example Company Inc for the Fiscal Year ending in FY 2022",
                                 example_a="us-gaap:AnExampleTagName")