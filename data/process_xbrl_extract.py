import xml.etree.ElementTree as ET
import re
import json
from typing import List, Dict
from tqdm import tqdm
import random
import os.path
import datasets

random.seed(42)
all_train = []


def find_elements_by_context_ref(xml_f_name, context_id):
    try:
        if not os.path.isfile(xml_f_name):
            print(f"missing file {xml_f_name}")
            xml_f_name = xml_f_name.replace("_htm", "")

        tree = ET.parse(xml_f_name)
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

                matching_elements.append(ele)

        xml_filtered = "".join(matching_elements)
        xml_filtered = xml_filtered.replace(" ", "").replace("xmlns:us-", "")
        return xml_filtered

    except FileNotFoundError:
        print(f"Error: XML file not found: {xml_f_name}")
        return ""


def gen_xbrl(cat):
    global all_train
    for split in ["train", "test"]:
        data = datasets.load_dataset("wangd12/XBRL_extraction", cat)[split]
        processed_data = []

        if split == "train" and (cat == "tags" or cat == "value"):  # tags and value have larger train split
            data = data.select(range(len(data) // 3))

        for entry in tqdm(data):
            doc_path = "train/DowJones30/" + entry["doc_path"]
            context_id = entry["context_id"]

            # Get the XML content using the custom grep function
            xml_content = find_elements_by_context_ref(doc_path, context_id)

            prompt = entry["instruction"] + "\nXML File: " + xml_content + "\n" + entry["input"]
            processed_data.append({"context": prompt, "target": str(entry["output"])})

        if split == "test":
            with open(f"test/xbrl_extract_{cat}_test.jsonl", "w") as f:
                print(cat, split, "length:", len(processed_data))
                for example in processed_data:
                    f.write(json.dumps(example) + "\n")
        else:  # train
            print(cat, split, "added to train, length:", len(processed_data))
            all_train += processed_data


if __name__ == '__main__':
    gen_xbrl("tags")
    gen_xbrl("value")
    gen_xbrl("formula")
    gen_xbrl("formula_calculations")

    sorted_train = sorted(
        all_train,
        key=lambda item: len(str(item.get('context', ''))),
        reverse=True
    )

    sorted_train = sorted_train[int(len(sorted_train) * .1):]
    print("longest train", len(sorted_train[0]['context']))

    with open(f"train/xbrl_extract_train.jsonl", "w") as f:
        print("train", "total length:", len(sorted_train))
        for example in sorted_train:
            f.write(json.dumps(example) + "\n")
