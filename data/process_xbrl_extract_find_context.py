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


def find_context_xml(xml_f_name):
    try:
        if not os.path.isfile(xml_f_name):
            print(f"missing file {xml_f_name}")
            xml_f_name = xml_f_name.replace("_htm", "")


        if "aapl-2022" not in xml_f_name:
            return ""

        tree = ET.parse(xml_f_name)
        root = tree.getroot()

        matching_elements = []

        for context_element in root.iter():
            if "context" in context_element.tag:
                # print(context_element)
                ele = ET.tostring(context_element, encoding='unicode', method='text')
                ele = ele.replace('ns0:', "").replace("ns1:", "").replace("  ", "").replace("\n", "")
                ele = re.sub(r"</.*?>", "</>", ele)
                ele = re.sub(r'<identifier.*?</identifier>', '', ele,
                                      flags=re.DOTALL)
                ele = re.sub(r'xmlns.*?".*?"', '', ele,)

                print(ele)

                matching_elements.append(ele)


        xml_filtered = "".join(matching_elements)
        xml_filtered = (xml_filtered.replace(" ", "").replace("xmlns:us-", "")
                        .replace("<entity>", "").replace("</entity>", "")
                        .replace("<segment>", "").replace("</segment>", "")
                        )
        return xml_filtered

    except FileNotFoundError:
        print(f"Error: XML file not found: {xml_f_name}")
        return ""


def gen_xbrl(cat):
    global all_train
    for split in ["train"]:
        data = datasets.load_dataset("wangd12/XBRL_extraction", cat)[split].shuffle(seed=42)

        processed_data = []
        # data = data.select(range(len(data) // 2))
        # if split == "train" and (cat == "tags" or cat == "value"):  # tags and value have larger train split
        #     data = data.select(range(len(data) // 3))

        for entry in tqdm(data):
            doc_path = "train/DowJones30/" + entry["doc_path"]
            context_id = entry["context_id"]

            # Get the XML content using the custom grep function
            xml_content = find_context_xml(doc_path)

            prompt = ("You are XBRL expert. You should find the correct context id from the xml segment according to the question.  Use the axis and time info to see which context match the question. The question is: '" +
            entry["input"] + "'\nXML File: " + xml_content + "\n" + "Output the context id and nothing else. Answer:")
            processed_data.append({"context": prompt, "target": entry["context_id"]})

        # if split == "test":
        #     with open(f"test/xbrl_extract_{cat}_test.jsonl", "w") as f:
        #         print(cat, split, "length:", len(processed_data))
        #         for example in processed_data:
        #             f.write(json.dumps(example) + "\n")


        print(cat, split, "added to train, length:", len(processed_data))
        all_train += processed_data


if __name__ == '__main__':
    gen_xbrl("tags")
    # gen_xbrl("value")
    # gen_xbrl("formula")
    # gen_xbrl("formula_calculations")

    sorted_train = sorted(
        all_train,
        key=lambda item: len(str(item.get('context', ''))),
        reverse=True
    )

    sorted_train = sorted_train[int(len(sorted_train) * .1):]
    print("longest train", len(sorted_train[0]['context']))

    with open(f"train/xbrl_extract_find_context_train.jsonl", "w") as f:
        print("train", "total length:", len(sorted_train))
        for example in sorted_train:
            f.write(json.dumps(example) + "\n")
