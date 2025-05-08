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
                context_id = context_element.attrib["id"]

                if context_id is None:
                    continue

                current_text = f"context_id: {context_id}"

                # --- Segment and Explicit Member (first one) ---
                dim_info_str = None
                # Path is relative to the current context_element
                segment_elem = context_element.find("ns0:entity/ns0:segment")

                print(ET.tostring(segment_elem))
                if segment_elem is not None:
                    first_explicit_member = None
                    for child in segment_elem:
                        # The tag name 'xbrldi:explicitMember' is used literally, as ElementTree
                        # will treat it as such if namespace prefixes are not resolved during parsing.
                        if child.tag == 'xbrldi:explicitMember':
                            first_explicit_member = child
                            break  # Interested in the first one only

                    if first_explicit_member is not None:
                        dimension_attr = first_explicit_member.get("dimension")
                        member_text_content = first_explicit_member.text

                        member_text = member_text_content.strip() if member_text_content else None

                        # The example format "dimension: [attr], [text]" implies both are needed.
                        if dimension_attr and member_text:
                            dim_info_str = f"dimension: {dimension_attr}, {member_text}"
                        # If only one part is present, we could choose to represent it,
                        # but sticking to the example's paired format for this part.

                if dim_info_str:
                    current_text += f", {dim_info_str}"

                # --- Period Information ---
                period_info_str = None
                period_elem = context_element.find("./period")  # Path relative to context_element
                if period_elem is not None:
                    instant_elem = period_elem.find("./instant")
                    if instant_elem is not None and instant_elem.text:
                        period_info_str = f"period: instant:{instant_elem.text.strip()}"
                    else:
                        start_date_elem = period_elem.find("./startDate")
                        end_date_elem = period_elem.find("./endDate")

                        start_date_text = start_date_elem.text.strip() if start_date_elem is not None and start_date_elem.text else None
                        end_date_text = end_date_elem.text.strip() if end_date_elem is not None and end_date_elem.text else None

                        if start_date_text and end_date_text:
                            period_info_str = f"period: {start_date_text} to {end_date_text}"
                        elif start_date_text:  # Only start date is present
                            period_info_str = f"period: startDate:{start_date_text}"
                        elif end_date_text:  # Only end date is present
                            period_info_str = f"period: endDate:{end_date_text}"

                if period_info_str:
                    current_text += f". {period_info_str}"  # Note the period before "period:"
                print(current_text)
                matching_elements.append(current_text)
                break

        xml_filtered = "\n".join(matching_elements)

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
