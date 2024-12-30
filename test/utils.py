import xml.etree.ElementTree as ET

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
                if "TextBlock" in ele:
                    continue

                ele = ele.replace('xmlns:us-gaap="http://fasb.org/us-gaap/2023"', "").replace(f'contextRef="{context_id}"', "")
                
                matching_elements.append(ele)

        return "".join(matching_elements)

    except FileNotFoundError:
        print(f"Error: XML file not found: {xml_file}")
        return ""


def add_xml(qa_string, limit=1000000):
    if '<' not in qa_string or ',id:' not in qa_string:
        return qa_string

    # Extract information from the QA string
    start = qa_string.find("<") + 1
    end = qa_string.find(">")
    placeholder = qa_string[start:end]
    parts = placeholder.split(",id:")
    doc_path = "../xbrl/DowJones30/" + parts[0]
    
    context_id = parts[1]

    # Get the XML content using the custom grep function
    xml_content = find_elements_by_context_ref(doc_path, context_id)[:limit]

    # Replace the placeholder with the XML content
    new_qa_string = qa_string.replace(f"<{placeholder}>", xml_content + "\n\n")
    return new_qa_string



if __name__ == '__main__':
    c = add_xml("""
    You are a knowledgeable XBRL assistant that can answer questions based on XML data. You will be provided with a context extracted from an XBRL file and a question related to it. The example question can help you to learn the format of the answer. Your task is to analyze the XBRL context and provide an accurate and very concise answer to the question, don't output xml, code, explanation, or create new question. XBRL file: <msft-20230630/msft-20230630_htm.xml,id:C_da7a1266-ce2b-41b7-83ff-bb6e03dc1fa5> Example question: What is the US GAAP XBRL tag for Cash and Cash Equivalents as reported by Example Company Inc for the Fiscal Year ending in FY 2022 Example answer: us-gaap:AnExampleTagName Question: What is the US GAAP XBRL tag for Cash Flow for Financing Activities as reported by Microsoft Corp for the Fiscal Year ending in FY 2023? Answer:
    """)
    print(len(c))