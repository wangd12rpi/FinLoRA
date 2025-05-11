import pandas as pd
import json
import requests
from sklearn.model_selection import train_test_split
from google.cloud import vision
import io
import time

from tqdm import tqdm


def process_xbrl_term():
    try:
        df = pd.read_excel("preprocessed/xbrl_term.xlsx")
    except FileNotFoundError:
        print(f"Error: The file xbrl term was not found.")
        return
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return

    if df.shape[1] < 2:
        print("Error: The Excel file must have at least two columns.")
        return

    # df = df.dropna(subset=['C0', 'C1'])

    if df.empty:
        print("Error: No valid data found in the specified columns after removing empty rows.")
        return

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

    def df_to_jsonl(dataframe, output_file_path):
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for _, row in dataframe.iterrows():
                json_record = {
                    "context": "Explain this XBRL term briefly in one sentence: " + str(row['Term']) + ". Answer:",
                    "target": str(row['Explanation'])
                }
                f.write(json.dumps(json_record) + '\n')
        print(f"Successfully saved {len(dataframe)} records to {output_file_path}")

    df_to_jsonl(train_df, "train/xbrl_term_train.jsonl")
    df_to_jsonl(test_df, "test/xbrl_term_test.jsonl")


def process_finance_bench():
    def parse_page_numbers(page_number_str, max_pages=3):
        if page_number_str == "130132":
            page_number_str = "130,132"
        elif page_number_str == "302304":
            page_number_str = "302,304"
        elif page_number_str == "108,110":
            page_number_str = "108,110"
        elif page_number_str == "173173174":
            page_number_str = "173,174"

        if pd.isna(page_number_str) or not isinstance(page_number_str, str) or page_number_str.strip() == "":
            return []

        pages_processed = []
        parts = page_number_str.split(',')
        for part in parts:
            try:
                page = int(part.strip())
                if page > 0:  # Page numbers are typically 1-indexed
                    pages_processed.append(page)
            except ValueError:
                # Handle cases like 'Summary' or ranges if necessary, for now, skip non-integers
                print(f"Warning: Could not parse page number part: {part}")
                continue
            if len(pages_processed) >= max_pages:
                break
        return sorted(list(set(pages_processed)))[:max_pages]

    def ocr_pdf_pages(pdf_content_bytes, pages_to_ocr, vision_client):
        if not pdf_content_bytes or not pages_to_ocr:
            return None

        try:
            input_config = vision.InputConfig(
                content=pdf_content_bytes,
                mime_type="application/pdf"
            )
            features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)]

            # Ensure pages are within typical PDF limits, though Vision API might have its own
            # For very large page numbers, this might be an issue, but typically documents are smaller.
            request = vision.AnnotateFileRequest(
                input_config=input_config,
                features=features,
                pages=pages_to_ocr  # Specify pages here
            )

            response = vision_client.batch_annotate_files(requests=[request])

            page_texts = []
            # The outer response is for the batch (one file in this case)
            # The inner responses are per page
            for file_response in response.responses:  # Should be one AnnotateFileResponse
                for i, page_response in enumerate(file_response.responses):  # AnnotateImageResponse per page
                    if page_response.full_text_annotation:
                        page_text = page_response.full_text_annotation.text
                        page_texts.append(f"Page {pages_to_ocr[i]}:\n{page_text}\n")
                    else:
                        # Log if a specific page had no text or an error
                        print(
                            f"Warning: No text found or error processing page {pages_to_ocr[i]}. Error: {page_response.error.message if page_response.error else 'Unknown'}")

            return "\n".join(page_texts) if page_texts else None

        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return None

    try:
        df = pd.read_excel("preprocessed/financebench.xlsx")
    except FileNotFoundError:
        print(f"Error: The file was not found.")
        return
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return

    required_columns = ['doc_link', 'page_number', 'question', 'answer']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}' in the Excel file.")
            return

    df_filtered = df[required_columns].copy()
    # df_filtered = df_filtered.sample(2)
    df_filtered.dropna(subset=['doc_link', 'question', 'answer'], inplace=True)

    if df_filtered.empty:
        print("Error: No valid data after filtering for essential columns (doc_link, question, answer).")
        return

    processed_data = []
    vision_client = vision.ImageAnnotatorClient()  # Initialize client once

    print(f"Starting processing of {len(df_filtered)} rows...")
    for index, row in tqdm(df_filtered.iterrows()):
        doc_link = row['doc_link']
        page_number_str = str(row['page_number'])  # Ensure it's a string
        question = str(row['question'])
        answer = str(row['answer'])

        pages_to_ocr = parse_page_numbers(page_number_str)

        if not pages_to_ocr:
            print(f"Warning: No valid page numbers to OCR for row {index}. Skipping PDF processing for this row.")
            # Decide if you want to include rows without OCR context
            # For now, we'll skip if no pages are specified, as context is crucial
            continue

        pdf_content_bytes = None
        try:
            response = requests.get(doc_link, timeout=30)  # Added timeout
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            pdf_content_bytes = response.content
            print(f"  PDF downloaded successfully ({len(pdf_content_bytes)} bytes).")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading PDF from {doc_link}: {e}. Skipping row {index}.")
            continue

        if pdf_content_bytes:
            time.sleep(1)
            extracted_text = ocr_pdf_pages(pdf_content_bytes, pages_to_ocr, vision_client)

            if extracted_text:
                print(f"  OCR successful. Text length: {len(extracted_text)}.")
                context = f"Answer the following question very briefly using scanned text from pages. No explanation needed. Question: {question}\nDocument Pages Context:\n{extracted_text}\nAnswer:"
                processed_data.append(
                    {"context": context, "target": str(answer), "url": doc_link, "pages": pages_to_ocr})
            else:
                print(
                    f"Warning: OCR processing failed or returned no text for PDF from {doc_link} for pages {pages_to_ocr}. Skipping row {index}.")
        else:
            # This case should be caught by the download error handling, but as a safeguard:
            print(f"Warning: PDF content was empty for {doc_link}. Skipping row {index}.")

    if not processed_data:
        print("No data was successfully processed after OCR. Output files will be empty or not created.")
        return

    print(f"\nSuccessfully processed {len(processed_data)} items with OCR context.")

    if len(processed_data) < 2:  # Need at least 2 samples to split
        print("Warning: Not enough data to perform a train/test split. Saving all to train file.")
        train_data = processed_data
        test_data = []
    else:
        train_data, test_data = train_test_split(processed_data, test_size=(1 / 3), random_state=42, shuffle=True)

    def save_to_jsonl(data, file_path_out):
        if not data:
            print(f"No data to save to {file_path_out}.")
            return
        with open(file_path_out, 'w', encoding='utf-8') as f:
            for record in data:
                f.write(json.dumps(record) + '\n')
        print(f"Successfully saved {len(data)} records to {file_path_out}")

    save_to_jsonl(train_data, "train/financebench_train.jsonl")
    save_to_jsonl(test_data, "test/financebench_test.jsonl")
    print("Processing complete.")


def process_formula_data_xlsx():
    # Processes an XLSX file containing formula names, and multiple question/answer pairs per formula.
    # Creates JSONL training (first 16 Q/A) and testing (last 4 Q/A) sets.
    def _save_to_jsonl(data, file_path_out, data_type_name="data"):
        if not data:
            print(f"No {data_type_name} to save to {file_path_out}.")
            return
        try:
            with open(file_path_out, 'w', encoding='utf-8') as f:
                for record in data:
                    f.write(json.dumps(record) + '\n')
            print(f"Successfully saved {len(data)} records to {file_path_out} ({data_type_name})")
        except IOError as e:
            print(f"Error saving {data_type_name} to {file_path_out}: {e}")

    file_path = "preprocessed/formula.xlsx"
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return

    if 'Formula Name' not in df.columns:
        print("Error: Missing required column 'Formula Name' in the Excel file.")
        return

    train_data = []
    test_data = []
    num_questions_per_formula = 20
    num_train_questions = 16

    print(f"Starting processing of {len(df)} rows for formula data...")
    for index, row in df.iterrows():
        formula_name = row.get('Formula Name')
        formula = row.get('Formula')
        explain = row.get('Explanation')
        if pd.isna(formula_name):
            print(f"Warning: Skipping row {index + 1} due to missing 'Formula Name'.")
            continue

        print(f"  Processing formula: {formula_name} (Row {index + 1})")

        for i in range(1, num_questions_per_formula + 1):
            question_col = f"Question {i}"
            answer_col = f"Answer {i}"

            if question_col not in df.columns or answer_col not in df.columns:
                print(
                    f"Warning: Missing '{question_col}' or '{answer_col}' for formula '{formula_name}'. Skipping this Q/A pair.")
                continue

            question_text = row.get(question_col)
            answer_text = row.get(answer_col)

            if pd.isna(question_text) or pd.isna(answer_text) or str(question_text).strip() == "" or str(
                    answer_text).strip() == "":
                # print(f"Info: Skipping Q/A pair {i} for formula '{formula_name}' due to missing question or answer.")
                continue

            context = f"Use formula {formula_name} to answer the question. Answer with a numerical answer with 2 decimal places and with no explanations or other text. Formula: {formula}, Explanation: {explain}. Question: {str(question_text[3:])}. Answer:"
            target = str(answer_text)

            record = {"context": context, "target": target}

            if i <= num_train_questions:
                train_data.append(record)
            else:
                test_data.append(record)

    if not train_data and not test_data:
        print("No data was successfully processed for formula data. Output files will be empty or not created.")
        return

    print(
        f"\nSuccessfully processed formula data. Total training samples: {len(train_data)}, Total testing samples: {len(test_data)}")

    _save_to_jsonl(train_data, "train/formula_train.jsonl", "formula train")
    _save_to_jsonl(test_data, "test/formula_test.jsonl", "formula test")
    print("Formula data processing complete.")


if __name__ == '__main__':
    # process_xbrl_term()
    # process_finance_bench()

    process_formula_data_xlsx()
