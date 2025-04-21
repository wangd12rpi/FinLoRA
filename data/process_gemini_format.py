import os
import json
import shutil

def process_to_gemini_format(input_dir="train", output_dir_suffix="gemini"):
    output_dir = os.path.join(input_dir, output_dir_suffix)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            input_filepath = os.path.join(input_dir, filename)
            base_name, ext = os.path.splitext(filename)
            output_filename = f"{base_name}_gemini{ext}"
            output_filepath = os.path.join(output_dir, output_filename)

            with open(input_filepath, 'r', encoding='utf-8') as infile, \
                 open(output_filepath, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    try:
                        data = json.loads(line.strip())
                        context = data.get("context", "")
                        target = data.get("target", "")

                        gemini_data = {
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [{"text": context}]
                                },
                                {
                                    "role": "model",
                                    "parts": [{"text": target}]
                                }
                            ]
                        }
                        outfile.write(json.dumps(gemini_data) + '\n')
                    except json.JSONDecodeError:
                        pass # Skip malformed lines silently


# --- Script execution ---
# Assumes 'train' directory exists and contains .jsonl files.
if __name__ == "__main__":
    input_directory = "train"

    if os.path.isdir(input_directory):
        print(f"Processing files in '{input_directory}'...")
        process_to_gemini_format(input_dir=input_directory)
        print(f"Processing complete. Output files are in '{os.path.join(input_directory, 'gemini')}'")
    else:
        # Although no error handling was requested for file processing,
        # checking the input dir existence is a basic prerequisite.
        print(f"Error: Input directory '{input_directory}' not found.")
        print("Please ensure the 'train' directory exists and contains your .jsonl files.")