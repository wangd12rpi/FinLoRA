import json
import pathlib
import tiktoken # Changed from transformers

def count_lines_in_file(file_path):
    """Counts the number of lines in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def calculate_average_context_tokens(file_path, tokenizer):
    """
    Calculates the average number of tokens in the 'context' field of a JSONL file.
    Processes all lines in the file.
    """
    lines_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                lines_data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {file_path}: {line.strip()}")
                continue

    if not lines_data:
        return 0 # No data

    total_tokens = 0
    valid_contexts = 0
    for item in lines_data: # Process all lines
        context = item.get("context")
        if isinstance(context, str): # Ensure context is a string
            tokens = tokenizer.encode(context) # tiktoken's encode method
            total_tokens += len(tokens)
            valid_contexts += 1
        else:
            print(f"Warning: Skipping line with missing or non-string 'context' in {file_path}: {item}")

    if valid_contexts == 0:
        return 0 # No valid contexts found

    return total_tokens / valid_contexts

def main():
    """
    Main function to process files in train/ and test/ directories.
    """
    base_path = pathlib.Path(".")  # Assumes script is run in the parent directory of train/ and test/
    train_path = base_path / "train"
    test_path = base_path / "test"

    # Initialize tiktoken tokenizer
    # "cl100k_base" is a common encoding used by OpenAI models and is generally fast.
    # For Llama 3.1, while it has its own specific tokenizer,
    # cl100k_base is often used with tiktoken for a close approximation if direct Llama support isn't available
    # or if the goal is simply a fast, good-quality BPE tokenizer.
    try:
        tokenizer_name = "cl100k_base"
        tokenizer = tiktoken.get_encoding(tokenizer_name)
        print(f"Successfully loaded tiktoken tokenizer: {tokenizer_name}\n")
    except Exception as e:
        print(f"Error loading tiktoken tokenizer '{tokenizer_name}': {e}")
        return

    for data_dir in [train_path, test_path]:
        print(f"--- Statistics for directory: {data_dir.name} ---")
        if not data_dir.exists() or not data_dir.is_dir():
            print(f"Directory {data_dir} does not exist or is not a directory. Skipping.")
            continue

        jsonl_files = sorted(list(data_dir.glob("*.jsonl"))) # Sort for consistent output

        if not jsonl_files:
            print("No .jsonl files found.")
            continue

        for file_path in jsonl_files:
            print(f"\n  File: {file_path.name}")
            try:
                num_lines = count_lines_in_file(file_path)
                print(f"    Number of lines: {num_lines}")

                if data_dir.name == "test":
                    if num_lines == 0:
                        print("    Average context tokens: N/A (empty file)")
                    else:
                        avg_tokens = calculate_average_context_tokens(file_path, tokenizer)
                        if num_lines > 0 and avg_tokens > 0 : # Check if there were valid contexts
                             print(f"    Average context tokens (all {num_lines} lines): {avg_tokens:.2f}")
                        elif num_lines > 0 and avg_tokens == 0:
                             print(f"    Average context tokens: N/A (no valid 'context' fields found in {num_lines} lines)")
                        else: # Should be covered by num_lines == 0
                            print(f"    Average context tokens: N/A (file might be empty or contain no processable lines)")


            except FileNotFoundError:
                print(f"    Error: File {file_path.name} not found.")
            except Exception as e:
                print(f"    An error occurred processing {file_path.name}: {e}")
        print("-" * (len(data_dir.name) + 28)) # Dynamic separator length

if __name__ == "__main__":

    main()
