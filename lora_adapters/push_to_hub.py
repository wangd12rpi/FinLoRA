import os
from huggingface_hub import HfApi


def remove_readme_metadata(lora_model_folder_path):
    """
    Checks for a README.md file in the given folder, and if found,
    removes the YAML frontmatter metadata block (enclosed by '---' lines).
    """
    readme_path = os.path.join(lora_model_folder_path, "README.md")

    if not os.path.exists(readme_path):
        print(f"      No README.md found in {lora_model_folder_path}. Skipping metadata removal.")
        return

    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            print(f"      README.md in {lora_model_folder_path} is empty. Nothing to modify.")
            return

        # Check if the first line (stripped of whitespace) is '---'
        if lines[0].strip() == "---":
            # Find the end of the metadata block (the next '---' line)
            end_metadata_index = -1
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    end_metadata_index = i
                    break

            if end_metadata_index != -1:
                # Content starts after the closing '---'
                content_after_metadata = lines[end_metadata_index + 1:]

                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.writelines(content_after_metadata)
                print(f"      Metadata successfully removed from README.md in {lora_model_folder_path}.")
            else:
                # Started with '---' but no closing '---' found.
                # To be safe, we won't modify the file in this ambiguous case.
                print(
                    f"      README.md in {lora_model_folder_path} starts with '---' but no closing '---' delimiter was found. File not modified.")
        else:
            print(
                f"      README.md in {lora_model_folder_path} does not start with a '---' metadata block. File not modified.")

    except Exception as e:
        print(f"      Error processing README.md in {lora_model_folder_path}: {e}")


def upload_lora_models_to_hub():
    api = HfApi()
    current_directory = os.getcwd()
    print(f"Scanning for LoRA models in subfolders of: {current_directory}")
    print("--------------------------------------------------")

    for parent_folder_name in os.listdir(current_directory):
        parent_folder_path = os.path.join(current_directory, parent_folder_name)
        if "misc" in parent_folder_name:
            continue
        if os.path.isdir(parent_folder_path):
            print(f"Processing parent folder: {parent_folder_name}")

            for lora_model_folder_name in os.listdir(parent_folder_path):
                lora_model_path = os.path.join(parent_folder_path, lora_model_folder_name)

                if os.path.isdir(lora_model_path):
                    model_repo_name = lora_model_folder_name

                    print(f"  Found LoRA model folder: {lora_model_folder_name}")
                    print(f"    Path: {lora_model_path}")

                    remove_readme_metadata(lora_model_path)

                    print(f"    Attempting to upload as Hugging Face Hub repo: '{model_repo_name}'")
                    repo_url = api.create_repo(
                        repo_id=model_repo_name,
                        repo_type="model",
                        exist_ok=True
                    )
                    print(f"    Repo '{model_repo_name}' ensured/created on Hugging Face Hub: {repo_url}")

                    api.upload_folder(
                        folder_path=lora_model_path,
                        repo_id="wangd12/" + model_repo_name,
                        repo_type="model",
                        commit_message=f"Add/update LoRA model: {model_repo_name}"
                    )
                    print(
                        f"    Successfully uploaded files from '{lora_model_folder_name}' to '{model_repo_name}'.")

                print("--------------------------------------------------")


print("All relevant folders processed.")

if __name__ == "__main__":
    print("Hugging Face LoRA Model Uploader Script (with README.md metadata removal)")
    print("========================================================================")
    print("IMPORTANT:")
    print("1. Ensure 'huggingface-cli login' has been run.")
    print("2. Script processes folders in the current directory, then their subfolders.")
    print("3. For each model subfolder, it attempts to remove metadata from README.md.")
    print("4. The model subfolder name becomes the Hugging Face repository name (YOUR_USERNAME/subfolder_name).")
    print("5. Existing repositories will be updated.")
    print("========================================================================")

    upload_lora_models_to_hub()
