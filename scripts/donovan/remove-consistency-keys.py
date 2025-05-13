from pathlib import Path
from tqdm import tqdm

import json

keys_to_remove = [
        "eval_prompt_consistency",
        "P1_prompt_consistency_score",
        "P2_prompt_consistency_score",
        "eval_index_consistency",
        "P2_index_consistency_score",
        "P1_index_consistency_score",
        "P1_survey_consistency_score",
        "P2_survey_consistency_score",
        "P2_backstory_test",
        "P1_backstory_test",
        "eval_survey_consistency"
    ]

def remove_consistency_keys(data, filename):
    for dialog in tqdm(data, desc=f"Cleaning {filename}", unit="dialog"):
        for key in keys_to_remove:
            if key in dialog:
                del dialog[key]

def verify_keys_removed(data):
    for dialog in data:
        for key in keys_to_remove:
            if key in dialog:
                print(f"Key '{key}' still exists in dialog: {dialog}")
                return False
    return True

def clean_all_jsons_in_directory(directory):
    for json_file in Path(directory).glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        remove_consistency_keys(data, filename=json_file.name)

        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Cleaned {json_file.name} and removed consistency keys.")

    for json_file in Path(directory).glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        if not verify_keys_removed(data):
            print(f"Error: Some keys were not removed from {json_file.name}")
            return False
        else:
            print(f"All keys removed successfully from {json_file.name}")

    return True



def main(directory):

    # Specify the directory containing the JSON files
    directory = Path(directory)

    # Clean all JSON files in the specified directory
    output = clean_all_jsons_in_directory(directory)

    if output:
        print("All JSON files cleaned successfully.")
    else:
        print("Some JSON files could not be cleaned successfully.")


if __name__ == "__main__":
    # directory = "/mmfs1/home/donoclay/cse/donoclay/consistency_LLMs/training_data/qwen-14b-evals/in"
    # directory = "/mmfs1/home/donoclay/cse/donoclay/consistency_LLMs/training_data/qwen-14b-evals/in_chatting"
    # directory = "/mmfs1/home/donoclay/cse/donoclay/consistency_LLMs/training_data/qwen-14b-evals/in_education"
    directory = "/mmfs1/home/donoclay/cse/donoclay/consistency_LLMs/training_data/qwen-14b-thinking-evals/in_chatting"
    main(directory)
