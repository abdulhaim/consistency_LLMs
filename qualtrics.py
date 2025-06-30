import json
import random
from pathlib import Path
import html

def load_data(base_path: Path, task_name, filter_str):
    """Loads the JSON data from the specified file path."""
    file_path: Path = Path()
    if task_name == "Chatting Nonfinetuned":
        file_path = base_path / "data/chatting/nonfinetuned"
    elif task_name == "Education Nonfinetuned":
        file_path = base_path / "data/education/nonfinetuned"
    elif task_name == "Therapy Nonfinetuned":
        file_path = base_path / "data/therapy/nonfinetuned"
    elif task_name == "Chatting Finetuned":
        file_path = base_path / "data/chatting/finetuned"
    elif task_name == "Education Finetuned":
        file_path = base_path / "data/education/finetuned"
    elif task_name == "Therapy Finetuned":
        file_path = base_path / "data/therapy/finetuned"
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    
    if not file_path.exists():
        raise ValueError(f"Error: File {file_path} does not exist.")
    
    # iterate through all json files in the directory
    data = []
    file_paths = []

    if ".json" in file_path.name:
        files = [file_path] 
    else:
        files = []
        for file in file_path.glob("*.json"):
            if filter_str in file.name:
                files.append(file)

    for file in sorted(files):
        with open(file, 'r') as f:
            try:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)

                file_paths.append(str(file))
                print(f"Loaded {len(file_data)} records from {file.name}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file}: {e}")
                continue
    if not data:
        print(f"Warning: No data loaded from {file_path}. Please check the files.")
        return None

    return data, file_paths

def select_random_dialogs(data, num_dialogs=10, seed=42):
    """Selects a specified number of random dialogs from the dataset using a seed and returns their indices."""
    random.seed(seed)
    indices = random.sample(range(len(data)), num_dialogs)
    sampled_dialogs = [data[i] for i in indices]
    return sampled_dialogs, indices

def convert_python_escapes_to_html(s):
    decoded = bytes(s, 'utf-8').decode('unicode_escape')
    # decoded = s.encode('utf-8').decode('unicode_escape')
    
    # Replace newline characters with <br>
    decoded = decoded.replace('\n', '<br>')
    
    # Optional: convert each character to its HTML numeric code (e.g., &#1593;)
    html_encoded = ''.join(f'&#{ord(c)};' if ord(c) > 127 else c for c in decoded)
    
    html_encoded.replace("â\x80\x93", "–")  # Replace the long dash with a single HTML entity
    return html_encoded

def generate_qualtrics_survey(output_txt_path, data=None, input_json_path=None):
    if not data:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    lines = []

    # Header required by Qualtrics
    lines.append("[[AdvancedFormat]]")

    # Start block
    lines.append("[[Block:Conversation Consistency Check]]")

    question_number = 1

    for convo in data:
        conversation = convo.get("conversation", [])
        persona = convo.get("P2", "")
        # lines.append("[[Text]]")
        lines.append("[[Question:DB]]")
        lines.append(f"Scenario Description<br>")
        lines.append("This conversation concerns a scenario between two speakers. "
                     "Please read the following exchange and answer questions about whether "
                     "the current line is consistent with the following description and the prior dialogue. <br><br>")
        lines.append(f"<b>Persona:</b> {convert_python_escapes_to_html(persona)}")

        # Randomly select a starting point (odd index, since we skip the other agent)
        if len(conversation) < 10:
            continue  # Not enough lines for 5 steps
        possible_starts = list(range(1, len(conversation) - 8, 2))
        if not possible_starts:
            continue
        start_idx = random.choice(possible_starts)
        for i in range(start_idx, min(start_idx + 10, len(conversation)), 2):
            previous_lines = ""
            for line in conversation[:i]:
                prev_name, prev_message = line[1].split(":", 1)
                previous_lines += f"<b>{prev_name}:</b> {convert_python_escapes_to_html(prev_message)}<br>"

            current_line = conversation[i][1]
            name, message = current_line.split(":", 1)
            current_line = f"<b>{name}:</b> {convert_python_escapes_to_html(message)}<br>"

            lines.append(f"[[Question:MC:SingleAnswer:Horizontal]]")
            lines.append(f"<b>Conversation so far:</b><br>{previous_lines}<br><br>")
            lines.append(f"<b>Current line:</b><br>{current_line}<br><br>")
            lines.append("How consistent is the current line with the persona description and conversation?")
            lines.append("[[Choices]]")
            lines.append("1")
            lines.append("2")
            lines.append("3")
            lines.append("4")
            lines.append("5")
            lines.append("6")

            question_number += 1
    #     lines.append("[[PageBreak]]")

        # for i in range(1, len(conversation), 2):

        #     previous_lines = ""
        #     for line in conversation[:i]:
        #         prev_name, prev_message = line[1].split(":", 1)
        #         previous_lines += f"<b>{prev_name}:</b> {prev_message}<br>"

        #     current_line = conversation[i][1]
        #     name, message = current_line.split(":", 1)
        #     current_line = f"<b>{name}:</b> {message}<br>"

        #     lines.append(f"[[Question:MC:SingleAnswer:Horizontal]]")
        #     lines.append(f"<b>Conversation so far:</b><br>{convert_python_escapes_to_html(previous_lines)}<br><br>")
        #     lines.append(f"<b>Current line:</b><br>{convert_python_escapes_to_html(current_line)}<br><br>")
        #     lines.append("How consistent is the current line with the persona description and conversation?")
        #     lines.append("[[Choices]]")
        #     lines.append("1")
        #     lines.append("2")
        #     lines.append("3")
        #     lines.append("4")
        #     lines.append("5")
        #     lines.append("6")

        #     question_number += 1
        lines.append("[[PageBreak]]")  # Optional; separates each question
    # Write to UTF-8 without BOM
    with open(output_txt_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(lines))

    print(f"✅ Qualtrics survey saved to: {output_txt_path}")

# Load three conversations from each model for each scenario using the filter string
# Also keep the indices and write them to a separate json with the filename

base_path = Path("/nfs/kun2/users/ryan_cheng/consistency_LLMs")

indices_record = {}

# Chatting Nonfinetuned
chatting_gemma, chatting_gemma_files = load_data(base_path, "Chatting Nonfinetuned", "gemma")
chatting_mistral, chatting_mistral_files = load_data(base_path, "Chatting Nonfinetuned", "mistral")
chatting_llama, chatting_llama_files = load_data(base_path, "Chatting Nonfinetuned", "Llama-3.1-8B-Instruct")

chatting_gemma_sel, chatting_gemma_idx = select_random_dialogs(chatting_gemma, num_dialogs=1, seed=42)
chatting_mistral_sel, chatting_mistral_idx = select_random_dialogs(chatting_mistral, num_dialogs=1, seed=42)
chatting_llama_sel, chatting_llama_idx = select_random_dialogs(chatting_llama, num_dialogs=1, seed=42)

indices_record["chatting_gemma"] = {"files": chatting_gemma_files, "indices": chatting_gemma_idx}
indices_record["chatting_mistral"] = {"files": chatting_mistral_files, "indices": chatting_mistral_idx}
indices_record["chatting_llama"] = {"files": chatting_llama_files, "indices": chatting_llama_idx}

# Chatting Finetuned
chatting_llama_ppo, chatting_llama_ppo_files = load_data(base_path, "Chatting Finetuned", "llama-8b-ppo-high-lr")
chatting_llama_sftppo, chatting_llama_sftppo_files = load_data(base_path, "Chatting Finetuned", "llama-8b-sft-ppo")

chatting_llama_ppo_sel, chatting_llama_ppo_idx = select_random_dialogs(chatting_llama_ppo, num_dialogs=1, seed=42)
chatting_llama_sftppo_sel, chatting_llama_sftppo_idx = select_random_dialogs(chatting_llama_sftppo, num_dialogs=1, seed=42)

indices_record["chatting_llama_ppo"] = {"files": chatting_llama_ppo_files, "indices": chatting_llama_ppo_idx}
indices_record["chatting_llama_sftppo"] = {"files": chatting_llama_sftppo_files, "indices": chatting_llama_sftppo_idx}

# Education Nonfinetuned
education_gemma, education_gemma_files = load_data(base_path, "Education Nonfinetuned", "gemma")
education_mistral, education_mistral_files = load_data(base_path, "Education Nonfinetuned", "mistral")
education_llama, education_llama_files = load_data(base_path, "Education Nonfinetuned", "Llama-3.1-8B-Instruct")

education_gemma_sel, education_gemma_idx = select_random_dialogs(education_gemma, num_dialogs=1, seed=42)
education_mistral_sel, education_mistral_idx = select_random_dialogs(education_mistral, num_dialogs=1, seed=42)
education_llama_sel, education_llama_idx = select_random_dialogs(education_llama, num_dialogs=1, seed=42)

indices_record["education_gemma"] = {"files": education_gemma_files, "indices": education_gemma_idx}
indices_record["education_mistral"] = {"files": education_mistral_files, "indices": education_mistral_idx}
indices_record["education_llama"] = {"files": education_llama_files, "indices": education_llama_idx}

# Education Finetuned
education_ppo_high_lr, education_ppo_high_lr_files = load_data(base_path, "Education Finetuned", "ppo_high_lr_Llama-3.1-8B")
education_ppo_sft_new_lr, education_ppo_sft_new_lr_files = load_data(base_path, "Education Finetuned", "ppo_sft_Llama")

education_ppo_high_lr_sel, education_ppo_high_lr_idx = select_random_dialogs(education_ppo_high_lr, num_dialogs=1, seed=42)
education_ppo_sft_new_lr_sel, education_ppo_sft_new_lr_idx = select_random_dialogs(education_ppo_sft_new_lr, num_dialogs=1, seed=42)

indices_record["education_ppo_high_lr"] = {"files": education_ppo_high_lr_files, "indices": education_ppo_high_lr_idx}
indices_record["education_ppo_sft_new_lr"] = {"files": education_ppo_sft_new_lr_files, "indices": education_ppo_sft_new_lr_idx}

# Therapy Nonfinetuned
therapy_gemma, therapy_gemma_files = load_data(base_path, "Therapy Nonfinetuned", "gemma")
therapy_mistral, therapy_mistral_files = load_data(base_path, "Therapy Nonfinetuned", "mistral")
therapy_llama, therapy_llama_files = load_data(base_path, "Therapy Nonfinetuned", "Llama-3.1-8B-Instruct")

therapy_gemma_sel, therapy_gemma_idx = select_random_dialogs(therapy_gemma, num_dialogs=1, seed=42)
therapy_mistral_sel, therapy_mistral_idx = select_random_dialogs(therapy_mistral, num_dialogs=1, seed=42)
therapy_llama_sel, therapy_llama_idx = select_random_dialogs(therapy_llama, num_dialogs=1, seed=42)

indices_record["therapy_gemma"] = {"files": therapy_gemma_files, "indices": therapy_gemma_idx}
indices_record["therapy_mistral"] = {"files": therapy_mistral_files, "indices": therapy_mistral_idx}
indices_record["therapy_llama"] = {"files": therapy_llama_files, "indices": therapy_llama_idx}

# Therapy Finetuned
therapy_ppo_llama, therapy_ppo_llama_files = load_data(base_path, "Therapy Finetuned", "ppo_Llama-3.1-8B")
therapy_ppo_sft_new_lr_llama, therapy_ppo_sft_new_lr_llama_files = load_data(base_path, "Therapy Finetuned", "ppo_sft_new_lr_Llama-3.1-8B-Instruct")

therapy_ppo_llama_sel, therapy_ppo_llama_idx = select_random_dialogs(therapy_ppo_llama, num_dialogs=1, seed=42)
therapy_ppo_sft_new_lr_llama_sel, therapy_ppo_sft_new_lr_llama_idx = select_random_dialogs(therapy_ppo_sft_new_lr_llama, num_dialogs=1, seed=42)

indices_record["therapy_ppo_llama"] = {"files": therapy_ppo_llama_files, "indices": therapy_ppo_llama_idx}
indices_record["therapy_ppo_sft_new_lr_llama"] = {"files": therapy_ppo_sft_new_lr_llama_files, "indices": therapy_ppo_sft_new_lr_llama_idx}

combined_data = (
    chatting_gemma_sel + chatting_mistral_sel + chatting_llama_sel +
    chatting_llama_ppo_sel + chatting_llama_sftppo_sel +
    education_gemma_sel + education_mistral_sel + education_llama_sel +
    education_ppo_high_lr_sel + education_ppo_sft_new_lr_sel +
    therapy_gemma_sel + therapy_mistral_sel + therapy_llama_sel +
    therapy_ppo_llama_sel + therapy_ppo_sft_new_lr_llama_sel
)

# Write indices and filenames to a separate JSON file
indices_json_path = base_path / "qualtrics_selected_indices.json"
with open(indices_json_path, "w", encoding="utf-8") as f:
    json.dump(indices_record, f, indent=2)
print(f"✅ Indices and filenames saved to: {indices_json_path}")

# Generate Qualtrics survey for the combined data
output_txt_path = "/nfs/kun2/users/ryan_cheng/consistency_LLMs/qualtrics_survey.txt"
generate_qualtrics_survey(output_txt_path, data=combined_data)

