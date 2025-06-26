import json
import random
from pathlib import Path
import html

def load_data(base_path: Path, task_name):
    """Loads the JSON data from the specified file path."""
    file_path: Path = Path()
    if task_name == "Chatting":
        file_path = base_path / "chatting/exp/04.26.25"
    elif task_name == "Education":
        file_path = base_path / "education/exp/05.12.25"
    elif task_name == "Therapy":
        file_path = base_path / "therapy" / "exp" / "05.14.25.marwa"
    elif task_name == "Chatting PPO":
        file_path = base_path / "chatting/exp/05.06.25/ppo"
    elif task_name == "Education PPO":
        file_path = base_path / "education/exp/05.14.25/ppo_sft_Llama-3.1-8B-Instruct_0_365.json"
    elif task_name == "Therapy PPO":
        file_path = base_path / "therapy/exp/05.15.25/ppo_sft_new_lr_Llama-3.1-8B-Instruct_0_433.json"
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    
    if not file_path.exists():
        raise ValueError(f"Error: File {file_path} does not exist.")
    
    # iterate through all json files in the directory
    data = []
    file_paths = []

    files = [file_path] if ".json" in file_path.name else list(file_path.glob("*.json"))

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
    
    html_encoded.replace("&#226;&#128;&#147;", "&#8211;")  # Replace the long dash with a single HTML entity
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

# Example usage:
# generate_qualtrics_survey("/nfs/kun2/users/ryan_cheng/consistency_LLMs/therapy/exp/05.14.25.marwa/Llama-3.1-8B-Instruct_10_932.json", "qualtrics_survey.txt")

chatting_data, chatting_paths = load_data(Path("/nfs/kun2/users/ryan_cheng/consistency_LLMs"), "Chatting")
education_data, education_paths = load_data(Path("/nfs/kun2/users/ryan_cheng/consistency_LLMs"), "Education")
therapy_data, therapy_paths = load_data(Path("/nfs/kun2/users/ryan_cheng/consistency_LLMs"), "Therapy")

chatting_selected, chatting_indices = select_random_dialogs(chatting_data, num_dialogs=5, seed=42)
education_selected, education_indices = select_random_dialogs(education_data, num_dialogs=5, seed=42)
therapy_selected, therapy_indices = select_random_dialogs(therapy_data, num_dialogs=5, seed=42)

combined_data = chatting_selected
combined_data.extend(education_selected)
combined_data.extend(therapy_selected)

# Generate Qualtrics survey for the combined data
output_txt_path = "/nfs/kun2/users/ryan_cheng/consistency_LLMs/qualtrics_survey.txt"
generate_qualtrics_survey(output_txt_path, data=combined_data)

