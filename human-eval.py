import json
import random
from pathlib import Path
from datetime import datetime

def printbf(text):
    """Prints text in bold."""
    print(f"\033[1m{text}\033[0m")

def initialize_directories(base_path: Path, task_name: str):
    output_dir = base_path / "results" / "human_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # create a directory for the specific task
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # create a timestamped directory for each run
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    run_dir = task_dir / f"human_eval_{timestamp}"
    print(f"Creating directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def load_data(base_path: Path, task_name):
    """Loads the JSON data from the specified file path."""
    if task_name == "Chatting":
        # /mmfs1/home/donoclay/cse/donoclay/consistency_LLMs/therapy/training_data/in
        file_path = base_path / "training_data" / "in_chatting"
    elif task_name == "Education":
        # /mmfs1/home/donoclay/cse/donoclay/consistency_LLMs/training_data/in_education
        file_path = base_path / "training_data" / "in_education"
    elif task_name == "Therapy":
        # /mmfs1/home/donoclay/cse/donoclay/consistency_LLMs/training_data/in
        file_path = base_path / "training_data" / "in"
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    
    if not file_path.exists():
        raise ValueError(f"Error: File {file_path} does not exist.")
    
    # iterate through all json files in the directory
    data = []
    file_paths = []
    for file in file_path.glob("*.json"):
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

def print_background_info(dialog, index, **kwargs):
    if dialog["task_name"] == "Therapy":
        print(f"Background Information (dialog #{index} from dataset):")
        print(f"Task Name: {dialog['task_name']}")
        print(f"Patient's Background (P2):")
        printbf(f"\t{dialog['P2']}")

    elif dialog["task_name"] == "Education":
        print(f"Background Information (dialog #{index} from dataset):")
        print(f"Task Name: {dialog['task_name']}")
        print(f"Student's Background (P2):")
        printbf(f"\t{dialog['P2']}")
    
    elif dialog["task_name"] == "Chatting":
        print(f"Background Information (dialog #{index} from dataset):")
        print(f"Task Name: {dialog['task_name']}")

        agent = "P1" if kwargs.get("agent_speaking") == 0 else "P2"
        print(f"Agent {agent}'s Background:")
        printbf(f"\t{dialog[agent]}")

    else:
        raise ValueError(f"Unknown task name: {dialog['task_name']}")

def get_task_eval_agent_indices(dialog):
    if dialog["task_name"] == "Therapy":
        return [1]
    elif dialog["task_name"] == "Education":
        return [1]
    elif dialog["task_name"] == "Chatting":
        return [1]
    else:
        raise ValueError(f"Unknown task name: {dialog['task_name']}")


def human_evaluation(dialog, index):
    """Presents a single dialog to the user for evaluation and collects feedback."""
    eval_agent_indices = get_task_eval_agent_indices(dialog)
    agent_scores = {f"P{int(agent + 1)}": [] for agent in eval_agent_indices}

    print("-" * 40)
    print(f"Evaluating Dialog: {dialog['task_name']} - #{index}")
    print("-" * 40)

    for turn_num, turn in enumerate(dialog['conversation']):
        utterance_index, utterance = turn
        agent_speaking = utterance_index % 2

        if agent_speaking in eval_agent_indices:
            print_background_info(dialog, index, **{"agent_speaking": agent_speaking})
            print(f"Utterance {turn_num + 1} out of {len(dialog['conversation'])}:")
            printbf(f"\t{utterance.strip()}")
            print(f"For the utterance above, enter 1 if the utterance is consistent with the persona description, 0 if it is inconsistent.")
            consistency_score = input("Enter consistency score (1/0): ").strip()
            while consistency_score not in ['1', '0']:
                print("Invalid input. Please enter 1 for consistent or 0 for inconsistent.")
                consistency_score = input("Enter consistency score (1/0): ").strip()
            agent_scores[f"P{agent_speaking + 1}"].append(int(consistency_score))
            print()

    print("-" * 40)

    evaluation = {}
    for agent, scores in agent_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            evaluation[f"Agent {agent}"] = {
                "average_consistency_score": avg_score,
                "individual_scores": scores
            }
        else:
            evaluation[f"Agent {agent}"] = {
                "average_consistency_score": None,
                "individual_scores": []
            }
    print("Evaluation Summary:")
    for agent, result in evaluation.items():
        print(f"{agent}:")
        print(f"  Average Consistency Score: {result['average_consistency_score']}")
        print(f"  Individual Scores: {result['individual_scores']}")
    print("-" * 40)
    print("End of Evaluation for this dialog.")


    return evaluation

def main(base_dir: str, num_dialogs: int=10):
    """Loads data, selects random dialogs with a seed, and walks the user through evaluation."""

    base_dir = Path(base_dir)

    # ask the user to select a task
    task_name = input("Enter the number corresponding to the task you want to evaluate:\n1. Therapy\n2. Education\n3. Chatting\n")
    task_name = task_name.strip()
    if task_name == "1":
        task_name = "Therapy"
    elif task_name == "2":
        task_name = "Education"
    elif task_name == "3":
        task_name = "Chatting"
    else:
        print("Invalid input. Please enter 1, 2, or 3.")
        return
    print(f"You selected: {task_name}")

    run_dir = initialize_directories(base_dir, task_name)
    print(f"Run directory created: {run_dir}")

    data, file_paths = load_data(base_dir, task_name)
    if not data:
        print("Error: No data loaded. Please ensure 'your_data.json' exists and is not empty.")
        return
    
    random_dialogs, indices = select_random_dialogs(data, num_dialogs=num_dialogs)
    evaluations = []

    print("Starting human evaluation of randomly selected dialogs...\n")

    for i, (dialog, index) in enumerate(zip(random_dialogs, indices)):
        print(f"\n--- Dialog {i + 1} of {len(random_dialogs)} ---")
        evaluation_result = human_evaluation(dialog, index)
        evaluations.append({
            "task_name": dialog["task_name"],
            "evaluation": evaluation_result,
            "dialog_index": index,
        })

    print("\n--- Evaluation Complete ---")

    # write to run_dir
    output_file = run_dir / "evaluations.json"
    with open(output_file, 'w') as f:
        json.dump(evaluations, f, indent=4)
    print(f"Evaluations saved to {output_file}")

    # store the metadata (order of the files that were loaded and appended)
    metadata_file = run_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(file_paths, f, indent=4)
    print(f"Metadata saved to {metadata_file}")

if __name__ == "__main__":

    base_dir = "/nfs/kun2/users/ryan_cheng/consistency_LLMs"
    num_dialogs = 10

    main(base_dir, num_dialogs)