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
    file_path: Path = Path()
    if task_name == "Chatting":
        file_path = base_path / "chatting/exp/04.26.25"
    elif task_name == "Education":
        file_path = base_path / "training_data" / "in_education"
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

def print_background_info(dialog, index, **kwargs):
    if dialog["task_name"] in ["Therapy", "Therapy PPO"]:
        print(f"Background Information (dialog #{index} from dataset):")
        print(f"Task Name: {dialog['task_name']}")
        print(f"Patient's Background (P2):")
        printbf(f"\t{dialog['P2']}")

    elif dialog["task_name"] in ["Education", "Education PPO"]:
        print(f"Background Information (dialog #{index} from dataset):")
        print(f"Task Name: {dialog['task_name']}")
        print(f"Student's Background (P2):")
        printbf(f"\t{dialog['P2']}")
    
    elif dialog["task_name"] in ["Chatting", "Chatting PPO"]:
        print(f"Background Information (dialog #{index} from dataset):")
        print(f"Task Name: {dialog['task_name']}")

        agent = "P1" if kwargs.get("agent_speaking") == 0 else "P2"
        print(f"Agent {agent}'s Background:")
        printbf(f"\t{dialog[agent]}")

    else:
        raise ValueError(f"Unknown task name: {dialog['task_name']}")

def get_task_eval_agent_indices(dialog):
    if dialog["task_name"] in ["Therapy", "Therapy PPO"]:
        return [1]
    elif dialog["task_name"] in ["Education", "Education PPO"]:
        return [1]
    elif dialog["task_name"] in ["Chatting", "Chatting PPO"]:
        return [1]
    else:
        raise ValueError(f"Unknown task name: {dialog['task_name']}")


def human_evaluation(dialog, index, task_name):
    """Presents a single dialog to the user for evaluation and collects feedback."""
    eval_agent_indices = get_task_eval_agent_indices(dialog)
    agent_scores = {f"P{int(agent + 1)}": [] for agent in eval_agent_indices}

    print("-" * 40)
    print(f"Evaluating Dialog: {dialog['task_name']} - #{index}")
    print("-" * 40)

    past_utterances = []

    if task_name not in ["Chatting PPO", "Education PPO", "Therapy PPO"]:

        for turn_num, turn in enumerate(dialog['conversation']):
            utterance_index, utterance = turn
            agent_speaking = utterance_index % 2
            agent_label = f"P{agent_speaking + 1}"
            

            if agent_speaking in eval_agent_indices:
                print_background_info(dialog, index, **{"agent_speaking": agent_speaking})

                # Display the last 5 utterances for each agent, bolding the evaluated agent's
                # Display the last 5 utterances in interleaved order
                print("Recent Utterances:")
                last_five = past_utterances[-5:]
                for past_agent, past_utterance in last_five:
                    if agent_speaking == int(past_agent[-1]) - 1:
                        printbf(f"\t{past_agent}: {past_utterance}")
                    else:
                        print(f"\t{past_agent}: {past_utterance}")
                print()

                print(f"Evaluating Utterance {turn_num + 1} out of {len(dialog['conversation'])}:")
                printbf(f"\t{utterance.strip()}")
                print(f"For the utterance above, enter 1 if the utterance is consistent with the persona description, 0 if it is inconsistent.")
                consistency_score = input("Enter consistency score (1/0): ").strip()
                while consistency_score not in ['1', '0']:
                    print("Invalid input. Please enter 1 for consistent or 0 for inconsistent.")
                    consistency_score = input("Enter consistency score (1/0): ").strip()
                agent_scores[f"P{agent_speaking + 1}"].append(int(consistency_score))
                print()
            
            # Store the current utterance for future reference
            past_utterances.append((agent_label, utterance))

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
    
    else:
        # For PPO tasks, we will only evaluate one utterance. The utterance will be randomly selected, but it should always be P2. Show the background info, and previous 10 utterances.
        # Select a random utterance to evaluate
        utterance_index = random.randint(0, len(dialog['conversation']) // 2) * 2 - 1
        # Get the agent speaking
        agent_speaking = 1
        agent_label = f"P{agent_speaking + 1}"
        # Get the utterance
        utterance = dialog['conversation'][utterance_index][1]
        # Get the last 10 utterances
        past_utterances = dialog['conversation'][:utterance_index]
        # Get the last 10 utterances in interleaved order
        past_utterances = past_utterances[-10:]

        # print the background info
        print_background_info(dialog, index, **{"agent_speaking": agent_speaking})

        # Display the last 10 utterances in interleaved order
        print("Recent Utterances:")
        # print("past_utterances:", past_utterances)
        for past_utterance_idx, past_utterance in past_utterances:
            past_speaker = past_utterance_idx % 2
            if agent_speaking == past_speaker:
                printbf(f"\tP{past_speaker + 1}: {past_utterance}")
            else:
                print(f"\tP{past_speaker + 1}: {past_utterance}")

        print()
        print(f"Evaluating Utterance {utterance_index + 1} out of {len(dialog['conversation'])}:")
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

    return evaluation

def main(base_dir: str, num_dialogs: int=10):
    """Loads data, selects random dialogs with a seed, and walks the user through evaluation."""

    base_dir = Path(base_dir)

    # ask the user to select a task
    task_name = input("Enter the number corresponding to the task you want to evaluate:\n1. Therapy\n2. Education\n3. Chatting\n4. Chatting PPO\n5. Education PPO\n6. Therapy PPO\n")
    task_name = task_name.strip()
    if task_name == "1":
        task_name = "Therapy"
    elif task_name == "2":
        task_name = "Education"
    elif task_name == "3":
        task_name = "Chatting"
    elif task_name == "4":
        task_name = "Chatting PPO"
    elif task_name == "5":
        task_name = "Education PPO"
    elif task_name == "6":
        task_name = "Therapy PPO"
    else:
        print("Invalid input. Please enter 1, 2, 3, 4, 5, or 6.")
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
        evaluation_result = human_evaluation(dialog, index, task_name)
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

    base_dir = "/mmfs1/home/donoclay/cse/donoclay/consistency_LLMs"
    num_dialogs = 10

    main(base_dir, num_dialogs)