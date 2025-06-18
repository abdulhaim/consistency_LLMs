import os
import json
import matplotlib.pyplot as plt
import numpy as np # Import numpy for potential future numerical operations
from collections import defaultdict

# --- Configuration ---
# Define the base directory where 'model_eval_ablations' is located
base_dir = "/mmfs1/home/donoclay/cse/donoclay/consistency_LLMs/training_data/model_eval_ablations/"

output_dir = "/mmfs1/home/donoclay/cse/donoclay/consistency_LLMs/results/model_ablations/"

# Define the benchmark model name (this is the EVALUATOR model)
benchmark_model_name = "Llama-3.1-8B-Instruct"

# Define the tasks (subdirectories within each model's directory)
tasks = ["in_chatting", "in_education", "in_therapy"]

eval_keys = [
    # "P1_prompt_consistency_score",
    "P2_prompt_2_stage_consistency_score",
    # Add any other keys you want to evaluate from the JSON files
    ]

# IMPORTANT: List all possible model names that could appear at the beginning
# of your JSON filenames within the task directories.
# These are the models whose DATA was evaluated.
KNOWN_DATA_GENERATING_MODEL_PREFIXES = [
    "Llama-3.1-8B-Instruct",
    "gemma-2-2b-it",
    "mistral-instruct",
    # Add any other model names that appear as prefixes in your JSON filenames
]
# Sort them by length in descending order to correctly match longer prefixes first
KNOWN_DATA_GENERATING_MODEL_PREFIXES.sort(key=len, reverse=True)

# --- Helper Functions ---
def get_metrics_from_json(filepath):
    """
    Reads a JSON file and extracts a numeric metric.
    It expects the key 'P2_prompt_2_stage_consistency_score'.
    """
    key_to_values_map = defaultdict(list)  

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

            for key in eval_keys:

                for idx, dialog in enumerate(data):
                    # print(dialog.keys())

                    if key in dialog:
                        value = dialog[key]
                        key_to_values_map[key].append(value)
                        
                    else:
                        # print(f"Warning: Value for key '{key}' in {filepath} is not numeric: {value}")
                        # print("Failed on dialog index:", idx)
                        continue
                # if values:
                #     key_to_values_map[filepath] = values
            # else:
            #     print(f"Warning: No numeric values found for keys {eval_keys} in {filepath}")
        return key_to_values_map
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None
    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {filepath}: {e}")
        return None

def parse_filename(filename):
    """
    Parses a filename to extract the data-generating model name and its run identifier.
    Expected format: 'DATA_GENERATING_MODEL_NAME_RUN_IDENTIFIER.json'
    Example: 'Llama-3.1-8B-Instruct_0_623.json' -> ('Llama-3.1-8B-Instruct', '0_623')
             'gemma-2-2b-it_len_60.json' -> ('gemma-2-2b-it', 'len_60')
    """
    name_without_ext = filename.replace('.json', '')
    for prefix in KNOWN_DATA_GENERATING_MODEL_PREFIXES:
        if name_without_ext.startswith(prefix):
            data_generating_model = prefix
            # Extract the part after the prefix and the first underscore (if present)
            remaining_part = name_without_ext[len(prefix):]
            run_identifier = remaining_part.lstrip('_') # Remove leading underscores
            
            if run_identifier: # Ensure there's a non-empty identifier
                return data_generating_model, run_identifier
    return None, None # Cannot parse if no known prefix matches

# --- Main Script Logic ---

# Get a list of all directories within the base_dir (these are the EVALUATOR models)
try:
    all_eval_model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
except FileNotFoundError:
    print(f"Error: Base directory '{base_dir}' not found. Please ensure the script is run from the correct location.")
    exit()

# Filter out 'test' and the benchmark model itself (as it's used for comparison)
eval_models_to_compare = [model for model in all_eval_model_dirs if model not in ["test", benchmark_model_name]]

if not eval_models_to_compare:
    print("No evaluation models found to compare (excluding 'test' and the benchmark model).")
    exit()

print(f"Starting comparison for the following models: {', '.join(eval_models_to_compare)}")
print(f"Benchmark evaluator model: {benchmark_model_name}")

# Loop through each evaluation model you want to compare
for eval_model in eval_models_to_compare:
    print(f"\nProcessing evaluation model: {eval_model}")

    # Loop through each task (in_chatting, in_education, in_therapy)
    for task in tasks:
        print(f"  Task: {task}")

        eval_model_task_dir = os.path.join(base_dir, eval_model, task)
        benchmark_model_task_dir = os.path.join(base_dir, benchmark_model_name, task)

        # Check if task directories exist for both models
        if not os.path.exists(eval_model_task_dir):
            print(f"    Warning: Task directory not found for {eval_model} in {task}. Skipping.")
            continue
        if not os.path.exists(benchmark_model_task_dir):
            print(f"    Warning: Task directory not found for {benchmark_model_name} in {task}. Skipping.")
            continue

        # Get list of JSON filenames for current eval model and benchmark model in this task
        eval_model_filenames = [f for f in os.listdir(eval_model_task_dir) if f.endswith('.json')]
        benchmark_model_filenames = [f for f in os.listdir(benchmark_model_task_dir) if f.endswith('.json')]

        if not eval_model_filenames:
            print(f"    No JSON files found for {eval_model} in {task}. Skipping.")
            continue
        if not benchmark_model_filenames:
            print(f"    No JSON files found for {benchmark_model_name} in {task}. Skipping.")
            continue

        # Map (data_generating_model, run_identifier) to full file paths
        eval_model_files_map = {} # Key: (data_gen_model, run_id), Value: filepath
        for filename in eval_model_filenames:
            data_gen_model, run_id = parse_filename(filename)
            if data_gen_model and run_id:
                eval_model_files_map[(data_gen_model, run_id)] = os.path.join(eval_model_task_dir, filename)
            else:
                print(f"      Warning: Could not parse filename '{filename}' for evaluator '{eval_model}'. Skipping.")

        benchmark_model_files_map = {} # Key: (data_gen_model, run_id), Value: filepath
        for filename in benchmark_model_filenames:
            data_gen_model, run_id = parse_filename(filename)
            if data_gen_model and run_id:
                benchmark_model_files_map[(data_gen_model, run_id)] = os.path.join(benchmark_model_task_dir, filename)
            else:
                print(f"      Warning: Could not parse filename '{filename}' for evaluator '{benchmark_model_name}'. Skipping.")

        # Prepare data for plotting
        model_to_values = {}

        # Iterate through common (data_generating_model, run_identifier) pairs
        common_file_keys = sorted(list(set(eval_model_files_map.keys()) & set(benchmark_model_files_map.keys())))

        if not common_file_keys:
            print(f"    No common (data_gen_model, run_id) pairs found for comparison in task '{task}' for model '{eval_model}'. Skipping plot.")
            continue

        

        # model_to_values[benchmark_model_name] = benchmark_values

        # print("commond file keys:\n", common_file_keys)

        for data_gen_model, run_id in common_file_keys:
            eval_filepath = eval_model_files_map[(data_gen_model, run_id)]
            

            eval_values = get_metrics_from_json(eval_filepath)

            benchmark_filepath = benchmark_model_files_map[(data_gen_model, run_id)]

            benchmark_values = get_metrics_from_json(benchmark_filepath)

            # print(eval_values.keys())

            # model_to_values[eval_model] = eval_values
            

            if eval_values is not None and benchmark_values is not None:
                # Label for the x-axis bar will combine data-generating model and run ID
                # plot_labels.append(f"{data_gen_model}_{run_id}")
                # eval_scores.append(eval_score)
                # benchmark_scores.append(benchmark_score)
                pass
            else:
                print(f"      Skipping comparison for (Data Model: '{data_gen_model}', Run ID: '{run_id}') due to missing score in one of the files.")

            # print(eval_values.keys())

            plt.plot(eval_values['P2_prompt_2_stage_consistency_score'], label=f'{eval_model} {task}', marker='o')
            plt.plot(benchmark_values['P2_prompt_2_stage_consistency_score'], label=f'{benchmark_model_name} {task}', marker='x')
            plt.title(f'{eval_model} vs {benchmark_model_name}; task: {task}, id: {run_id}')
            plt.xlabel('Data Generating Model and Run ID')
            plt.ylabel('P2 Prompt Consistency Score')
            plt.ylim(ymin=-0.1, ymax=1.1)
            plt.legend()

            plt.tight_layout() # Adjust layout to prevent labels from overlapping
            plot_filename = f"{eval_model}_{task}_{run_id}_comparison_P2_consistency.png"
            plt.savefig(output_dir + plot_filename)
            print(f"    Generated plot: {plot_filename}")
            plt.close()

        # print(model_to_values['gemma-12b-it'].keys())

print("\nScript execution finished.")
