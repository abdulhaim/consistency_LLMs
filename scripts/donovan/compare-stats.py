from pathlib import Path
import json

from matplotlib import pyplot as plt

eval_keys = [
    # "eval_prompt_consistency",
    "P1_prompt_consistency_score",
    "P2_prompt_consistency_score",
    # "eval_index_consistency",
    # "P2_index_consistency_score",
    # "P1_index_consistency_score",
    # "P1_survey_consistency_score",
    # "P2_survey_consistency_score",
    # "P2_backstory_test",
    # "P1_backstory_test",
    # "eval_survey_consistency"
]

qwen_eval = Path("/mmfs1/home/donoclay/cse/donoclay/consistency_LLMs/training_data/qwen-14b-evals/in_chatting/Llama-3.1-8B-Instruct_0_623.json")
lamma_eval = Path("/mmfs1/home/donoclay/cse/donoclay/consistency_LLMs/training_data/in_chatting/Llama-3.1-8B-Instruct_0_623.json")

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def compare_evals(qwen_eval, lamma_eval):
    # find the MSE between the two evals
    # the json file is a list of dictionaries. each dictionary has at least `eval_keys` as keys.
    # when the dictionary doesn't have the key, the comparison should stop and return the MSE for each key
    # return a dictionary with the key as the eval key and the value as the MSE
    # also return lists of the values for each key
    qwen_data = load_json(qwen_eval)
    lamma_data = load_json(lamma_eval)

    eval_mse = {}
    eval_values = {}

    for key in eval_keys:
        qwen_values = []
        lamma_values = []
        for qwen_dialog, lamma_dialog in zip(qwen_data, lamma_data):
            if key in qwen_dialog and key in lamma_dialog:
                qwen_values.append(qwen_dialog[key])
                lamma_values.append(lamma_dialog[key])
            else:
                break
        if len(qwen_values) > 0 and len(lamma_values) > 0:
            mse = sum((q - l) ** 2 for q, l in zip(qwen_values, lamma_values)) / len(qwen_values)
            eval_mse[key] = mse
            eval_values[key] = (qwen_values, lamma_values)
        else:
            eval_mse[key] = None
            eval_values[key] = (None, None)
    return eval_mse, eval_values

def main():
    eval_mse, eval_values = compare_evals(qwen_eval, lamma_eval)
    for key, mse in eval_mse.items():
        if mse is not None:
            print(f"{key}: {mse:.4f}")
        else:
            print(f"{key}: No data available for comparison.")


    # Plot the values for each key
    for key, (qwen_values, lamma_values) in eval_values.items():
        if qwen_values is not None and lamma_values is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(qwen_values, label='Qwen Values', color='blue')
            plt.plot(lamma_values, label='Lamma Values', color='orange')
            plt.title(f'Comparison of {key}')
            plt.xlabel('Index')
            plt.ylabel(key)
            plt.legend()
            # plt.show()
            plt.savefig(f"./results/images/{key}_comparison.png")
        else:
            print(f"No data available for plotting {key}.")


if __name__ == "__main__":
    main()
    

