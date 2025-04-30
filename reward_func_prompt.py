import torch
from openai import OpenAI #for vLLM server
import re
import json
import ray

metric_model = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
metadata_path = './training_data/out/metadata.json' # path to metadata json for evals to use
port = "8000" # port number vLLM is hosted on
eval_prompt_path = "./config/eval_prompts.json"
with open(metadata_path, 'r') as f:
    metadata_dict_ref = ray.put(json.load(f))

with open(eval_prompt_path, 'r') as f:
    eval_prompts = json.load(f)

def completion_create(prompt, model):
    # print('Prompt:', prompt)
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1" 

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    chat_response = client.chat.completions.create(model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256)
    ret = chat_response.choices[-1].message.content
    #print('Ret:', ret)
    return ret

def eval_prompt_consistency(metadata, line):
    prompt_consistency_score = 0

    pturn = conv_dict["pturn"]
            if 1 in agents:
    prompt = eval_prompts["combined_prompt_consistency"].replace("%SCENARIO_DESC%", prompts["scenario"]) \
                                                        .replace("%SPEAKER_ROLE%", prompts["agent_role"]) \
                                                        .replace("%SPEAKER_BACKSTORY%", conv_dict["P"]) \
                                                        .replace("%SPEAKER_LINE%", line)
    output = completion_create(prompt, metric_model)
    if "YES" not in output:  # no contradiction
        return 1
    return 0

def reward_func(queries, prompts, labels):
    '''
    OpenRLHF uses this to score the online model outputs
    queries is prompts + responses
    labels are answers (responses?)
    '''
    
    scores = []
    for i, query in enumerate(queries):
        metadata = ray.get(metadata_dict_ref)[prompts[i]] # 0: preference_distribution, 1: beliefs, 2: listener_alignment
        print(labels[i]) # remove when done debugging
        scores.append(eval_prompt_consistency(metadata, labels[i])

    return torch.tensor(scores)
