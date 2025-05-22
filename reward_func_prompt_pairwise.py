import torch
from openai import OpenAI #for vLLM server
import re
import json
import ray

metric_model = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
metadata_path = '/nfs/kun2/users/ryan_cheng/consistency_LLMs/training_data/out_education/metadata.json' # path to metadata json for evals to use
port = "8001" # port number vLLM is hosted on
eval_prompt_path = "/nfs/kun2/users/ryan_cheng/consistency_LLMs/config/eval_prompts.json"
with open(metadata_path, 'r') as f:
    metadata_dict_ref = ray.put(json.load(f))

with open(eval_prompt_path, 'r') as f:
    eval_prompts_ref = ray.put(json.load(f))

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
    eval_prompts = ray.get(eval_prompts_ref)
    prompt = eval_prompts["combined_prompt_consistency"].replace("%SCENARIO_DESC%", metadata["scenario"]) \
                                                        .replace("%SPEAKER_ROLE%", metadata["agent_role"]) \
                                                        .replace("%SPEAKER_BACKSTORY%", metadata["P"]) \
                                                        .replace("%SPEAKER_LINE%", line)
    output = completion_create(prompt, metric_model)
    if "YES" not in output:  # no contradiction
        return 1
    return 0

def format_conversation(conversation):
    return "".join([str(i) + ": " + line for i, line in enumerate(conversation)])

def extract_list(text):
    pattern = r'\[.*?\]'
    match = re.search(pattern, text)
    if match:
        try:
            ret = eval(match.group())
            if ret and isinstance(ret[0], str):
                try:
                    ret = [eval(line) for line in ret]
                except (SyntaxError, NameError):
                    pass
            return eval(match.group())
        except (SyntaxError, NameError):
            return []
    return[]

def eval_index_consistency(metadata, line):
    '''
    proxy for pairwise consistency, asks for indices of the previous lines that are inconsistent
    agents is a list of what agents to include in evals (e.g. both agents: [1,2], only agent 2: [2])
    '''
    index_consistency_score = 0
    eval_prompts = ray.get(eval_prompts_ref)
    prompt = eval_prompts["index_consistency"].replace("%SCENARIO_DESC%", metadata["scenario"]) \
                                              .replace("%SPEAKER_ROLE%", metadata["agent_role"]) \
                                              .replace("%CONVERSATION%", format_conversation(metadata["conversation_history"])) \
                                              .replace("%SPEAKER_LINE%", line)
    
    output = completion_create(prompt, metric_model)
    index_list = extract_list(output)
    for j in index_list:
        if j != None and j % 2 == len(metadata["conversation"]) % 2:
            index_consistency_score += 1

    return 1-(index_consistency_score / round(len(metadata["conversation"]) / 2))
   
def reward_func(queries, prompts, labels):
    '''
    OpenRLHF uses this to score the online model outputs
    queries is prompts + responses
    labels are answers (responses?)
    '''
    
    scores = []
    for i, query in enumerate(queries):
        metadata = ray.get(metadata_dict_ref)[prompts[i]] # 0: preference_distribution, 1: beliefs, 2: listener_alignment
        # print("prompt:", prompts[i])
        # print("query:", query)
        cut_query = str(query.replace("<|eot_id|>", "")[len(prompts[i]):])
        # print("cut query:", cut_query)
        # print(labels[i]) # remove when done debugging
        prompt_consistency_score = eval_prompt_consistency(metadata, cut_query)
        index_consistency_score = eval_index_consistency(metadata, cut_query)
        scores.append(float(prompt_consistency_score * 0.5 + index_consistency_score * 0.5))

    return torch.tensor(scores)
