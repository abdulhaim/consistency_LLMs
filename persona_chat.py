import openai
from openai import OpenAI
import os
import glob
import time
import pickle
import pandas
import random

openai_key = glob.glob(os.path.abspath('../*openai*'))[0]
model = "gpt-3.5-turbo"
max_tokens = 256
with open(openai_key, 'r') as f:
    client = OpenAI(api_key=f.read().rstrip('\n'))

def completion_create_helper(prompt):
    if model in ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", 'gpt-4o-mini']:
        ret = client.chat.completions.create(model=model,
                                             messages=[{"role": "user", "content": prompt}],
                                             max_tokens=max_tokens)
        ret = ret.choices[-1].message.content
        return ret
    else:
        raise NotImplementedError

def completion_create(prompt, keep_trying=True):
    try:
        return completion_create_helper(prompt)
    except (openai.APIError, openai.OpenAIError) as e:
        print("ERROR", e)
        print("sleeping for 10 seconds.")
        time.sleep(10)
        if keep_trying:
            return completion_create(prompt, keep_trying)
        else:
            return None

with open('Meta-Llama-3-70B_demographics_survey+political_affiliation_batch_1+2+3_no_word_cutoff.pkl', 'rb') as f:
    df = pickle.load(f)

def get_backstory(i):
    s = df['backstory'][i]
    return s[s.find("Answer: ") + len("Answer: "):]

def get_backstory_test(backstory, num_questions):
    ret = [[], []] # a list of questions, a list of answers
    for i in range(num_questions):
        qa = completion_create("Based on the following backstory, generate a new fact-based multiple choice question with 5 choices addressed directly IN SECOND PERSON, along with its correct answer. Preface the question with 'Question:' and the answer with 'Answer:'." + '\n' + backstory + ("" if len(ret[0]) == 0 else "\nFor reference, all of the following questions have already been asked:\n" + ''.join(ret[0])))
        q = qa[qa.find('Question:'): qa.find('Answer:')]
        a = qa[qa.find('Answer:'): ]
        ret[0].append(q)
        ret[1].append(a)
    return ret

def score_backstory_test(prompt, backstory_test):
    total_score = 0
    for i in range(len(backstory_test[0])):
        answer = completion_create(prompt + "\n Based on your conversation above so far, answer the following multiple choice question.\n" + backstory_test[0][i])
        verdict = completion_create("As part of grading a test, determine whether the given answer " + answer + " matches the following correct answer. Respond with either YES or NO.\n" + "Correct " + backstory_test[1][i])
        score = 1 if 'yes' in verdict.lower() else 0
        if score == 0:
            #print("WRONG!")
            #print("The prompt is\n" + prompt)
            print("The correct answer to the question\n" + backstory_test[0][i] + "\nwas\n" + backstory_test[1][i] + "\nBut they answered\n" + answer)
        total_score += score
    return total_score

p1 = random.randint(0, len(df)-1)
p2 = random.randint(0, len(df)-1)
p1_backstory = get_backstory(p1)
p2_backstory = get_backstory(p2)
p1_backstory_test = get_backstory_test(p1_backstory, 5)
p2_backstory_test = get_backstory_test(p2_backstory, 5)

print("p1 backstory is\n" + p1_backstory)
print("p2 backstory is\n" + p2_backstory)

pturn = 1
conv_history = ""

for i in range(10):
    print(f"pturn={pturn}")
    if pturn == 1:
        prompt = "You are P1, and you are having a conversation with P2. Your backstory is:\n" + p1_backstory + "\n" + "So far, the conversation is as below, and it is your turn to speak next.\n" + (conv_history if len(conv_history) != 0 else "[You are starting the conversation.]") + "\nP1: "
        print("p1 score:", score_backstory_test(prompt, p1_backstory_test))
        p1_response = completion_create(prompt)
        conv_history += "P1: " + p1_response + "\n"
        pturn = 2
    else:
        prompt = "You are P2, and you are having a conversation with P1. Your backstory is:\n" + p1_backstory + "\n" + "So far, the conversation is as below, and it is your turn to speak next.\n" + (conv_history if len(conv_history) != 0 else "[You are starting the conversation.]") + "P2: "
        print("p2 score:", score_backstory_test(prompt, p2_backstory_test))
        p2_response = completion_create(prompt)
        conv_history += "P2: " + p2_response + "\n"
        pturn = 1

print(conv_history)
