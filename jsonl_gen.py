import json
import os
import glob
from tqdm import tqdm
import random
from absl import app, flags
import re


flags.DEFINE_string('folder', './training_data', 'folder of jsons to conglomerate into RL training data, expects a folder called "in" and "out" within that (./training_data default)')

def phrase(b: bool):
    return " " if b else " not "

def format_conversation(convo):
    '''
    Formats the conversation as a list of dictionaries with "in_text" and "out_text" corresponding to the input prompt and desired output of the LLM
    "score" is used for offline RL/KTO rewards (binary 0 or 1)
    additional fields are added to the dictionary as necessary for online RL, which loads particular model data from metadata_dict
    '''
    conversation = convo['conversation']

    statements = re.split(r'(Buyer:|Seller:)', conversation)
    
    # Combine the speaker tag with the following statement
    paired_statements = [statements[i] + statements[i+1] for i in range(1, len(statements) - 1, 2)]
    utterances = [statement.strip() for statement in paired_statements]

    ret = []
    prompt_desc = "A seller is trying to convince a buyer to come to a house showing. "
    prompt_desc += "In this situation, there are five features: whether the house is big, whether the house has a garage, whether the neighborhood is quiet, whether the house has a basement, and whether the house has a backyard. The buyer has their own preferences about whether they want each of these features to be true or false, but they do not know about the reality of any of these features. Only the seller knows about the reality of each of these features, but they do not initially know about the buyer's preferences. If the buyer believes enough of the features match their preferences, they will decide to come to the house showing. Otherwise, they will not come to the showing. There are up to 10 turns (consisting of the buyer followed by the seller) to have a conversation, after which it automatically ends, defaulting to the buyer not coming to the house showing. As soon as the buyer has made their decision and would like to end the conversation, the buyer can state either '<YES>' denoting that they will come to the house showing or '<NO>' denoting that they will not come to the house showing, along with the reasoning for their decision.\n\n BE SHORT AND CONCISE IN YOUR DISCUSSION, LIMIT YOUR RESPONSE TO A MAXIMUM OF 2 SENTENCES. \n\n"
    for i in range(len(utterances)-1): 
        if utterances[i] != 'Buyer:' and utterances[i] != 'Seller:' and utterances[i+1] != 'Buyer:' and utterances[i+1] != 'Seller:':
            # TASK REWARD - BELIEF MISALIGNMENT
            if int(convo['agree'])/2 - convo['belief_misalignment'] > 0.5:
                score = 1
            else:
                score = 0
            # TASK REWARD BINARY
            if convo['agree']:
                score = 1
            else:
                score = 0

            if 'Buyer:' in utterances[i] and "Seller:" in utterances[i+1]:
                seller_desc = f"\nYou are the seller. Here are the true features of the house: the house is{phrase(convo['big_truth'])}big, the house does{phrase(convo['garage_truth'])}have a garage, the neighborhood is{phrase(convo['quiet_truth'])}quiet, the house does{phrase(convo['basement_truth'])}have a basement, and the house does{phrase(convo['backyard_truth'])}have a backyard. Remember that all the information you have on the buyer's preferences is what they tell you and your only goal is to convince the buyer to come to the house showing."
                try:
                    ret.append({
                        "in_text": prompt_desc + seller_desc + "\nThis is the conversation so far, and you will speak next:\n\n" + ''.join(utterances[:i+1]), 
                        "out_text": utterances[i+1], 
                        'score':score,
                        'preference_distribution': [convo['big_pref'], convo['garage_pref'], convo['quiet_pref'], convo['basement_pref'], convo['backyard_pref']],
                        'beliefs': convo['belief_bool'][i//2],
                        'listener_alignment': convo['listener_alignment']
                    })
                except:
                    print(i//2)
                    print(len(convo['belief_bool']))
                    print(len(utterances))
                    raise Exception("")
    return ret

def main(argv):
    random.seed(0)
    jsonl_total = []

    metadata_dict = {}

    for filename in tqdm(glob.glob(flags.FLAGS['folder'] + '/in/*.json')): # ./training_data/in/*.json
        with open(filename, 'r') as f:
            convos = json.load(f)
        for convo in convos:
            lines = format_conversation(convo)
            for line in lines:
                metadata_dict[line['in_text']] = [ # metrics to save within the metadata
                    line['preference_distribution'],
                    line['beliefs'],
                    line['listener_alignment']
                ]
                del line['preference_distribution']
                del line['beliefs']
                del line['listener_alignment']
            jsonl_total += lines

    random.shuffle(jsonl_total)

    train_len = int(0.8 * len(jsonl_total))
    train_data = jsonl_total[:train_len]
    eval_data = jsonl_total[train_len:]

    # Save to JSONL
    with open(flags.FLAGS['folder'] + '/out/train.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(flags.FLAGS['folder'] + '/out/test.jsonl', 'w') as f:
        for item in eval_data:
            f.write(json.dumps(item) + '\n')

    # Save metadata dictionary
    with open(flags.FLAGS['folder'] + '/out/metadata.json', 'w') as f:
        json.dump(metadata_dict, f, indent=4)


if __name__ == '__main__':
    app.run(main)