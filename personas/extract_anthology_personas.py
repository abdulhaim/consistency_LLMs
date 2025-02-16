"""
Extract personas from the input file and save them to a json file.
"""
import json
import pathlib
import pickle
import pandas as pd

INPUT_FILE = "raw_anthology_personas.pkl"
OUTPUT_FILE = "anthology_personas.pkl"
OUTPUT_JSON = "tiny_anthology_personas.json"
NUM_PERSONAS = 30 # number of personas to save into a json

def extract_personas():
    """
    Extract personas from the input file and save them to a json file.
    """
    data_path = pathlib.Path(__file__).parent / "data"
    with open(f"{data_path}/{INPUT_FILE}", "rb") as file:
        data = pickle.load(file)
    df = pd.DataFrame(data)

    # for each row in column, extract the persona
    personas = []
    for _, value in df['age_category_4_llm_parsing_prompt'].items():
        answer_str = "Answer:"
        start_idx = value.find(answer_str) + len(answer_str)
        question_str = "Question:"
        end_idx = value.find(question_str)
        persona = value[start_idx:end_idx].strip()
        personas.append(persona)

    with open(f"{data_path}/{OUTPUT_FILE}", "wb") as pkl_file:
        pickle.dump(personas, pkl_file)

    with open(f"{data_path}/{OUTPUT_JSON}", "w") as json_file:
        json.dump(personas[:NUM_PERSONAS], json_file, indent=4)

if __name__ == "__main__":
    extract_personas()
