"""
The purpose of this script is to generate a new persona.
"""
from openai import OpenAI
import os
import json
from persona import Persona
from persona_prompts import generate_persona_prompt, extract_persona_prompt

MODEL = "gpt-4o-mini" # davinci-002
MAX_COMPLETION_TOKENS = 500
PERSONAS_FILE = "personas.json"
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

NUM_PERSONAS = 4
MODE = 1
PERSONAS_FILES = [
    "generic_personas.json", # 0 - Generic
    "car_dealer_personas.json", # 1 - Car Dealer
    "therapist_personas.json", # 2 - Therapist
    "teacher_personas.json", # 3 - Teacher
    "student_personas.json" # 4 - Student
]

def generate_biography(mode = 0):
    """
    Generate a detailed biography for the persona.

    Returns:
        str: The generated biography as a natural language paragraph.
    """
    modifier = [
        "a person", # Mode 0 - Generic
        "a car salesperson.", # Mode 1 - Car Dealer
        "a therapist with a background in psychology.", # Mode 2 - Therapist
        "a person who provides tutoring in one or more areas", # Mode 3 - Teacher
        "a student in high school or college" # Mode 4 - Student
    ]
    memory = [{"role": "user", "content": generate_persona_prompt.replace("<type>", modifier[mode])}]

    response = client.chat.completions.create(
        model=MODEL,
        messages=memory,
        max_completion_tokens=MAX_COMPLETION_TOKENS
    )
    answer = response.choices[0].message.content.strip()
    return answer

def load_existing_personas(file_path):
    """
    Load existing personas from the JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: List of existing personas or an empty list if the file doesn't exist or is empty.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
                return data if isinstance(data, list) else []
            except json.JSONDecodeError:
                return []
    return []
            
def save_personas(file_path, personas):
    """
    Save the list of personas to the JSON file.

    Args:
        file_path (str): Path to the JSON file.
        personas (list): List of personas to save.
    """
    with open(file_path, "w") as file:
        json.dump(personas, file, indent=4)

def append_personas_to_file(file_path, new_personas):
    """
    Append new personas to the JSON file.

    Args:
        file_path (str): Path to the JSON file.
        new_persona (dict): The new persona to append.
    """
    personas = load_existing_personas(file_path)
    for new_persona in new_personas:
        personas.append(new_persona)
    save_personas(file_path, personas)

def extract_features(persona):
    """
    Extract one-sentence summaries from a biography.

    Args:
        biography (str): The detailed biography.

    Returns:
        dict: A dictionary of extracted features and their summaries.
    """
    prompt = extract_persona_prompt.replace("<biography>", persona.biography)
    memory = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model = MODEL,
        messages = memory,
        max_tokens = MAX_COMPLETION_TOKENS
    )
    summaries = response.choices[0].message.content.strip()

    for line in summaries.split("\n"):
        if ": " in line:
            key, value = line.split(": ", 1)
            key = key.strip()
            value = value.strip().strip('"')

            if hasattr(persona, key):
                setattr(persona, key, value)
    return persona

def generate_personas(num_personas, mode):
    """
    Generate a set of personas with detailed biographies and extracted features.

    Args:
        num_personas (int): The number of personas to generate.
        mode (int): The mode of the personas to generate.
    """
    file_path = f"data/{PERSONAS_FILES[mode]}"
    personas = []
    for _ in range(num_personas):
        persona = Persona()
        persona.biography = generate_biography(mode)
        print(persona.biography + "\n")
        updated_persona = extract_features(persona)
        persona_dict = {attr: getattr(updated_persona, attr) for attr in vars(updated_persona)}
        personas.append(persona_dict)
    append_personas_to_file(file_path, personas)

def main():
    generate_personas(NUM_PERSONAS, MODE)

if __name__ == "__main__":
    main()
