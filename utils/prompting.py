import random
import re
from tqdm import tqdm
from os import listdir
import openai
import os
import pandas as pd
import time
from pathlib import Path
import random
import re
from tqdm import tqdm

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../prompts")


def get_prompt(name):
    with open(os.path.join(PROMPT_DIR, name + ".txt")) as f:
        return "".join([line for line in f])


def get_completion(prompt, model, max_tokens=1000, temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]
