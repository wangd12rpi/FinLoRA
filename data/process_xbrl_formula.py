import json
from typing import List, Dict
from tqdm import tqdm
import re
import random
import os.path
import pandas as pd

if __name__ == '__main__':
    df = pd.read_excel('test/xbrl_agent_formula.xlsx')
    for index, row in df.iterrows():
        context, target = row["Question"], row["Correct Answer"]
        print(f"Question {index + 1} result: {context}")


