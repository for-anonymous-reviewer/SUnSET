import requests
import datetime
import time
import json
from tqdm import tqdm
from pathlib import Path
import ast
import time
import pickle
import spacy
import os

def load_dictionary(file_path):
    try:
        with open(file_path, "rb") as file:
            dictionary = pickle.load(file)
        return dictionary
    except FileNotFoundError:
        print("File not found.")
        return None

def save_dictionary(dictionary, dict_path):
    with open(dict_path, "wb") as file:
        pickle.dump(dictionary, file)


###########################################################################################################
root_path =  r"C:\Users\t84401143\Documents\work\datasets\t17"
output_path = Path(r"C:\Users\t84401143\Documents\work\scoring")
output_path.mkdir(exist_ok=True, parents=True)
# full_context=[] #to store the dictionary of every article
# save_name = output_path / f"set_newv2.jsonl"

# File path to save the dictionary
dict_path = output_path / f"t17.pkl" #secretly a list :D

# Load the (not-not)-really-a-dictionary
# middle_dict = load_dictionary(dict_path)
# print("First 10 items:", list(middle_dict.items())[:20])
k=0
main={}
names=['bpoil','egypt','finan','h1n1','haiti','iraq','libya','mj','syria']
for dirpath, _, filenames in os.walk(root_path):
    for filename in filenames:
        if filename == "set_newv2.jsonl":  # Match filenames
            json_file = os.path.join(dirpath, filename)
            # Loading...
            with open(json_file, 'r', encoding='utf-8') as f:
                data_list = [json.loads(line) for line in f]
            name=names[k]
            k+=1
            lol={} #unique stakeholders for topic
            for i, data in tqdm(enumerate(data_list), total=len(data_list)):
                for hackers in data['stake']: #use per article than per iterated event teehee~
                #####ADD HERE KEK
                    if hackers not in lol:
                        lol[hackers] = 1
                    else:
                        lol[hackers]+=1
                    if hackers not in main:
                        main[hackers]=1
                    else:
                        main[hackers]+=1
            dict_path2= output_path / f"t17_{name}.pkl"
            save_dictionary(lol, dict_path2)
            print("Indi. dictionary saved.")

save_dictionary(main, dict_path)
print("Main Dictionary saved.")


#count per top plus count across all