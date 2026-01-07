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
import statistics
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

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
root_path =  Path(r"C:\Users\t84401143\Documents\work\scoring")
output_path = Path(r"C:\Users\t84401143\Documents\work\scoring\p2")
output_path.mkdir(exist_ok=True, parents=True)
# full_context=[] #to store the dictionary of every article
# save_name = output_path / f"set_newv2.jsonl"

# File path to save the dictionary
# dict_path = output_path / f"p1_t17.pkl" #p1 differs across topics
# new_main={}

main_path = root_path / f"crisis.pkl"
# Load the main dict with all stakeholders
main_dict = load_dictionary(main_path)
# print("First 10 items:", list(middle_dict.items())[:20])
k=0
#names=['bpoil','egypt','finan','h1n1','haiti','iraq','libya','mj','syria']
names=['egypt','libya','syria','yemen']
colours=['red','blue','green','yellow']
curr_dict=[] #[{},{},{},{}] etc.
for dirpath, _, filenames in os.walk(root_path):
    for filename in filenames:
        if k==len(names):
            break
        name=names[k]
        if filename == f"crisis_{name}.pkl":  # Match filenames
            sep_path = os.path.join(dirpath, filename)
            # Loading...
            curr_dict.append(load_dictionary(sep_path))
            k+=1
            

fullplt=[[],[],[],[]]
e={}
l={}
s={}
y={}
for i, (key,value) in tqdm(enumerate(main_dict.items()), total=len(main_dict)):
    temp=[]
    for files in range(len(names)): #9
        try:
            temp.append(curr_dict[files][key])
        except:
            temp.append(0)
    #check
    if value!=sum(temp):
        print(f'error:{value} v.s. {sum(temp)}')

    # gold=[]
    # # Calculate the mean
    mean_val = statistics.mean(temp)

    # # Calculate the standard deviations
    # std_dev_population = statistics.pstdev(temp)  # Population standard deviation
    std_dev_sample = statistics.stdev(temp) 
    cv=std_dev_sample/mean_val
    cv_norm=cv/sqrt(len(names))
    percentages = temp / np.sum(temp)
    pv=cv_norm * percentages
    # if max(pv)==2:
    #     print(temp)
    # pv2={}
    for files in range(4): #9
        fullplt[files].append(pv[files])
        # pv2[names[files]]=pv[files]

    # if i%500==0:
    #     print(f'distribution:{temp}')
    #     print(f'percentage:{percentages}')
    #     print(f'cv:{cv}')
    #     print(f'pv:{pv}')

    e[key]=pv[0]
    l[key]=pv[1]
    s[key]=pv[2]
    y[key]=pv[3]
    # gold.append(mean_val)
    # gold.append(std_dev_sample)
    # gold.append(cv) #plot these out then multiply to get pv2...?
    # # gold.append(pv2)

# for series, color in zip(fullplt, colours):
#     plt.plot(series, marker='o', color=color, label=str(series))
for i, (series, color) in enumerate(zip(fullplt, colours)):
    plt.scatter(range(len(series)), series, color=color, label=f"Series {i+1}")
    print(f'mean: {statistics.mean(series)}')
    print(f'std: {statistics.stdev(series) }')
    print(f'max: {max(series) }')
# Optional: add labels and a title.
plt.xlabel('Index')
plt.ylabel('P_v2')
plt.title('Plot of P_v2 values')
plt.legend()  # Display a legend showing which line corresponds to which array.

# Save the plot as an image in the current path.
plt.savefig('crisis_pv2.png')

# Optionally, display the plot.
plt.show()

    #################UNTOUCHED#################
for i in range(4):
    name=names[i]
    dict_path2= output_path / f"crisis_{name}.pkl"
    if i==0:
        save_dictionary(e, dict_path2)
    elif i==1:
        save_dictionary(l,dict_path2)
    elif i==2:
        save_dictionary(s,dict_path2)
    else:
        save_dictionary(y,dict_path2)
    print("Indi. dictionary saved.")

# save_dictionary(main, dict_path)
# print("Main Dictionary saved.")