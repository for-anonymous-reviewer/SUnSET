import json
import pickle
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tilse.data.timelines import Timeline as TilseTimeline
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.evaluation import rouge
import operator
from evaluation import get_scores, evaluate_dates, get_average_results
from data import Dataset, get_average_summary_length
from datetime import datetime
from pprint import pprint
from tqdm import tqdm

from langchain.embeddings import SentenceTransformerEmbeddings

from params import TARGET_KEYWORDS, METRIC

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_data(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def get_average_summary_length(ref_tl):
    lens = [len(summary) for date, summary in ref_tl.dates_to_summaries.items()]
    return round(sum(lens) / len(lens))

def text_rank(sentences, embedding_func, personalization=None):
    sentence_embeddings = embedding_func.embed_documents(texts=sentences)
    cosine_sim_matrix = cosine_similarity(sentence_embeddings)
    nx_graph = nx.from_numpy_array(cosine_sim_matrix)
    scores = nx.pagerank(nx_graph, personalization=personalization)
    return sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

def get_pairs(event_pool):
    pairs, singletons = [], []
    for cluster_id, nodes in event_pool.items():
        if len(nodes) > 1:
            pairs.extend((min(nodes[i], nodes[j]), max(nodes[i], nodes[j])) for i in range(len(nodes)) for j in range(i + 1, len(nodes)))
        else:
            singletons.append(nodes[0])
    return pairs, singletons

def get_avg_score(scores):
    return sum(scores) / len(scores)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timelines_path", type=str)
    parser.add_argument("--model_path", type=str, default="/home/ma-user/work/model/gte-modernbert-base")
    parser.add_argument("--scoring_path", type=str)
    parser.add_argument("--info_save_path", type=str, default="")
    parser.add_argument("--events_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--dataset", type=str, default="t17")
    parser.add_argument("--text_rank", action='store_true')
    parser.add_argument("--raw_data_path", type=str)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--p", type=int, default=2, help="either 1 or 2")
    return parser.parse_args()

evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])

def process_event_pool(
    event_pool: Dict[str, List[int]],
    docs_map: Dict
) -> List[Tuple[int, str, int, List[List[str]]]]:

    date2stake_list = []
    for cls_id, near_doc_ids in event_pool.items():
        date = docs_map[near_doc_ids[0]]['timestamp']
        stake_set = set()
        event_num = len(near_doc_ids)
        for doc_id in near_doc_ids:
            doc = docs_map[doc_id]
            stake_set.update(doc['steak'])
        date2stake_list.append((cls_id, date, event_num, list(stake_set)))

    return date2stake_list


class PRScorer():
    def __init__(self, args, topic):
        self.dataset = args.dataset
        self.scoring_path = Path(args.scoring_path)
        self.topic = topic
        self.p = args.p
        self.beta = args.beta
        self.pdict = {}
        self.rdict = {}

    @staticmethod
    def load_dictionary(path):
        """
        Given the path to a .json file, returns its contents as a Python dict.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print("File not found.")
            return None

    @staticmethod
    def damp_tanh(x, L=10):
        return np.tanh(x / L)


    def get_pr_dict(self, p=1):
        base = self.scoring_path
        # pick the right p-path
        if p == 1:
            ppath = base / 'p1' / f"{self.dataset}_pv1_2.json"
        elif p == 2:
            ppath = base / 'p2' / f"{self.dataset}_{self.topic}.json"
        else:
            raise ValueError(f"Invalid p={self.p}, must be 1 or 2")
        # the r-path
        rpath = base / 'r_dts' / f"{self.dataset}_{self.topic}.json"
        pdict=self.load_dictionary(ppath)
        rdict=self.load_dictionary(rpath)
        return pdict, rdict


    def calc_pr_score(self, date2stake_list): #stakes is list of EM stakes: ['Q37468','Q2367','Obama','Q8374'] etc. DO NOT lowercase this list.
        self.pdict, self.rdict = self.get_pr_dict(self.p)
        results = {}
        for cls_id, date, event_num, stakes in date2stake_list:
            # C_d : event-count factor (log-scaled)
            C_d = 1.0 + math.log(event_num)
            K_d = 0
            for stake in stakes:
                p = self.pdict[stake]
                r = self.damp_tanh(self.rdict[stake])
                K_d += self.beta*p*r      #this is relevancy score, with a coefficient of Beta.

            K_d = 1.0 + K_d / len(stakes) if len(stakes) else 1

            final_score = C_d * K_d
            # final_score = C_d + K_d

            if date not in results:
                results[date] = (cls_id, final_score)
            elif date in results and final_score > results[date][1]:
                results[date] = (cls_id, final_score)
        results = [(item[0], item[1][0], item[1][1]) for item in results.items()]
        return sorted(results, key=lambda x: -x[2])


def evaluate_timeline(overall_results, trials_path, collections, embedding_func, args):
    overall_trials = sorted(os.listdir(trials_path))
    overall_r1_list, overall_r2_list, overall_d1_list = [], [], []

    for trial in overall_trials:
        print(f"Trial {trial}:")
        trial_res = process_trial(collections, trial, trials_path, args, embedding_func, overall_results)
        pprint(trial_res)
        print(f"tiral_res: {trial_res}")
        
        rouge_1 = trial_res[0]['f_score']
        rouge_2 = trial_res[1]['f_score']
        date_f1 = trial_res[2]['f_score']
        trial_save_path = os.path.join(args.output_path, trial)
        os.makedirs(trial_save_path, exist_ok=True)

        overall_r1_list.append(rouge_1)
        overall_r2_list.append(rouge_2)
        overall_d1_list.append(date_f1)

        save_json(trial_res, os.path.join(trial_save_path, 'avg_score.json'))

    save_json(overall_results, os.path.join(args.output_path, 'global_result.json'))

    avg_r1 = get_avg_score(overall_r1_list)
    avg_r2 = get_avg_score(overall_r2_list)
    avg_d1 = get_avg_score(overall_d1_list)
    final_results = {
        'rouge1': avg_r1,
        'rouge2': avg_r2,
        'dateF1': avg_d1
    }
    print(f"final result: {final_results}")

    save_json(final_results, os.path.join(args.output_path, 'average_result.json'))


def process_trial(collections, trial, trials_path, args, embedding_func, overall_results):
    results = []
    for keyword, index in tqdm(TARGET_KEYWORDS[args.dataset]):
    # for keyword, index in tqdm(collections):
        col = collections[index]
        events, content2type = load_events(os.path.join(args.events_path, f"{keyword.replace(' ', '_')}_events.jsonl"))
        for tl_index, gt_timeline in enumerate(col.timelines):
            summary_length = get_average_summary_length(TilseTimeline(gt_timeline.time_to_summaries))
            timeline_res = process_timeline(gt_timeline, summary_length, keyword, tl_index, trial, trials_path, args, embedding_func, content2type)
            (rouge_scores, date_scores, pred_timeline_dict) = timeline_res
            results.append(timeline_res)

            # Update overall results
            if keyword not in overall_results:
                overall_results[keyword] = {}
            if tl_index not in overall_results[keyword]:
                overall_results[keyword][tl_index] = []
            overall_results[keyword][tl_index].append((rouge_scores, date_scores))
    trial_res = get_average_results(results)
    return trial_res


def load_events(path):
    with open(path, 'r') as f:
        events = [json.loads(x) for x in f]
    content2type = {}
    for e in events:
        for key in ['llm', 'sentence']:
            if key in e:
                content = e[key].split(':')[-1].strip()
                content2type[content] = key
                break
    return events, content2type


def process_timeline(gt_timeline, summary_length, keyword, tl_index, trial, trials_path, args, embedding_func, content2type):
    trial_path = os.path.join(trials_path, trial, 'is_incremental', f"{keyword.replace(' ', '_')}/{tl_index}")
    # trial_path = os.path.join(trials_path, trial, 'not_incremental', f"{keyword.replace(' ', '_')}/{tl_index}")
    docs = pickle.load(open(os.path.join(trial_path, 'docs.pickle'), 'rb'))
    event_pool = pickle.load(open(os.path.join(trial_path, 'event_pool.pickle'), 'rb'))

    times = gt_timeline.times
    L = len(times)

    # Generate pairs and clusters from event pool
    pairs, _ = get_pairs(event_pool)
    pr_scorer = PRScorer(args, keyword)
    cluster_info = cluster_events(docs, pairs, pr_scorer, L)

    pred_timeline = perform_text_summarization(cluster_info, summary_length, embedding_func, args.text_rank, content2type)

    # Evaluate summarization
    ground_truth = TilseGroundTruth([TilseTimeline(gt_timeline.date_to_summaries)])
    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
    rouge_scores = get_scores(METRIC, pred_timeline, ground_truth, evaluator)
    date_scores, date_info = evaluate_dates(pred_timeline, ground_truth)
    timeline_res = (rouge_scores, date_scores, pred_timeline)
    

    if args.info_save_path:
        with open(os.path.join(args.info_save_path, f"{keyword}_{tl_index}_date_info.json"), 'w') as f:
            json.dump(date_info, f)

        pred_tl_df = {'date': [], 'summaries': []}
        for date_obj, summaries in pred_timeline.dates_to_summaries.items():
            pred_tl_df['date'].append(str(date_obj))
            pred_tl_df['summaries'].append(summaries)
        
        pred_tl_df = pd.DataFrame(pred_tl_df)
        pred_tl_df.to_csv(os.path.join(args.info_save_path, f"{keyword}_{tl_index}_pred_tl.csv"), index=False)

    return timeline_res


def cluster_events(docs, pairs, pr_scorer, top_l):
    # Build pools
    event_pool = dict()
    event2cluster = dict()
    for edge in pairs:
        event_id, event_near_id = edge[0], edge[1]
        cls_id = event2cluster.get(event_id, -1)
        neighbor_cls_id = event2cluster.get(event_near_id, -1)

        if cls_id == -1 and neighbor_cls_id == -1:
            new_cls_id = max(list(event_pool.keys()) or [-1]) + 1
            event_pool[new_cls_id] = [event_id, event_near_id]
            event2cluster[event_id] = new_cls_id
            event2cluster[event_near_id] = new_cls_id
        elif cls_id == -1 and neighbor_cls_id != -1:
            event_pool[neighbor_cls_id] = event_pool[neighbor_cls_id] + [event_id]
            event2cluster[event_id] = neighbor_cls_id
        elif cls_id != -1 and neighbor_cls_id == -1:
            event_pool[cls_id] = event_pool[cls_id] + [event_near_id]
            event2cluster[event_near_id] = cls_id
        elif cls_id != -1 and neighbor_cls_id != -1 and cls_id != neighbor_cls_id:
            event_pool[cls_id] = list(set(event_pool[cls_id] + event_pool[neighbor_cls_id]))
            del event_pool[neighbor_cls_id]
            for e_id in event_pool[cls_id]:
                event2cluster[e_id] = cls_id
        else:
            pass
    
    total_ids = [doc['id'] for doc in docs]
    for node in total_ids:
        if node in event2cluster:
            continue 
        new_cls_id = max(list(event_pool.keys()) or [-1]) + 1
        event_pool[new_cls_id] = [node]
        event2cluster[node] = new_cls_id
    
    
    clusters = list(event_pool.values())
    clusters.sort(key=len, reverse=True)

    docs_map = {}
    for doc in docs:
        id = doc['id']
        docs_map[id] = doc

    date2stake_list = process_event_pool(event_pool, docs_map)
    date_scores = pr_scorer.calc_pr_score(date2stake_list)[:top_l]

    cluster_info = []
    for date, cls_id, score in date_scores:
        cluster_docs = event_pool[cls_id]
        all_events = [docs_map[i] for i in cluster_docs]
        cluster_info.append((date, score, all_events))

    # Sort clusters by 
    cluster_info.sort(key=operator.itemgetter(0))
    return cluster_info


def perform_text_summarization(cluster_info, summary_length, embedding_func, use_text_rank, content2type):
    pred_timeline_dict = {}
    # Print clusters
    for i, info in tqdm(enumerate(cluster_info)):
        date, score, cluster_events_lst = info
        date_obj = datetime.strptime(date, '%Y-%m-%d').date()
        summary_list = []

        personalization = {}
        event_list = []
        for e in cluster_events_lst:
            text = e['text'].split(':')[-1].strip()
            event_list.append(text)
            text_idx = len(event_list) - 1
            if content2type.get(text, "sentence") == 'llm':
                personalization[text_idx] = 2.0
            else:
                personalization[text_idx] = 1.0
                
        output = []
        if use_text_rank:
            ranked_list = [s for _, s in text_rank(event_list, embedding_func, personalization)]
            num_sentences = summary_length  # or whatever number of sentences you want in your summary
            output = [s for s in ranked_list[:num_sentences]]
        else:
            output = [s for s in event_list[:num_sentences]]
        summary_list.extend(output)

        pred_timeline_dict[date_obj] = summary_list
    pred_timeline_ = TilseTimeline(pred_timeline_dict)
    return pred_timeline_


def main():
    args = parse_arguments()
    if args.info_save_path:
        os.makedirs(args.info_save_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    dataset_path = os.path.join(args.raw_data_path, args.dataset)
    dataset = Dataset(dataset_path)
    collections = dataset.collections
    model_kwargs={
        'device': 'cuda',
        'trust_remote_code': True
    }
    embedding_func = SentenceTransformerEmbeddings(model_name=args.model_path, model_kwargs=model_kwargs)

    evaluate_timeline({}, args.timelines_path, collections, embedding_func, args)

if __name__=="__main__":
    main()
