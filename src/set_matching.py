import os
import pickle
import json
import argparse
import time
import numpy as np
import heapq
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
from pymilvus import MilvusClient
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from params import search_params, OUTPUT_FIELDS_OPTIONS, TARGET_KEYWORDS


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Alibaba-NLP/gte-modernbert-base")  # gte-modernbert-base, gte-qwen2-7b-
    parser.add_argument("--vector_db_path", type=str, default='')
    parser.add_argument("--dataset", type=str, default='t17')
    parser.add_argument("--gt_path", type=str)
    parser.add_argument("--scoring_path", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_common_stake", type=int, default=1)
    parser.add_argument("--time_window", type=int, default=0)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0) #(default=0)==EM only #BETA IS NOT WORKING!
    parser.add_argument("--pay", type=float, default=2)
    parser.add_argument("--directional", type=bool, default=False) #false: double add
    return parser.parse_args()


class SETProcessor():
    def __init__(self, args):
        self.gt_path = Path(args.gt_path)
        self.input_path = Path(args.input_path)
        self.output_path = Path(args.output_path)
        self.scoring_path = Path(args.scoring_path)
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.min_common_stake = args.min_common_stake
        self.time_window = args.time_window
        self.top_n = args.top_n
        self.topic='' #topic of the dataset
        self.p= args.pay #1 or 2 only, p version
        self.pdict={}
        self.rdict={}
        self.directional=args.directional 

        # Vector DB
        self.embedding_model = SentenceTransformer(args.model_path, trust_remote_code=True).to('cuda')
        if args.vector_db_path:
            self.vectordb = MilvusClient(os.path.join(args.vector_db_path, f"{self.dataset}.db"))
        else:
            self.vectordb = MilvusClient(f"{self.dataset}.db")
        self.output_fields = OUTPUT_FIELDS_OPTIONS[self.dataset]

        # weights for combining cosine + ID‐bonus
        self.alpha = 1 #cosine score will retain its previous weight
        self.beta = args.beta #since uses summation with a max of 5, we use 0.2 so the sum will be 1 at max

    def load_docs(self, keyword, tl_idx):
        with open(self.input_path / self.dataset / "0/is_incremental" / keyword / str(tl_idx) / "docs.pickle", 'rb') as f:
            docs = pickle.load(f)
        with open(self.input_path / self.dataset / "0/is_incremental" / keyword / str(tl_idx) / "id2pos.pickle", 'rb') as f:
            id2pos = pickle.load(f)
        return docs, id2pos

    def load_gt_timelines(self, keyword):
        source_path = self.gt_path / self.dataset / keyword
        with open(source_path / 'timelines.jsonl', 'r') as f:
            gt_timelines = [json.loads(line) for line in f]
        return gt_timelines

    def scored(self):
        base = self.scoring_path
        # pick the right p-path
        if self.p == 1:
            ppath = base / 'p1' / f"{self.dataset}_pv1_2.json"
        elif self.p == 2:
            ppath = base / 'p2' / f"{self.dataset}_{self.topic}.json"
        else:
            raise ValueError(f"Invalid p={self.p}, must be 1 or 2")
        # the r-path
        rpath = base / 'r_dts' / f"{self.dataset}_{self.topic}.json"
        with open(ppath, "r", encoding="utf-8") as f:
            pdict = json.load(f)
        with open(rpath, "r", encoding="utf-8") as f:
            rdict = json.load(f)
        return pdict, rdict
    
    @staticmethod
    def damp_tanh(x, L=10):
        return np.tanh(x / L)

    def score(self, stakes, cos): #stakes is list of EM stakes
        curr=cos*self.alpha #current alpha uses full value
        for stake in stakes:
            p=self.pdict[stake]
            r=SETProcessor.damp_tanh(self.rdict[stake])
            curr+=self.beta*p*r
        return curr


    def em_matching(self, clusters, docs, graph,  distances, max_date_diff=0):
        reclus={}
        doc1 = clusters[0]
        stake_key = 'steak' if 'steak' in docs[doc1] else 'stake'
        doc1_stakes = docs[doc1][stake_key]
        doc1_date = datetime.fromisoformat(docs[doc1]['timestamp'])
        for i in range(len(clusters)-1): #i be 0-19 to get items 1-20
            doc2 = clusters[i + 1]
            doc2_stakes = docs[doc2][stake_key]
            common_stakes = [s for s in doc1_stakes if s in doc2_stakes]
            new_score=self.score(common_stakes, distances[i])
            reclus[doc2]=new_score

        top20_keys = heapq.nlargest(self.top_n, reclus, key=reclus.get) 
        for i in range(len(top20_keys)):
            doc2 = top20_keys[i]
            doc2_stakes = docs[doc2][stake_key]
            common_stakes = [s for s in doc1_stakes if s in doc2_stakes] 
            doc1_date = datetime.fromisoformat(docs[doc1]['timestamp'])
            doc2_date = datetime.fromisoformat(docs[doc2]['timestamp'])
            date_diff = abs(doc1_date - doc2_date) <= timedelta(days=self.time_window)
            if len(common_stakes) >= self.min_common_stake and date_diff:
                if not self.directional:
                    graph[doc2].add(doc1)
                graph[doc1].add(doc2)
        return graph


    def em_clustering(self, doc_id_range, graph):
        visited = set()
        clusters = []

        for doc_idx in range(doc_id_range):
            if doc_idx not in visited:
                cluster = []

                stack = [doc_idx]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        cluster.append(node)
                        for neighbor in graph[node]:
                            if neighbor not in visited:
                                stack.append(neighbor)
                clusters.append(cluster)

        result = {}
        for i, cluster in enumerate(clusters, -1):
            result[f'{i}'] = cluster

        return result


    def run(self):
        s = time.time()
        output_path = self.output_path / self.dataset / "0/is_incremental"
        for keyword, index in TARGET_KEYWORDS[self.dataset]:
            self.topic=keyword
            print(f'DATASET: {self.dataset}')
            print(f'KEYWORD(topic): {keyword}')
            self.pdict, self.rdict= self.scored()
            tl_idx = 0
            gt_timelines = self.load_gt_timelines(keyword)
            for gt_timeline in tqdm(gt_timelines):
                if gt_timeline==[]:
                    continue
                print(f"Processing {keyword} timeline {tl_idx}")
                docs, id2pos = self.load_docs(keyword, tl_idx)

                # create date filter
                start = int(datetime.fromisoformat(gt_timeline[0][0].split('T')[0]).timestamp())
                end = int(datetime.fromisoformat(gt_timeline[-1][0].split('T')[0]).timestamp())
                filter = f"unix_time >= {start} and unix_time <= {end}"

                queries = []
                for doc in docs:
                    query = f"{doc['timestamp']}: {doc['text']}. Stakeholders: " + ", ".join(doc['stake'])
                    queries.append(query)

                # Batch encode
                query_vectors = self.embedding_model.encode(
                    queries, batch_size=self.batch_size, convert_to_numpy=True
                )

                # Batch search
                all_results = self.vectordb.search(
                    collection_name=keyword,
                    data=query_vectors,
                    anns_field="embedding",
                    filter=filter,
                    search_params=search_params,
                    limit=self.top_n*3, #take 60; 3 of 20 nearest neighbours
                    output_fields=self.output_fields,
                )

                graph = defaultdict(set)

                for doc_idx, res in enumerate(all_results):
                    clusters = [id2pos[c['entity']['source']] for c in res]
                    distances = np.array([h.distance for h in res if h['entity']['source'] != docs[doc_idx]['source']], dtype=float) #cos. sim.
                    graph = self.em_matching(clusters, docs, graph, distances )

                event_pool = self.em_clustering(len(docs), graph)

                final_output_path = output_path / keyword / str(tl_idx)
                final_output_path.mkdir(exist_ok=True, parents=True)
                with open(final_output_path / "event_pool.pickle", 'wb') as f:
                    pickle.dump(event_pool, f)
                tl_idx+=1
        e = time.time()
        print(f"SET Matching takes {e-s} seconds")


def main():
    args = parse_arguments()
    set_processor = SETProcessor(args)
    set_processor.run()

if __name__=="__main__":
    main()
