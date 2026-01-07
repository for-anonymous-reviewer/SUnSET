import os
import json
import argparse
import shutil
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

from utils import is_valid_date
from params import TARGET_KEYWORDS


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Alibaba-NLP/gte-modernbert-base")  # gte-modernbert-base, gte-qwen2-7b-
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--client_name", type=str, default="milvus_set.db")
    parser.add_argument("--dataset", type=str, default='t17')
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--set_path", type=str)
    parser.add_argument("--set_filename", type=str, default='set.jsonl')
    args = parser.parse_args()
    return args


class MilvusAPI():
    def __init__(self, args):
        self.embedding_dim = args.embedding_dim
        self.dataset = args.dataset
        self.input_path = Path(args.input_path) / args.dataset
        self.set_path = args.set_path
        self.set_filename = args.set_filename
        # connect to milvus cient
        self.client = MilvusClient(args.client_name)
        # define embedding model
        self.model = SentenceTransformer(args.model_path, trust_remote_code=True)
        self.model.to('cuda')


    def create_schema(self, keyword: str):
        # Create Milvus dataset schema
        schema = MilvusClient.create_schema(
            enable_dynamic_field=True
        )

        # add fields to schema
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=512)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="summary", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="time", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="unix_time", datatype=DataType.INT64)
        schema.add_field(field_name="url", datatype=DataType.VARCHAR,max_length=1024)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)

        schema.verify()

        # Prepare index parameters
        index_params = self.client.prepare_index_params()

        # Add indexes
        index_params.add_index(
            field_name="embedding",
            index_name="dense_index",
            index_type="FLAT",
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=keyword,
            schema=schema,
            index_params=index_params
        )


    def insert_index(self, keyword: str):
        data_list = []
        print(f"Processing {keyword}")
        with open(Path(self.input_path) / keyword / self.set_filename, 'r') as f:
            data = [json.loads(line) for line in f]
        
        for i, node in tqdm(enumerate(data), total=len(data)):
            for j, (time, item) in enumerate(node['set'].items()):
                title = ''
                content = node['text']
                summary = item['event']
                # check whether summary is a valid string; if not, try to fix
                if not isinstance(summary, str):
                    print(f"Format error for article {i} set {j}: {summary}")
                    continue
                stakeholders = item['stake']

                # check whether date is valid
                if not is_valid_date(time):
                    print(f"Event date not valid for article {i} set {j}: {time}")
                    continue

                # convert time string to unix timestamp
                unix_time = int(datetime.strptime(time, '%Y-%m-%d').timestamp())
                url = ''
                source = f'{self.dataset}-{keyword}-summary-{i}-set-{j}'

                text_to_embed = f"{time}: {summary}. Stakeholders: "
                for stake in stakeholders:
                    text_to_embed+= f"{stake}, "
                text_to_embed = text_to_embed.strip(", ")

                # Get embedding
                dense = self.model.encode(summary)
                result = {
                    'title': title,
                    'content': content, 
                    'summary': summary,
                    'time': time,
                    'unix_time': unix_time,
                    'url': url,
                    'source': source,
                    'embedding': dense,
                    'stakeholders': stakeholders
                }    

                data_list.append(result)

        res = self.client.insert(
            collection_name=keyword,
            data=data_list
        )
        print("Insert index done!")

    def move_set_data(self):
        set_input_path = Path(self.set_path) / self.dataset
        for filename in os.listdir(set_input_path):
            src_path = set_input_path / filename
            keyword, _ = src_path.stem.split('_')
            dst_path = self.input_path / keyword / self.set_filename
            shutil.copy(src_path.as_posix(), dst_path.as_posix())

    def run(self):
        # Move SET data to input path
        self.move_set_data()
        for keyword, index in tqdm(TARGET_KEYWORDS[self.dataset]):
            self.create_schema(keyword)
            self.insert_index(keyword)


def main():
    args = parse_arguments()
    api = MilvusAPI(args)
    api.run()

if __name__=="__main__":
    main()
    
