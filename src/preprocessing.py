import re
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Dict

from utils import is_valid_date
from params import TARGET_KEYWORDS

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='t17')
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--set_path", type=str)
    parser.add_argument("--kg_path", type=str, default='')
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--set_surfix", type=str, default='set.jsonl')
    return parser.parse_args()

# def stake_replacing(d, stake_dict):
#     processed_text = d['text']
#     processed_summary = d['summary']
#     ori_stake = set()
#     for set_name, set_item in d['set'].items():
#         ori_stake.update(set_item['stake'])

#     for stake_word in ori_stake:
#         if stake_word not in stake_dict:
#             continue
#         stake_id = stake_dict[stake_word]
#         if re.match(r"Q\d+", stake_id):
#             processed_text = processed_text.replace(stake_word, stake_id)
#             processed_summary = processed_summary.replace(stake_word, stake_id)

#     d['processed_text'] = processed_text
#     d['processed_summary'] = processed_summary
#     return d


def get_common_time(d):
    event_date_counter = Counter()
    for item in d['set'].values():
        if is_valid_date(item['time']):
            event_date_counter[item['time']]+=1

    if event_date_counter:
        d['event_time'] = event_date_counter.most_common(1)[0][0]
    else:
        d['event_time'] = d['time'].split('T')[0]
    return d


def get_docs(set_data, keyword, gt_timelines, dataset, output_path):
    output_path = output_path / keyword
    tl_idx = 0
    for gt_timeline in gt_timelines:
        if gt_timeline==[]:
            continue

        final_output_path = output_path / str(tl_idx)
        final_output_path.mkdir(exist_ok=True, parents=True)

        start, end = datetime.fromisoformat(gt_timeline[0][0].split('T')[0]), datetime.fromisoformat(gt_timeline[-1][0].split('T')[0])

        docs = []
        id2pos = {}

        for i, d in enumerate(set_data):
            for j, (time, item) in enumerate(d['set'].items()):
                # check for format errors
                if not is_valid_date(time):
                    continue
                if not isinstance(item['event'], str):
                    print(f"Format error: {item['event']}")
                    continue

                event_date = datetime.strptime(time, '%Y-%m-%d')

                if event_date >= start and event_date <= end:
                    result = {
                        'source': f"{dataset}-{keyword}-summary-{i}-set-{j}",
                        'stake': item['stake'],
                        'steak': item['steak'],
                        'text': item['event'],
                        'timestamp': time,
                        'id': len(docs)
                    }
                    id2pos[result['source']] = len(docs)
                    docs.append(result)

        with open(final_output_path / "docs.pickle", "wb") as f:
            pickle.dump(docs, f)
        with open(final_output_path / "id2pos.pickle", "wb") as f:
            pickle.dump(id2pos, f)
        tl_idx+=1


def get_event_output(set_data, keyword, output_path):
    output_path.mkdir(exist_ok=True, parents=True)

    save_path = output_path / f"{keyword}_events.jsonl"

    index = 0
    for i, d in enumerate(set_data):
        for time, item in d['set'].items():
            # check for format errors
            if not isinstance(item['event'], str):
                print(f"Format error: {item['event']}")
                continue

            result = {
                "keyword": keyword,
                "title": "",
                "date": time,
                "llm": item['event'],
                "index": index
            }
            index+=1
            with open(save_path, "a") as f:
                f.write(json.dumps(result) + "\n")


def kg_resolution(set_data: List[Dict], keyword: str, file_path: Path, kg_path: Path):
    if not kg_path:
        print(f"Knowledge graph path not provided for {keyword}")
        return
    # load knowledge graph
    with open(kg_path, 'r') as f:
        kg = pickle.load(f)
    
    for i, d in enumerate(set_data):
        if 'set' not in d:
            continue

        for event_date, item in d['set'].items():
            stakes = item['stake']
            item['steaks'] = []
            for stake in stakes:

                if stake.lower() in kg:
                    item['steaks'].append(kg[stake.lower()])
                else:
                    item['steaks'].append(stake.lower())
        import pdb; pdb.set_trace()
    

def main():
    args = parse_arguments()

    dataset = args.dataset
    input_path = Path(args.input_path) / dataset
    set_path = Path(args.set_path) / dataset
    kg_path = Path(args.kg_path) if args.kg_path!='' else None
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)

    for keyword, index in TARGET_KEYWORDS[dataset]:
        print(f"Processing {keyword}")
        file_path = set_path / f"{keyword}_{args.set_surfix}"
        with open(file_path, 'r') as f:
            set_data = [json.loads(line) for line in f]

        source_path = input_path / keyword
        with open(source_path / 'timelines.jsonl', 'r') as f:
            gt_timelines = [json.loads(line) for line in f]

        if kg_path:
            kg_resolution(set_data, keyword, file_path, kg_path)

        get_event_output(set_data, keyword, output_path / "event_outputs" / dataset)
        get_docs(set_data, keyword, gt_timelines, dataset, output_path / "timeline_outputs" / dataset / "0/is_incremental")


if __name__=="__main__":
    main()
    
