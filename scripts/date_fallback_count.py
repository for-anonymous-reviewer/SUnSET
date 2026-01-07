import json
from pathlib import Path
from datetime import datetime

def is_same_date(publish_time_str: str, event_time_str: str) -> bool:
    """
    Compares a timestamp string (ISO 8601) and a date string (YYYY-MM-DD)
    to see if they represent the same calendar day.
    """
    try:
        # 1. Parse the publish time (e.g., "2011-01-20T00:00:00+00:00")
        # datetime.fromisoformat handles the 'T' and timezone offsets automatically
        pub_dt = datetime.fromisoformat(publish_time_str)
        
        # 2. Parse the event time (e.g., "2010-04-20")
        try:
            event_dt = datetime.strptime(event_time_str, "%Y-%m-%d")
        except:
            event_dt = datetime.fromisoformat(event_time_str)
        
        # 3. Compare only the date parts (ignoring time and timezone info)
        return pub_dt.date() == event_dt.date()
        
    except ValueError as e:
        return True
    

def main():
    root_path = Path("/mnt/disk0/y84387018/sunset/data/set/qwen2_5-7b-instruct")
    datasets = ['t17', 'crisis']

    for dataset in datasets:
        input_path = root_path / dataset
        print(f"Processing dataset: {dataset}")
        print("------------------------------------")
        all_jsonl_files = list(input_path.glob("*.jsonl"))
    
        for jsonl_file in all_jsonl_files:
            with open(jsonl_file, "r") as f:
                topic = jsonl_file.stem.split("_")[0]
                article_wise = []
                event_wise_total = []

                for line in f:
                    data = json.loads(line)
                    event_wise = []
                    publish_time = data["time"]
                    
                    for event_date, event_list in data["set"].items():
                        if is_same_date(publish_time, event_date):
                            event_wise.append(1)
                        else:
                            event_wise.append(0)

                    event_wise_total.extend(event_wise)
                    if sum(event_wise) == 0:
                        article_wise.append(0)
                    else:
                        article_wise.append(1)
                print(f"Date fallback rate for topic {topic}:")
                # print(f"article-wise: {sum(article_wise)}/{len(article_wise)}, {sum(article_wise)/len(article_wise)}")
                print(f"event-wise: {sum(event_wise_total)}/{len(event_wise_total)}, {sum(event_wise_total) / len(event_wise_total)}")

if __name__ == "__main__":
    main()