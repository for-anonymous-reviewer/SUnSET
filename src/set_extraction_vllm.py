import json
import argparse
import time
import asyncio
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path
from openai import AsyncOpenAI, APIError, RateLimitError
from datetime import datetime
from typing import Dict, Any, List
from params import TARGET_KEYWORDS

from schema import SYSTEM_PROMPT, EVENTS_SCHEMA

def timestamp_to_date_string(timestamp):
    """Convert timestamp to YYYY-MM-DD string format"""
    try:
        if timestamp > 10**10:
            timestamp = timestamp / 1000
        dt = datetime.fromtimestamp(float(timestamp))
        return dt.strftime('%Y-%m-%d')
    except:
        # print(f"Cannot convert date format")
        return str(timestamp)

async def completion_with_llm(
        user_prompt: str, 
        llm: AsyncOpenAI, 
        model_path: str,
        max_tokens: int,
        temperature: float,
        semaphore: asyncio.Semaphore
        ):
    retries = 3
    base_delay = 2

    async with semaphore:
        for attempt in range(retries):
            try:
                chat_response = await llm.chat.completions.create(
                    model=model_path,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    timeout=60, # 适当增加超时时间
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "json_schema",
                            "schema": EVENTS_SCHEMA
                        },
                    },
                )
                generated_text = chat_response.choices[0].message.content
                
                # print("GENERATED TEXT:", generated_text[:50] + "...")
                
                try:
                    return json.loads(generated_text)["items"]
                except json.JSONDecodeError:
                    print(f"JSON Decode Error on attempt {attempt}")
                    return []

            except (RateLimitError, APIError) as e:
                if attempt == retries - 1:
                    print(f"Failed after {retries} retries: {e}")
                    return []
                
                sleep_time = base_delay * (2 ** attempt)
                # print(f"Rate limit or API error, retrying in {sleep_time}s...")
                await asyncio.sleep(sleep_time)
            except Exception as e:
                print(f"Unexpected error: {e}")
                return []
                
    return []


async def summ(llm, date_content_tuple, args, semaphore): 
    # DESIRED FORMAT: {Event_a: {event: content, stake: [person_a,person_b], place: [location_a], object: [object_a], time: datetime}
    publish_date = date_content_tuple[0]
    article_content = date_content_tuple[1]

    user_prompt = (
        f"Publication date：{publish_date}. "
        f"Article content：{article_content}. "
        "Please extract the relevant news event while adhering to the required JSON Schema. "
        "If multiple events happen on the same date, summarize them into one event. "
    )
    
    # 异步调用
    items = await completion_with_llm(
        user_prompt, 
        llm, 
        args.model_path, 
        args.max_tokens, 
        args.temperature,
        semaphore
    )

    date2events: Dict[str, Dict[str, Any]] = {}
    for ev in items:
        date_str = ev.get("date") or publish_date
        if date_str not in date2events:
            val = {
                "event": ev.get("summary", ""),
                "stake": (ev.get("stakeholders", [])),
            }
            date2events[date_str] = val

    return date2events

async def process_single_item(data, llm, args, semaphore):
    """
    处理单条数据的包装函数，用于构建最终结果结构
    """
    listabc = [data['time'], data['text']]
    # 等待 summ 执行完毕
    summary_result = await summ(llm, listabc, args, semaphore)
    
    return {
        "time": data['time'], 
        "text": data['text'], 
        "set": summary_result
    }

async def main_async():
    # 初始化异步客户端
    llm = AsyncOpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_api_base,
    )

    semaphore = asyncio.Semaphore(args.concurrency)

    for keyword, index in TARGET_KEYWORDS[args.dataset]:
        print(f"Processing {keyword}...")
        input_path = Path(args.input_path) / args.dataset / keyword
        json_file = input_path / "articles.preprocessed.jsonl"
        
        # 确保输出目录存在
        if not input_path.exists():
            print(f"Path not found: {input_path}")
            continue
            
        output_path = Path(args.output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        save_name = output_path / f"{keyword}_set.jsonl"

        if not json_file.exists():
            print(f"File not found: {json_file}")
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            data_list = [json.loads(line) for line in f]

        print(f"Total items to process: {len(data_list)}")

        # 创建任务列表
        tasks = [
            process_single_item(data, llm, args, semaphore) 
            for data in data_list
        ]

        full_context = await tqdm_asyncio.gather(*tasks)

        # 写入结果
        print("Writing results to file...")
        with open(save_name, "w", encoding="utf-8") as file:
            for entry in full_context:
                file.write(json.dumps(entry) + "\n")
        
        print(f"Finished {keyword}. Saved to {save_name}")

def main():
    asyncio.run(main_async())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='t17')
    parser.add_argument("--model_path", type=str, default='Qwen/Qwen2.5-72B')
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--openai_api_key", type=str, default="EMPTY")
    parser.add_argument("--openai_api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=64, help="Max concurrent API requests")
    args = parser.parse_args()

    main()