import argparse
import pickle
import json
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import multiprocessing


class OpenRouter:
    def __init__(self, model, api_key, base_url):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def get_output(self, system_prompt, user_prompt):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            extra_body={
                "reasoning": {
                    "exclude": True,
                    "effort": "none"
                },
            },
        )

        return completion.choices[0].message.content


def process_task(args):
    i, api_key, model, system_prompt, user_prompt = args
    open_router = OpenRouter(model, api_key, 'https://openrouter.ai/api/v1')

    try:
        return i, open_router.get_output(system_prompt, user_prompt)
    except Exception as e:
        print(f"index: {i}, error processing task: {e}")
        return i, ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    parser.add_argument('--max_dis', type=int, default=100)
    parser.add_argument('--model', type=str, default='qwen/qwen3-8b')
    parser.add_argument('--workers', type=int, default=128)

    parser.add_argument('--api_key', type=str)
    args = parser.parse_args()

    with open(f'../llm_cache/{args.dataset}/road_poi_prompt_{args.max_dis}.pkl', 'rb') as file:
        llm_prompt = pickle.load(file)

    road_poi_tasks = [
        (i, args.api_key, args.model, p['system_prompt'], p['user_prompt'])
        for i, p in enumerate(llm_prompt)
    ]

    with multiprocessing.Pool(processes=args.workers) as pool:
        road_poi_results = [None for _ in range(len(road_poi_tasks))]
        with tqdm(total=len(road_poi_tasks), desc='Processing Road-POI Prompts') as pbar:
            for i, result in pool.imap_unordered(process_task, road_poi_tasks):
                road_poi_results[i] = result
                pbar.update()

    road_poi_output_list = []
    for content in road_poi_results:
        road_poi_output_list.append(content)

    with open(f'../llm_cache/{args.dataset}/road_poi_description_{args.max_dis}_new.pkl', 'wb') as file:
        pickle.dump(road_poi_output_list, file)
