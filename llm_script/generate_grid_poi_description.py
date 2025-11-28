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
    api_key, model, system_prompt, user_prompt, grid_id, poi_cat = args
    open_router = OpenRouter(model, api_key, 'https://openrouter.ai/api/v1')

    try:
        return (open_router.get_output(system_prompt, user_prompt), ) + (grid_id, poi_cat)
    except Exception as e:
        print(f"Error processing task: {e}")
        return ("", grid_id, poi_cat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    parser.add_argument('--grid_size', type=int, default=1000)
    parser.add_argument('--top_ratio', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='qwen/qwen3-8b')
    parser.add_argument('--workers', type=int, default=32)

    parser.add_argument('--api_key', type=str)
    args = parser.parse_args()

    with open(f'../llm_cache/{args.dataset}/grid_poi_prompt_{args.grid_size}_top{int(args.top_ratio*100)}.pkl', 'rb') as file:
        grid_poi_llm_prompt = pickle.load(file)

    llm_output = [{} for _ in range(len(grid_poi_llm_prompt))]

    grid_poi_tasks = []
    for grid_id, poi_info in enumerate(grid_poi_llm_prompt):
        for poi_cat, prompt in poi_info.items():
            grid_poi_tasks.append((args.api_key, args.model, prompt['system_prompt'], prompt['user_prompt'], grid_id, poi_cat))

    with multiprocessing.Pool(processes=args.workers) as pool:
        grid_poi_results = []
        with tqdm(total=len(grid_poi_tasks), desc='Processing Grid-POI Prompts') as pbar:
            for result in pool.imap(process_task, grid_poi_tasks):
                grid_poi_results.append(result)
                pbar.update()

    grid_poi_prompt_tokens = 0
    grid_poi_completion_tokens = 0
    for content, grid_id, poi_cat in grid_poi_results:
        llm_output[grid_id][poi_cat] = content

    with open(f'../llm_cache/{args.dataset}/grid_poi_description_{args.grid_size}_top{int(args.top_ratio*100)}_new.pkl', 'wb') as file:
        pickle.dump(llm_output, file)
