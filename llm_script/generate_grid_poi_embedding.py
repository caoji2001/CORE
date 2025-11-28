import argparse
import pickle
import torch
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    parser.add_argument('--grid_size', type=int, default=1000)
    parser.add_argument('--top_ratio', type=float, default=0.1)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.cuda}'

    with open(f'../llm_cache/{args.dataset}/grid_poi_description_{args.grid_size}_top{int(args.top_ratio*100)}.pkl', 'rb') as file:
        llm_description = pickle.load(file)

    grid_id_and_poi_cat = []
    for grid_id, poi_info in enumerate(llm_description):
        for poi_cat in poi_info:
            grid_id_and_poi_cat.append((grid_id, poi_cat))

    model = SentenceTransformer('Qwen/Qwen3-Embedding-8B', cache_folder='../llm_cache/sbert', device=device)
    sentences = [llm_description[grid_id][poi_cat]['content'] for grid_id, poi_cat in grid_id_and_poi_cat]

    embeddings = model.encode(sentences, batch_size=1, show_progress_bar=True)
    embeddings = torch.from_numpy(embeddings)

    result = [{} for _ in range(len(llm_description))]
    for i, (grid_id, poi_cat) in enumerate(grid_id_and_poi_cat):
        result[grid_id][poi_cat] = embeddings[i]
    torch.save(result, f'../llm_cache/{args.dataset}/grid_poi_embedding_{args.grid_size}_top{int(args.top_ratio*100)}.pt')
