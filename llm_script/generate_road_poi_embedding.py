import argparse
import pickle
import torch
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    parser.add_argument('--max_dis', type=int, default=100)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.cuda}'

    with open(f'../llm_cache/{args.dataset}/road_poi_description_{args.max_dis}.pkl', 'rb') as file:
        llm_description = pickle.load(file)

    model = SentenceTransformer('Qwen/Qwen3-Embedding-8B', cache_folder='../llm_cache/sbert', device=device)
    sentences = [llm_description[i]['content'] for i in range(len(llm_description))]

    embeddings = model.encode(sentences, batch_size=1, show_progress_bar=True)
    embeddings = torch.from_numpy(embeddings)

    torch.save(embeddings, f'../llm_cache/{args.dataset}/road_poi_embedding_{args.max_dis}.pt')
