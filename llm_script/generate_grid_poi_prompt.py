import argparse
import pickle
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['Beijing', 'Chengdu', 'Xian', 'Porto'])
    parser.add_argument('--grid_size', type=int, default=1000)
    parser.add_argument('--top_ratio', type=float, default=0.1)
    args = parser.parse_args()

    dataset = args.dataset
    grid_size = args.grid_size
    top_ratio = args.top_ratio

    with open(f'../llm_cache/{dataset}/grid_poi_data_{grid_size}_top{int(top_ratio*100)}.pkl', 'rb') as file:
        grid_poi = pickle.load(file)
    poi = pd.read_csv(f'../data/{dataset}/poi/poi.csv')

    dataset2name = {
        'Beijing': '北京市',
        'Chengdu': '成都市',
        'Xian': '西安市',
    }

    llm_prompt = [{} for _ in range(len(grid_poi))]

    for i in range(len(grid_poi)):
        for poi_cat in grid_poi[i]:
            selected_poi = poi.loc[grid_poi[i][poi_cat]]

            if dataset == 'Porto':
                system_prompt = f'You are a resident living in Porto, familiar with the local transportation network and surrounding POI information.'

                user_prompt = f'In a {grid_size}-meter × {grid_size}-meter area of Porto, POIs of the type [{poi_cat}] exhibit significant clustering characteristics. Data analysis reveals that the number of [{poi_cat}]-category POIs in this region ranks within the top 10% in Porto. Further subdivision of these [{poi_cat}]-category POIs shows the following:\n\n'

                grouped = selected_poi.groupby('subcategory')
                for group_name, group_data in grouped:
                    poi_name_list = [x for x in group_data['name'] if not pd.isna(x)]

                    user_prompt += f'{len(group_data)} [{group_name}]'

                    if len(poi_name_list) == 0:
                        user_prompt += '.\n'
                    elif len(group_data) == 1:
                        user_prompt += f', its POI name is: {poi_name_list[0]}.\n'
                    elif len(group_data) <= 3:
                        user_prompt += f', their POI names are: {", ".join(poi_name_list[:3])}.\n'
                    else:
                        user_prompt += f', their POI names are: {", ".join(poi_name_list[:3])}, etc.\n'

                    user_prompt += '\n'

                user_prompt += f'Please use concise language to characterize the relevant features of this {grid_size}-meter × {grid_size}-meter area, which contains a high concentration of {poi_cat}-type POIs.'

                llm_prompt[i][poi_cat] = {
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt
                }

            else:
                system_prompt = f'你是一位生活在{dataset2name[dataset]}的居民，熟悉当地交通路网和周围POI的相关信息。'

                user_prompt = f'在{dataset2name[dataset]}某{grid_size}米×{grid_size}米的区域内，类型为【{poi_cat}】的POI呈现出显著集聚特征。经数据分析显示，这片区域内【{poi_cat}】类POI数量位居{dataset2name[dataset]}的前10%。对这些【{poi_cat}】类POI进一步进行细分，其中有：\n\n'

                grouped = selected_poi.groupby('中类')
                for group_name, group_data in grouped:
                    poi_name_list = group_data['name'].tolist()
                    if len(poi_name_list) == 1:
                        user_prompt += f'{len(poi_name_list)}个【{group_name}】，它的POI名称为：{poi_name_list[0]}。\n'
                    else:
                        user_prompt += f'{len(poi_name_list)}个【{group_name}】，它们的POI名称为：'
                        user_prompt += '、'.join(poi_name_list[:5])
                        if len(poi_name_list) > 5:
                            user_prompt += '等'
                        user_prompt += '。\n'
                    user_prompt += '\n'

                user_prompt += f'请使用凝练的语言，描述这个聚集了大量【{poi_cat}】类POI的{grid_size}米×{grid_size}米的区域的相关特征。'

                llm_prompt[i][poi_cat] = {
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt
                }

    with open(f'../llm_cache/{dataset}/grid_poi_prompt_{grid_size}_top{int(top_ratio*100)}_new.pkl', 'wb') as file:
        pickle.dump(llm_prompt, file)
