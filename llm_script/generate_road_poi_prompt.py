import argparse
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['Beijing', 'Chengdu', 'Xian', 'Porto'])
    parser.add_argument('--max_dis', type=int, default=100)
    args = parser.parse_args()

    dataset = args.dataset
    max_dis = args.max_dis

    poi = pd.read_csv(f'../data/{dataset}/poi/poi.csv')

    with open(f'../llm_cache/{dataset}/road_poi_data_{max_dis}.pkl', 'rb') as file:
        nearby_poi = pickle.load(file)

    dataset2name = {
        'Beijing': '北京市',
        'Chengdu': '成都市',
        'Xian': '西安市',
    }

    if dataset == 'Porto':
        poi_cat_list = poi['category'].unique().tolist()
    else:
        poi_cat_list = ['购物服务', '餐饮服务', '公司企业', '生活服务', '交通设施服务', '科教文化服务',\
                        '商务住宅', '政府机构及社会团体', '金融保险服务', '医疗保健服务', '体育休闲服务',\
                        '住宿服务', '汽车服务', '风景名胜']
    poi2idx = {cat: idx for idx, cat in enumerate(poi_cat_list)}

    road_poi_prompt_list = []
    for nearby_poi_per_road_segment in tqdm(nearby_poi, total=len(nearby_poi)):
        poi_cat_cnt = np.zeros(len(poi_cat_list), dtype=np.int64)
        poi_sub_cat_name = [dict() for _ in range(len(poi_cat_list))]

        for poi_idx in nearby_poi_per_road_segment:
            if dataset == 'Porto':
                poi_cat_cnt[poi2idx[poi.at[poi_idx, 'category']]] += 1

                if poi.at[poi_idx, 'subcategory'] not in poi_sub_cat_name[poi2idx[poi.at[poi_idx, 'category']]]:
                    poi_sub_cat_name[poi2idx[poi.at[poi_idx, 'category']]][poi.at[poi_idx, 'subcategory']] = []
                poi_sub_cat_name[poi2idx[poi.at[poi_idx, 'category']]][poi.at[poi_idx, 'subcategory']].append(poi.at[poi_idx, 'name'])
            else:
                if poi.at[poi_idx, '大类'] not in poi2idx:
                    continue

                poi_cat_cnt[poi2idx[poi.at[poi_idx, '大类']]] += 1

                if poi.at[poi_idx, '中类'] not in poi_sub_cat_name[poi2idx[poi.at[poi_idx, '大类']]]:
                    poi_sub_cat_name[poi2idx[poi.at[poi_idx, '大类']]][poi.at[poi_idx, '中类']] = []
                poi_sub_cat_name[poi2idx[poi.at[poi_idx, '大类']]][poi.at[poi_idx, '中类']].append(poi.at[poi_idx, 'name'])

        if dataset == 'Porto':
            system_prompt = f'You are a resident living in Porto, familiar with the local transportation network and surrounding POI information.'
        else:
            system_prompt = f'你是一位生活在{dataset2name[dataset]}的居民，熟悉当地交通路网和周围POI的相关信息。'

        if np.sum(poi_cat_cnt) > 0:
            if dataset == 'Porto':
                user_prompt = f'Among the road segments, there is one where the following POIs are located within a radius of {max_dis} meters around it.\n\n'
            else:
                user_prompt = f'其中有一条路段，其周围{max_dis}米的范围内有以下这些POI。\n\n'

            for i in range(len(poi_cat_list)):
                if poi_cat_cnt[i] > 0:
                    if dataset == 'Porto':
                        user_prompt += f'{poi_cat_cnt[i]} POIs of the type [{poi_cat_list[i]}]. Further subdivide their POI categories, including:\n'
                        for k, v in poi_sub_cat_name[i].items():
                            assert len(v) >= 1

                            poi_name_list = [x for x in v if not pd.isna(x)]

                            user_prompt += f'{len(v)} [{k}]'
                            if len(poi_name_list) == 0:
                                user_prompt += '.\n'
                            elif len(v) == 1:
                                user_prompt += f', its POI name is: {poi_name_list[0]}.\n'
                            elif len(v) <= 3:
                                user_prompt += f', their POI names are: {", ".join(poi_name_list[:3])}.\n'
                            else:
                                user_prompt += f', their POI names are: {", ".join(poi_name_list[:3])}, etc.\n'
                    else:
                        user_prompt += f'{poi_cat_cnt[i]}个类型为【{poi_cat_list[i]}】的POI。进一步对其POI类别进行细分，其中有：\n'
                        for k, v in poi_sub_cat_name[i].items():
                            assert len(v) >= 1

                            if len(v) == 1:
                                user_prompt += f'{len(v)}个【{k}】，它的POI名称为：{v[0]}。\n'
                            else:
                                user_prompt += f'{len(v)}个【{k}】，它们的POI名称为：'
                                user_prompt += '、'.join(v[:3])
                                if len(v) > 3:
                                    user_prompt += '等'
                                user_prompt += '。\n'
                    user_prompt += '\n'
        else:
            if dataset == 'Porto':
                user_prompt = f'There is a road segment where there are no POIs within a range of {max_dis} meters around it.\n\n'
            else:
                user_prompt = f'其中有一条路段，其周围{max_dis}米的范围内没有任何POI。\n\n'

        if dataset == 'Porto':
            user_prompt += f'Using POI information within {max_dis} meters of this road segment, concisely characterize the segment’s relevant attributes.'
        else:
            user_prompt += f'请根据这个路段周围{max_dis}米范围内的POI信息，使用凝练的语言描述这个路段的相关特征。'

        road_poi_prompt_list.append({
            'system_prompt': system_prompt,
            'user_prompt': user_prompt
        })

    with open(f'../llm_cache/{dataset}/road_poi_prompt_{max_dis}_new.pkl', 'wb') as file:
        pickle.dump(road_poi_prompt_list, file)
