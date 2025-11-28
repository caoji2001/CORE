import random
import ast
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def convert_highway(value):
    try:
        lst = ast.literal_eval(value)
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
    except (ValueError, SyntaxError):
        pass
    return value

def dict_to_namespace(d):
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_namespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

def info_nce_loss(z1, z2, temperature):
    assert z1.size() == z2.size()
    B = z1.size(0)

    features = torch.concat([z1, z2], dim=0)

    labels = torch.concat([torch.arange(B) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(z1.device)

    similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)

    mask = torch.eye(B*2, dtype=torch.bool).to(z1.device)
    labels = labels[~mask].view(B*2, -1)
    similarity_matrix = similarity_matrix[~mask].view(B*2, -1)

    positives = similarity_matrix[labels.bool()].view(B*2, -1)
    negatives = similarity_matrix[~labels.bool()].view(B*2, -1)

    logits = torch.concat([positives, negatives], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.int64).to(z1.device)

    logits /= temperature

    loss = F.cross_entropy(logits, labels)
    return loss

def is_integer_lane(x):
    if pd.isna(x):
        return False
    if isinstance(x, int) and 1 <= x <= 4:
        return True
    if isinstance(x, str) and x.isdigit() and 1 <= eval(x) <= 4:
        return True
    return False
