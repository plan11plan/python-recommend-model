import torch
import numpy as np
from config import device
import os
from gmf import GMF
from mlp import MLP
from neumf import NeuMF
from data import MakeCFDataSet
from config import config
import queue

input_queue = None
output_queue = None


def set_input_queue(queue):
    global input_queue
    input_queue = queue


def get_output_queue(queue):
    global output_queue
    output_queue = queue


def recommend_poi(model, make_cf_data_set, top_k=10):
    """
    주어진 TRAVEL_ID에 대해 POI_ID와 점수를 추천하는 메소드.
    """
    global input_queue, output_queue

    travel_id = input_queue.get()

    user_idx = make_cf_data_set.user_encoder.get(travel_id)
    if user_idx is None:
        output_queue.put([])
        return

    all_items = make_cf_data_set.exist_items
    users = torch.tensor([user_idx] * len(all_items)).to(device)
    items = torch.tensor(all_items).to(device)

    with torch.no_grad():
        output = model(users, items)

    top_k_indices = output.argsort(descending=True)[:top_k]
    recommended_poi_ids = [all_items[i] for i in top_k_indices.cpu().numpy()]
    recommended_scores = output[top_k_indices].cpu().numpy()

    original_poi_ids = [make_cf_data_set.item_decoder[poi_id] for poi_id in recommended_poi_ids]

    # NaN 값 처리
    recommendations = []
    for poi_id, score in zip(original_poi_ids, recommended_scores):
        if np.isnan(score):
            score = 0.0  # 또는 다른 기본값으로 설정
        if isinstance(poi_id, float) and np.isnan(poi_id):
            poi_id = "Unknown"  # 또는 다른 기본값으로 설정
        recommendations.append((str(poi_id), float(score)))

    output_queue.put(recommendations)