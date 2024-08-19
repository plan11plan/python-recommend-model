import torch
import numpy as np
from config import device

def train(model, data_loader, criterion, optimizer, make_cf_data_set):
    model.train()
    loss_val = 0

    for users, items in data_loader:
        neg_users, neg_items = make_cf_data_set.neg_sampling(users.numpy().tolist())

        all_users = torch.concat([users, torch.tensor(neg_users)]).to(device)
        all_items = torch.concat([items, torch.tensor(neg_items)]).to(device)

        optimizer.zero_grad()

        output = model(all_users, all_items)
        pos_output, neg_output = torch.split(output, [len(users), len(neg_users)])
        pos_output = torch.concat([pos_output.view(-1, 1), pos_output.view(-1, 1), pos_output.view(-1, 1)], dim = 1).view(-1)
        loss = criterion(pos_output, neg_output)

        loss.backward()
        optimizer.step()

        loss_val += loss.item()

    loss_val /= len(data_loader)

    return loss_val

def get_ndcg(pred_list, true_list):
    ndcg = 0
    for rank, pred in enumerate(pred_list):
        if pred in true_list:
            ndcg += 1 / np.log2(rank + 2)
    return ndcg

# 대회 메트릭인 recall과 동일
def get_hit(pred_list, true_list):
    hit_list = set(true_list) & set(pred_list)
    hit = len(hit_list) / len(true_list)
    return hit

def evaluate(model, user_train, user_valid, make_cf_data_set):
    model.eval()

    NDCG = 0.0 # NDCG@10
    HIT = 0.0 # HIT@10

    all_users = make_cf_data_set.exist_users
    all_items = make_cf_data_set.exist_items
    with torch.no_grad():
        for user in all_users:
            users = [user] * len(all_items)
            users, items = torch.tensor(users).to(device), torch.tensor(all_items).to(device)

            output = model(users, items)
            output = output.softmax(dim = 0)
            output[user_train[user]] = -1.

            uv = user_valid[user]
            up = output.argsort()[-10:].cpu().numpy().tolist()

            NDCG += get_ndcg(pred_list = up, true_list = uv)
            HIT += get_hit(pred_list = up, true_list = uv)

    NDCG /= len(all_users)
    HIT /= len(all_users)

    return NDCG, HIT