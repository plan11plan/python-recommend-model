from data import MakeCFDataSet,CFDataset
from config import config,device
from neumf import BPR_Loss,NeuMF
from gmf import GMF
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import train,evaluate
import os
from mlp import MLP

make_cf_data_set = MakeCFDataSet(config = config)
user_train, user_valid = make_cf_data_set.get_train_valid_data()
# 현재 디렉터리 출력
current_directory = os.getcwd()
print("현재 디렉터리:", current_directory)
cf_dataset = CFDataset(user_train = user_train)
data_loader = DataLoader(
    cf_dataset,
    batch_size = config.batch_size,
    shuffle = True,
    drop_last = False)


# GMF
model = GMF(
    num_user = make_cf_data_set.num_user,
    num_item = make_cf_data_set.num_item,
    num_factor = config.num_factor).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
criterion = BPR_Loss()

best_hit = 0
for epoch in range(1, config.num_epochs + 1):
    tbar = tqdm(range(1))
    for _ in tbar:
        train_loss = train(
            model = model,
            data_loader = data_loader,
            criterion = criterion,
            optimizer = optimizer,
            make_cf_data_set = make_cf_data_set
            )

        ndcg, hit = evaluate(model, user_train, user_valid, make_cf_data_set)

        if best_hit < hit:
            best_hit = hit
            torch.save(model.state_dict(), os.path.join(config.model_path, config.gmf_path))

        tbar.set_description(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')
        
        

# MLP
model = MLP(
    num_user = make_cf_data_set.num_user,
    num_item = make_cf_data_set.num_item,
    num_factor = config.num_factor,
    num_layers = config.num_layers,
    dropout = config.dropout,).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
criterion = BPR_Loss()
best_hit = 0
for epoch in range(1, config.num_epochs + 1):
    tbar = tqdm(range(1))
    for _ in tbar:
        train_loss = train(
            model = model,
            data_loader = data_loader,
            criterion = criterion,
            optimizer = optimizer,
            make_cf_data_set = make_cf_data_set
            )

        ndcg, hit = evaluate(model, user_train, user_valid, make_cf_data_set)

        if best_hit < hit:
            best_hit = hit
            torch.save(model.state_dict(), os.path.join(config.model_path, config.mlp_path))

        tbar.set_description(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')

# NewMF     
gmf = GMF(
    num_user = make_cf_data_set.num_user,
    num_item = make_cf_data_set.num_item,
    num_factor = config.num_factor).to(device)

gmf.load_state_dict(torch.load(os.path.join(config.model_path,config.gmf_path)))

mlp = MLP(
    num_user = make_cf_data_set.num_user,
    num_item = make_cf_data_set.num_item,
    num_factor = config.num_factor,
    num_layers = config.num_layers,
    dropout = config.dropout,).to(device)

mlp.load_state_dict(torch.load(os.path.join(config.model_path, config.mlp_path)))

model = NeuMF(
    GMF = gmf,
    MLP = mlp,
    num_factor = config.num_factor).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr = config.lr, momentum = 0.9)
criterion = BPR_Loss()


best_hit = 0
for epoch in range(1, config.num_epochs + 1):
    tbar = tqdm(range(1))
    for _ in tbar:
        train_loss = train(
            model = model,
            data_loader = data_loader,
            criterion = criterion,
            optimizer = optimizer,
            make_cf_data_set = make_cf_data_set
            )

        ndcg, hit = evaluate(model, user_train, user_valid, make_cf_data_set)

        if best_hit < hit:
            best_hit = hit
            torch.save(model.state_dict(), os.path.join(config.model_path, config.nmf_path))

        tbar.set_description(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')