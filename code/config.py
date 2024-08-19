from box import Box
import torch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

config = {
    'data_path' : os.path.join(current_dir, '..', 'dataset'),  # 데이터 경로
    'model_path' : os.path.join(current_dir, '..', 'model'),    # 모델 저장 경로
    
    'gmf_path' : "GMF3.pt",
    'mlp_path' : "MLP3.pt",
    'nmf_path' : "NMF3.pt",
    'num_epochs' : 5,
    'lr' : 0.005,
    'batch_size' : 1024,

    "num_factor" : 512,
    "num_layers" : 3,
    "dropout" : 0.2,

    'valid_samples' : 1, # 검증에 사용할 sample 수
    'seed' : 22,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Box(config)