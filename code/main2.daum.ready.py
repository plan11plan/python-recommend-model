from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any
import torch

from config import device, config
import os
from gmf import GMF
from mlp import MLP
from neumf import NeuMF
from train_auto import run_recommendation_models
from recommend import recommend_poi
from data import MakeCFDataSet

train_num = int(3)
app = FastAPI()
make_cf_data_set = MakeCFDataSet(config=config)

# Initialize global models and dataset
gmf = GMF(num_user=make_cf_data_set.num_user, num_item=make_cf_data_set.num_item, num_factor=config.num_factor).to(
    device)
mlp = MLP(num_user=make_cf_data_set.num_user, num_item=make_cf_data_set.num_item, num_factor=config.num_factor,
          num_layers=config.num_layers, dropout=config.dropout).to(device)
model = NeuMF(GMF=gmf, MLP=mlp, num_factor=config.num_factor).to(device)

# Load model weights
model_path = os.path.join(config.model_path, "NMF3.pt")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

is_updating = False


class UpdateRequest(BaseModel):
    train_num: int
    data_json: Dict[str, Any]


class RecommendRequest(BaseModel):
    travel_id: str


def update_model_task(request: UpdateRequest):
    global model, make_cf_data_set, is_updating, train_num
    is_updating = True
    try:
        train_num = int(train_num) + 1
        update_make_cf_data_set = run_recommendation_models(config, device, data_json=request.data_json,
                                                            train_num=train_num)

        update_gmf = GMF(num_user=make_cf_data_set.num_user, num_item=make_cf_data_set.num_item,
                         num_factor=config.num_factor).to(device)
        update_mlp = MLP(num_user=make_cf_data_set.num_user, num_item=make_cf_data_set.num_item,
                         num_factor=config.num_factor, num_layers=config.num_layers, dropout=config.dropout).to(device)
        update_model = NeuMF(GMF=update_gmf, MLP=update_mlp, num_factor=config.num_factor).to(device)

        update_model_path = os.path.join(config.model_path, "NMF", f"{train_num}.pt")
        update_model.load_state_dict(torch.load(update_model_path, map_location=torch.device('cpu')))
        update_model.eval()

        model = update_model
        make_cf_data_set = update_make_cf_data_set
    finally:
        is_updating = False


@app.post("/update_model")
async def update_model(request: UpdateRequest, background_tasks: BackgroundTasks):
    if is_updating:
        raise HTTPException(status_code=409, detail="Model is currently being updated")

    background_tasks.add_task(update_model_task, request)

    return {"message": "Model update started in the background"}


@app.post("/recommend")
async def get_recommendation(request: RecommendRequest):
    global model, make_cf_data_set
    if is_updating:
        raise HTTPException(status_code=409, detail="Model is currently being updated")
    if model is None or make_cf_data_set is None:
        raise HTTPException(status_code=400, detail="Model not initialized. Please update the model first.")
    try:
        recommended_pois_with_scores = recommend_poi(model, request.travel_id, make_cf_data_set, top_k=10)
        return {"recommendations": [{"poi_id": poi_id, "score": float(score)} for poi_id, score in
                                    recommended_pois_with_scores]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9999)