from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
import os
import json
from config import device, config
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
gmf = GMF(num_user=make_cf_data_set.num_user, num_item=make_cf_data_set.num_item, num_factor=config.num_factor).to(device)
mlp = MLP(num_user=make_cf_data_set.num_user, num_item=make_cf_data_set.num_item, num_factor=config.num_factor,
          num_layers=config.num_layers, dropout=config.dropout).to(device)
model = NeuMF(GMF=gmf, MLP=mlp, num_factor=config.num_factor).to(device)

# Load model weights
model_path = os.path.join(config.model_path, "NMF3.pt")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

is_updating = False

class RecommendRequest(BaseModel):
    travel_id: str

@app.post("/update_model")
async def update_model(request: dict, background_tasks: BackgroundTasks):
    print("모델 업데이트 요청이 왔습니다.")
    #if is_updating:
        #raise HTTPException(status_code=409, detail="Model is currently being updated")
    # print(f"모델 업데이트를 시작합니다. 요청 데이터 수: {len(request.get('attractions', []))}")
    background_tasks.add_task(update_model_task, request)
    return {"message": "Model update started in the background"}


import json
from collections import defaultdict


def update_model_task(request: dict):
    global model, make_cf_data_set, is_updating, train_num
    is_updating = True
    try:
        train_num = int(train_num) + 1
        print(f"모델 업데이트 작업 시작. train_num: {train_num}")

        # Request 객체에서 데이터 추출 및 처리
        data_list = []
        for key, value in request.items():
            if isinstance(value, list):
                # 이미 리스트인 경우 직접 추가
                data_list.extend(value)
            elif isinstance(value, str):
                try:
                    # 문자열로 된 리스트를 파싱
                    items = json.loads(value.replace("'", "\""))
                    if isinstance(items, list):
                        data_list.extend(items)
                    else:
                        data_list.append(items)
                except json.JSONDecodeError:
                    print(f"Warning: 잘못된 JSON 형식 - {value}")
                    continue
            else:
                print(f"Warning: 예상치 못한 데이터 형식 - {type(value)}")

        # MakeCFDataSet이 기대하는 형식으로 변환
        data_json = {str(i): item for i, item in enumerate(data_list) if isinstance(item, dict)}

        print("요청객체 항목 수:", len(data_list))
        print(f"데이터 처리 완료. 처리된 데이터 수: {len(data_json)}")

        if len(data_json) > 0:
            print("data_json의 첫 번째 항목:", next(iter(data_json.items())))
        else:
            raise ValueError("처리된 데이터가 없습니다.")

        # POI_ID 존재 여부 확인
        poi_ids = [item.get('POI_ID') for item in data_json.values() if 'POI_ID' in item]
        if not poi_ids:
            raise ValueError("데이터에 'POI_ID'가 없습니다.")

        update_make_cf_data_set = run_recommendation_models(config, device, data_json=data_json, train_num=train_num)

        # 모델 업데이트 로직
        update_gmf = GMF(num_user=update_make_cf_data_set.num_user, num_item=update_make_cf_data_set.num_item,
                         num_factor=config.num_factor).to(device)
        update_mlp = MLP(num_user=update_make_cf_data_set.num_user, num_item=update_make_cf_data_set.num_item,
                         num_factor=config.num_factor, num_layers=config.num_layers, dropout=config.dropout).to(device)
        update_model = NeuMF(GMF=update_gmf, MLP=update_mlp, num_factor=config.num_factor).to(device)
        update_model_path = os.path.join(config.model_path, "NMF"+str(train_num)+".pt")
        update_model.load_state_dict(torch.load(update_model_path, map_location=torch.device('cpu')))
        update_model.eval()
        model = update_model
        make_cf_data_set = update_make_cf_data_set

        print("모델 업데이트 완료")
    except Exception as e:
        print(f"모델 업데이트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        is_updating = False
@app.post("/recommend")
async def get_recommendation(request: RecommendRequest):
    global model, make_cf_data_set
    # if is_updating:
    #     raise HTTPException(status_code=409, detail="Model is currently being updated")
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