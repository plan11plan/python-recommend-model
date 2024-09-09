import sys
import os
import logging
from typing import List, Dict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
code_dir = os.path.join(project_root, 'code')
sys.path.insert(0, project_root)
sys.path.insert(0, code_dir)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import torch
from code.gmf import GMF
from code.mlp import MLP
from code.neumf import NeuMF
from code.data import MakeCFDataSet
from code.config import config, device
from code.recommend import recommend_poi, set_input_queue, get_output_queue
import queue

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 입력과 출력을 위한 큐 생성
input_queue = queue.Queue()
output_queue = queue.Queue()

# recommend.py에 큐 전달
set_input_queue(input_queue)
get_output_queue(output_queue)

# 데이터셋 생성 및 모델 초기화
make_cf_data_set = MakeCFDataSet(config=config,data_json = None)
gmf = GMF(num_user=make_cf_data_set.num_user, num_item=make_cf_data_set.num_item, num_factor=config.num_factor).to(
    device)
mlp = MLP(num_user=make_cf_data_set.num_user, num_item=make_cf_data_set.num_item, num_factor=config.num_factor,
          num_layers=config.num_layers, dropout=config.dropout).to(device)
model = NeuMF(GMF=gmf, MLP=mlp, num_factor=config.num_factor).to(device)

model_path = os.path.join(config.model_path, config.nmf_path)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

logger.info("모델이 성공적으로 불러와졌습니다.")


class RecommendRequest(BaseModel):
    travel_id: str


class RecommendItem(BaseModel):
    poi_id: str
    score: float


class RecommendResponse(BaseModel):
    recommendations: List[RecommendItem] = Field(..., description="List of recommendations")


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    try:
        logger.info(f"Received request for travel_id: {request.travel_id}")

        # 입력 큐에 travel_id 추가
        input_queue.put(request.travel_id)

        # recommend_poi 함수 호출
        recommend_poi(model, make_cf_data_set, top_k=10)

        # 결과를 출력 큐에서 가져옴
        recommendations = output_queue.get(timeout=30)  # 30초 타임아웃 설정

        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found for the given travel_id")

        # 결과 변환
        validated_recommendations = [
            RecommendItem(poi_id=str(poi_id), score=float(score))
            for poi_id, score in recommendations
        ]

        logger.info(f"Returning {len(validated_recommendations)} recommendations")
        return RecommendResponse(recommendations=validated_recommendations)
    except queue.Empty:
        logger.error("Request timed out")
        raise HTTPException(status_code=408, detail="Request timed out")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9999)