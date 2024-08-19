import torch
from config import device
import os
from gmf import GMF
from mlp import MLP
from neumf import NeuMF
from data import MakeCFDataSet
from config import config

def recommend_poi(model, travel_id, make_cf_data_set, top_k=10):
    """
    주어진 TRAVEL_ID에 대해 POI_ID와 점수를 추천하는 메소드.

    :param model: 학습된 추천 모델
    :param travel_id: 추천을 원하는 여행 ID
    :param make_cf_data_set: 데이터셋 객체
    :param top_k: 추천할 POI의 수
    :return: 추천된 POI_ID와 점수의 리스트
    """
    # 여행 ID에 해당하는 사용자 인덱스를 가져옵니다.
    user_idx = make_cf_data_set.user_encoder.get(travel_id)  # 원본 TRAVEL_ID로부터 인덱스 가져오기
    if user_idx is None:
        raise ValueError("유효하지 않은 여행 ID입니다.")

    # 모든 POI에 대해 추천 점수를 계산합니다.
    all_items = make_cf_data_set.exist_items  # 모든 POI 목록
    users = torch.tensor([user_idx] * len(all_items)).to(device)
    items = torch.tensor(all_items).to(device)

    with torch.no_grad():
        output = model(users, items)  # 모델에 사용자와 POI를 입력하여 점수를 계산합니다.

    # output이 비어있는지 확인
    if len(output) == 0:
        raise ValueError("모델의 출력이 비어 있습니다.")

    # top_k가 0보다 큰지 확인
    if top_k <= 0:
        raise ValueError("top_k는 0보다 커야 합니다.")

    # 점수를 기반으로 상위 K개의 POI를 추천합니다.
    top_k_indices = output.argsort(descending=True)[:top_k]  # 상위 K개의 인덱스

    recommended_poi_ids = [all_items[i] for i in top_k_indices.cpu().numpy()]  # POI ID로 변환
    recommended_scores = output[top_k_indices].cpu().numpy()  # 추천 점수 추출

    # POI ID와 점수를 함께 반환합니다.
    recommendations = list(zip(recommended_poi_ids, recommended_scores))

    # POI ID를 원본 값으로 변환
    original_poi_ids = [make_cf_data_set.item_decoder[poi_id] for poi_id, _ in recommendations]

    # 원본 POI ID와 점수를 함께 반환
    recommendations_with_original_ids = list(zip(original_poi_ids, recommended_scores))

    return recommendations_with_original_ids

# 데이터셋 생성
make_cf_data_set = MakeCFDataSet(config=config)

# 모델 재구성 (훈련 시 사용한 동일한 파라미터로)
gmf = GMF(
    num_user=make_cf_data_set.num_user,
    num_item=make_cf_data_set.num_item,
    num_factor=config.num_factor).to(device)

mlp = MLP(
    num_user=make_cf_data_set.num_user,
    num_item=make_cf_data_set.num_item,
    num_factor=config.num_factor,
    num_layers=config.num_layers,
    dropout=config.dropout).to(device)

model = NeuMF(
    GMF=gmf,
    MLP=mlp,
    num_factor=config.num_factor).to(device)

# 저장된 모델 파일 경로
model_path = os.path.join(config.model_path, config.nmf_path)

# 모델 가중치 불러오기
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))

# 모델이 제대로 로드되었는지 확인
print("모델이 성공적으로 불러와졌습니다.")

model.eval()

# 무한 반복문으로 유저 ID 입력 받기
while True:
    travel_id = input("추천을 원하는 여행 ID를 입력하세요 (종료하려면 'exit' 입력): ")
    if travel_id.lower() == 'exit':
        print("프로그램을 종료합니다.")
        break
    
    try:
        recommended_pois_with_scores = recommend_poi(model, travel_id, make_cf_data_set, top_k=10)
        for poi_id, score in recommended_pois_with_scores:
            print(f'POI ID: {poi_id}, Score: {score}')
    except ValueError as e:
        print(e)
