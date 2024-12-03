from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.security import APIKeyHeader
from app.core.config import Settings
from app.models.model import RAPIDModel
from app.core.utils import save_upload_file_temp
import torch
import logging

router = APIRouter()
settings = Settings()
api_key_header = APIKeyHeader(name="X-API-Key")

# 모델 초기화
model = RAPIDModel(settings.config)
model.load_state_dict(torch.load(settings.MODEL_PATH, map_location='cuda')['state_dict'], strict=False)
model = model.to(settings.config['device'])
model.eval()

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@router.post("/predict")
async def predict_video(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    try:
        # 임시 파일로 저장
        temp_path = save_upload_file_temp(file)
        
        # 예측 수행
        with torch.no_grad():
            logits, pred, highlight_timestamp, all_tokens = model.predict(str(temp_path))
        
        # 임시 파일 삭제
        temp_path.unlink()
        
        return {
            "prediction": int(pred),
            "logits": logits.cpu().numpy().tolist(),
            "pred": pred,
            "highlight_timestamp": highlight_timestamp,
            "all_tokens": all_tokens.to_dict(orient='list')
        }
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))