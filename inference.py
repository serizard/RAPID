import torch
import argparse
import numpy as np
from main import Model 

def load_model(checkpoint_path, gpu_id=0):
    """체크포인트에서 모델을 로드합니다."""
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # 체크포인트에서 하이퍼파라미터 가져오기
    hparams = checkpoint['hyper_parameters']
    
    # 모델 초기화 및 가중치 로드
    model = Model.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda(gpu_id)
    
    return model

def inference(text, audio_feature, video_feature, model):
    """
    입력 데이터에 대한 추론을 수행합니다.
    
    Args:
        text (str): 입력 텍스트
        audio_feature (numpy.ndarray): 오디오 특징 (chunk_size x feature_dim)
        video_feature (numpy.ndarray): 비디오 특징 (chunk_size x feature_dim)
        model (Model): 로드된 모델
    
    Returns:
        predicted_class (int): 예측된 클래스
        probabilities (numpy.ndarray): 각 클래스에 대한 확률
    """
    model.eval()
    
    # 텍스트 전처리
    encoded_text = model.tokenizer.encode(
        str(text),
        padding='max_length',
        max_length=model.args.max_length,
        truncation=True,
        return_tensors='pt'
    )
    
    # 특징 전처리
    audio_feature = torch.FloatTensor(audio_feature).unsqueeze(0)
    video_feature = torch.FloatTensor(video_feature).unsqueeze(0)
    
    # GPU 사용 가능시 데이터를 GPU로 이동
    if torch.cuda.is_available():
        encoded_text = encoded_text.cuda()
        audio_feature = audio_feature.cuda()
        video_feature = video_feature.cuda()
    
    with torch.no_grad():
        # 더미 라벨 생성 (forward 함수에 필요)
        dummy_labels = torch.zeros(1, dtype=torch.long)
        if torch.cuda.is_available():
            dummy_labels = dummy_labels.cuda()
            
        # 더미 인덱스 생성
        dummy_idx = torch.zeros(1, dtype=torch.long)
        if torch.cuda.is_available():
            dummy_idx = dummy_idx.cuda()
            
        # 추론
        logits, _, _, _ = model(dummy_labels, encoded_text, audio_feature, video_feature, dummy_idx)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1)
    
    return predicted_class.item(), probabilities.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                      help="Path to the model checkpoint file")
    parser.add_argument("--gpu", type=int, default=0,
                      help="GPU device ID to use")
    args = parser.parse_args()
    
    # 모델 로드
    model = load_model(args.checkpoint_path, args.gpu)
    
    # 예시 데이터 (실제 사용시 적절한 크기로 수정 필요)
    text = "sample text"
    audio_feature = np.random.randn(50, 768)  
    video_feature = np.random.randn(50, 768)  
    
    # 추론 실행
    predicted_class, probabilities = inference(text, audio_feature, video_feature, model)
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities}")