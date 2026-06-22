"""V-JEPA 2 토큰 레이아웃 실측 (구간 매칭 설계 검증용, 일회성).

질문: get_vision_features 출력 토큰이 시간축으로 분해 가능한가?
가능하면 추론 1회로 구간 임베딩을 뽑을 수 있다.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

from video_feedback.video_utils import load_frames

MODEL = "facebook/vjepa2-vitl-fpc64-256"
SAMPLE = "연기영상/clips/C-pro__1UTp6xH_4kc.mp4"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}, 모델 로딩...")
processor = AutoVideoProcessor.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device).eval()

frames = load_frames(SAMPLE, num_frames=64)
print(f"frames: {frames.shape}")  # (64, H, W, 3)
video = torch.from_numpy(frames).permute(0, 3, 1, 2)  # T,C,H,W
inputs = processor(video, return_tensors="pt").to(device)

print("=== processor inputs ===")
for k, v in inputs.items():
    print(f"  {k}: {getattr(v, 'shape', type(v))}")

with torch.no_grad():
    feats = model.get_vision_features(**inputs)

print(f"=== get_vision_features ===")
print(f"  output shape: {feats.shape}")  # (1, num_tokens, D)
n_tokens = feats.shape[1]
print(f"  num_tokens={n_tokens}, D={feats.shape[2]}")

# 토큰 수 인수분해 후보 (시간 그룹 추정)
print("=== 토큰 수 인수분해 (시간×공간 후보) ===")
for t in [8, 16, 32, 64]:
    if n_tokens % t == 0:
        spatial = n_tokens // t
        side = int(spatial**0.5)
        ok = "✓ 정사각" if side * side == spatial else ""
        print(f"  time={t:>2} → spatial={spatial} (side≈{side}) {ok}")
