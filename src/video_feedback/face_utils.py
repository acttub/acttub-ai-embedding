"""얼굴 검출/크롭 유틸리티 (MTCNN).

표정 임베딩은 얼굴이 또렷한 크롭을 입력으로 받을 때 가장 잘 작동한다.
프레임마다 가장 큰 얼굴 하나를 잘라내고, 얼굴이 없으면 호출부에서
중앙 크롭(`center_crop_square`)으로 폴백한다.
"""

import cv2
import numpy as np


def center_crop_square(frame_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """프레임 중앙을 정사각으로 잘라 size×size로 리사이즈한다 (얼굴 검출 폴백).

    Args:
        frame_rgb: (H, W, 3) RGB uint8 프레임.
        size: 출력 한 변 길이.

    Returns:
        (size, size, 3) RGB uint8 배열.
    """
    h, w = frame_rgb.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    crop = frame_rgb[top : top + side, left : left + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.uint8)


class FaceDetector:
    """프레임에서 가장 큰 얼굴을 검출해 정사각 크롭으로 반환한다 (MTCNN)."""

    def __init__(
        self, device: str | None = None, image_size: int = 224, margin: int = 20
    ) -> None:
        """검출기를 초기화한다.

        Args:
            device: 추론 장치. None이면 가능 시 cuda, 아니면 cpu.
            image_size: 얼굴 크롭 출력 한 변 길이 (ViT 입력 224).
            margin: 얼굴 박스 바깥 여백 픽셀.
        """
        import torch
        from facenet_pytorch import MTCNN

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        # keep_all=False + select_largest=True → 가장 큰 얼굴 1개.
        # post_process=False → 0~255 범위 유지(정규화 안 함, ViT 프로세서가 따로 정규화).
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=False,
            select_largest=True,
            post_process=False,
            device=self.device,
        )

    def crop_largest(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """RGB 프레임에서 가장 큰 얼굴 크롭을 반환한다.

        Args:
            frame_rgb: (H, W, 3) RGB uint8 프레임.

        Returns:
            (image_size, image_size, 3) RGB uint8 얼굴 크롭. 얼굴이 없으면 None.
        """
        from PIL import Image

        img = Image.fromarray(frame_rgb.astype(np.uint8))
        face = self.mtcnn(img)  # (3, S, S) float(0~255) 또는 None
        if face is None:
            return None
        arr = face.permute(1, 2, 0).cpu().numpy()  # (S, S, 3)
        return np.clip(arr, 0, 255).astype(np.uint8)
