"""얼굴 검출/크롭 유틸리티 (YuNet).

표정 임베딩은 얼굴이 또렷한 크롭을 입력으로 받을 때 가장 잘 작동한다.
ML 스코어러(acting-score)와 **같은 검출기**(OpenCV YuNet)를 써서 크롭 파이프라인을
맞춘다 — 같은 얼굴 crop → 같은 HSEmotion 입력 → 두 시스템이 같은 표정 공간을 공유.
프레임마다 가장 큰 얼굴 하나를 잘라내고, 얼굴이 없으면 호출부에서 중앙
크롭(`center_crop_square`)으로 폴백한다.
"""

from pathlib import Path

import cv2
import numpy as np

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"


def _default_yunet_path() -> str:
    """YuNet ONNX 파일 경로를 찾는다 (레포 루트/models 우선, 없으면 cwd/models)."""
    candidates = [
        Path(__file__).resolve().parents[2] / "models" / YUNET_FILENAME,
        Path.cwd() / "models" / YUNET_FILENAME,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])  # 없으면 첫 후보 경로로 명확한 에러를 내게 둔다


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
    """프레임에서 가장 큰 얼굴을 검출해 정사각 크롭으로 반환한다 (YuNet).

    스코어러의 ``analyze_frames``와 동일하게 YuNet으로 검출하고 가장 큰 얼굴을
    고른다. 검출은 BGR 프레임에서 하고, 크롭은 원본 RGB에서 떠서 반환한다
    (HSEmotion이 RGB 입력을 기대).
    """

    def __init__(
        self,
        device: str | None = None,
        image_size: int = 224,
        model_path: str | None = None,
        score_threshold: float = 0.6,
        min_face: int = 20,
    ) -> None:
        """검출기를 초기화한다.

        Args:
            device: 하위호환용 인자. YuNet은 CPU로 실행되어 무시된다.
            image_size: 얼굴 크롭 출력 한 변 길이.
            model_path: YuNet ONNX 경로. None이면 repo `models/`에서 찾는다.
            score_threshold: YuNet 검출 신뢰도 임계값.
            min_face: 이보다 작은 얼굴 박스는 버린다 (px).
        """
        self.device = "cpu"
        self.image_size = image_size
        self.min_face = min_face
        self.detector = cv2.FaceDetectorYN.create(
            model_path or _default_yunet_path(),
            "",
            (320, 320),
            score_threshold=score_threshold,
        )

    def crop_largest(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """RGB 프레임에서 가장 큰 얼굴 크롭을 반환한다.

        Args:
            frame_rgb: (H, W, 3) RGB uint8 프레임.

        Returns:
            (image_size, image_size, 3) RGB uint8 얼굴 크롭. 얼굴이 없거나 너무
            작으면 None.
        """
        h, w = frame_rgb.shape[:2]
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])[:4]
        x, y, fw, fh = max(0, int(x)), max(0, int(y)), int(fw), int(fh)
        crop = frame_rgb[y : y + fh, x : x + fw]
        if crop.shape[0] < self.min_face or crop.shape[1] < self.min_face:
            return None
        return cv2.resize(
            crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        ).astype(np.uint8)
