import torch
import cv2
from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.5, min_height=10):
        """
        YOLO 검출기 초기화
        
        Args:
            model_path (str): YOLO 모델 경로
            conf_threshold (float): 신뢰도 임계값 (기본값: 0.5)
            min_height (int): 최소 높이 임계값 (기본값: 10)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.min_height = min_height
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
    def _load_model(self):
        """YOLO 모델 로드"""
        try:
            return YOLO(self.model_path)
        except Exception as e:
            raise Exception(f"모델 로드 실패: {e}")
    
    def _preprocess_image(self, image):
        """
        이미지 전처리
        
        Args:
            image (numpy.ndarray): 입력 이미지
        
        Returns:
            numpy.ndarray: 전처리된 이미지
        """
        # 이미지가 2D(그레이스케일)인 경우에만 3채널로 변환
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = image.copy()

        # dtype 변환 (필요한 경우)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        return image
    
    def _sort_detections(self, boxes):
        """
        검출된 박스를 y좌표 기준으로 정렬
        
        Args:
            boxes: YOLO 검출 결과의 박스들
            
        Returns:
            list: y좌표로 정렬된 박스 목록
        """
        sorted_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            sorted_boxes.append((y1, box))
        
        # y좌표 기준으로 정렬
        sorted_boxes.sort(key=lambda x: x[0])
        return sorted_boxes
    
    def detect(self, image):
        """
        이미지에서 객체 검출 수행
        
        Args:
            image (numpy.ndarray): 입력 이미지
            
        Returns:
            list: 검출된 y좌표 범위 리스트 [(y1,y2), ...]
        """
        try:
            # 이미지 전처리
            processed_image = self._preprocess_image(image)
            
            # 모델 추론
            results = self.model(processed_image)
            
            # 박스 정보 추출
            boxes = results[0].boxes
            
            # 박스 정렬
            sorted_boxes = self._sort_detections(boxes)
            
            # y좌표 추출
            det_cut_y_arr = []
            for _, box in sorted_boxes:
                # 바운딩 박스 좌표, 신뢰도 추출
                xyxy = box.xyxy.cpu().detach().numpy().tolist()[0]
                conf = box.conf.cpu().detach().numpy().tolist()[0]
                cls_idx = int(box.cls.cpu().detach().numpy().tolist()[0])  # 클래스 인덱스
                                
                # 신뢰도 임계값 확인
                if conf < self.conf_threshold:
                    continue
                
                # 좌표를 정수형으로 변환
                x1, y1, x2, y2 = map(int, xyxy)
                
                det_cut_y_arr.append((x1, y1, x2, y2, cls_idx))
            
            return det_cut_y_arr
            
        except Exception as e:
            raise Exception(f"검출 중 오류 발생: {e}")

