import os
from PIL import Image
import json
from pathlib import Path

class YOLOBBoxSaver:
    def __init__(self):
        self.class_list = []
        self.class_file_path = "classes.txt"
        self.load_classes()
    
    def load_classes(self):
        """클래스 목록 로드"""
        try:
            if os.path.exists(self.class_file_path):
                with open(self.class_file_path, 'r', encoding='utf-8') as f:
                    self.class_list = [line.strip() for line in f.readlines()]
            else:
                self.class_list = ["class_0"]  # 기본 클래스
                self.save_classes()
        except Exception as e:
            print(f"클래스 로드 중 오류 발생: {e}")
            self.class_list = ["class_0"]
    
    def save_classes(self):
        """클래스 목록 저장"""
        try:
            with open(self.class_file_path, 'w', encoding='utf-8') as f:
                for class_name in self.class_list:
                    f.write(f"{class_name}\n")
        except Exception as e:
            print(f"클래스 저장 중 오류 발생: {e}")
    
    def add_class(self, class_name):
        """새로운 클래스 추가"""
        if class_name not in self.class_list:
            self.class_list.append(class_name)
            self.save_classes()
    
    def get_classes(self):
        """클래스 목록 반환"""
        return self.class_list
    
    def convert_to_yolo_format(self, bbox, image_size):
        """
        bbox 좌표를 YOLO 형식으로 변환
        
        Args:
            bbox (tuple): (class_id, x1, y1, x2, y2)
            image_size (tuple): (width, height)
        
        Returns:
            tuple: (class_id, x_center, y_center, width, height)
        """
        class_id, x1, y1, x2, y2 = bbox
        image_width, image_height = image_size
        
        # 입력값 검증
        if not all(isinstance(x, (int, float)) for x in [x1, y1, x2, y2]):
            raise ValueError("Invalid bbox coordinates")
        
        # 좌표 정규화
        try:
            # 중심점 계산 및 정규화
            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            
            # 값 범위 검증 (0~1)
            for value in [x_center, y_center, width, height]:
                if not 0 <= value <= 1:
                    raise ValueError("Normalized values must be between 0 and 1")
            
            return (class_id, x_center, y_center, width, height)
        except ZeroDivisionError:
            raise ValueError("Invalid image dimensions")

    def save_bbox_data(self, image_path, bboxes, canvas_size=None, image_position=None):
        """
        YOLO 형식으로 bbox 데이터 저장
        
        Args:
            image_path (str): 이미지 파일 경로
            bboxes (list): [(class_id, x1, y1, x2, y2), ...]
            canvas_size (tuple): 캔버스 크기 (width, height)
            image_position (tuple): 이미지 위치 (x, y)
        """
        try:
            # 이미지 크기 가져오기
            with Image.open(image_path) as img:
                image_size = img.size

            # 저장할 파일 경로
            txt_path = Path(image_path).with_suffix('.txt')

            # 캔버스 좌표를 실제 이미지 좌표로 변환
            scale_x = image_size[0] / canvas_size[0] if canvas_size else 1
            scale_y = image_size[1] / canvas_size[1] if canvas_size else 1
            offset_x = image_position[0] if image_position else 0
            offset_y = image_position[1] if image_position else 0

            # YOLO 형식으로 변환하여 저장
            with open(txt_path, 'w', encoding='utf-8') as f:
                for bbox in bboxes:
                    # 캔버스 좌표를 이미지 좌표로 변환
                    class_id = bbox[0]
                    x1 = (bbox[1] - offset_x) * scale_x
                    y1 = (bbox[2] - offset_y) * scale_y
                    x2 = (bbox[3] - offset_x) * scale_x
                    y2 = (bbox[4] - offset_y) * scale_y

                    # 이미지 범위 내로 클리핑
                    x1 = max(0, min(x1, image_size[0]))
                    y1 = max(0, min(y1, image_size[1]))
                    x2 = max(0, min(x2, image_size[0]))
                    y2 = max(0, min(y2, image_size[1]))

                    # YOLO 형식으로 변환 (중심점과 너비, 높이)
                    x_center = (x1 + x2) / 2 / image_size[0]
                    y_center = (y1 + y2) / 2 / image_size[1]
                    width = (x2 - x1) / image_size[0]
                    height = (y2 - y1) / image_size[1]

                    # 좌표값 검증
                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        f.write(line)

            return True
        except Exception as e:
            print(f"데이터 저장 중 오류 발생: {e}")
            return False
    
    
    def save_dataset_config(self, save_dir, data_name="dataset"):
        """
        YOLO 학습을 위한 dataset.yaml 설정 파일 저장
        
        Args:
            save_dir (str): 저장할 디렉토리 경로
            data_name (str): 데이터셋 이름
        """
        try:
            config = {
                'path': save_dir,
                'train': 'train/images',
                'val': 'valid/images',
                'test': 'test/images',
                'names': {i: name for i, name in enumerate(self.class_list)}
            }
            
            yaml_path = os.path.join(save_dir, f"{data_name}.yaml")
            
            # YAML 형식으로 저장
            with open(yaml_path, 'w', encoding='utf-8') as f:
                # YAML 형식으로 직접 작성
                f.write(f"path: {config['path']}\n")
                f.write(f"train: {config['train']}\n")
                f.write(f"val: {config['val']}\n")
                f.write(f"test: {config['test']}\n")
                f.write("names:\n")
                for idx, name in config['names'].items():
                    f.write(f"  {idx}: {name}\n")
            
            return True
        except Exception as e:
            print(f"설정 파일 저장 중 오류 발생: {e}")
            return False