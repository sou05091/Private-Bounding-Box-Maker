import tkinter as tk
import os
import cv2
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from save_txt import YOLOBBoxSaver
from model_predict import YOLODetector
from tkinter import simpledialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD

class ImageViewer:
    def __init__(self, root):
        # self.root = root
        self.root = root
        self.root.title("BBOX Label Maker (MB)")

        # directory 설정
        self.file_paths = None
        self.setup_drag_drop()
        
        # 초기 윈도우 크기 설정
        self.root.geometry("1000x600")
        
        # 메인 프레임 생성
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(expand=True, fill='both')
        
        # 상단 메뉴 프레임
        self.menu_frame = ttk.Frame(self.main_frame)
        self.menu_frame.pack(fill='x', padx=5, pady=5)
        
        # 모드 설정
        self.mode = "view"  # 'view', 'draw', 'select'
        
        # 버튼 프레임
        self.button_frame = ttk.Frame(self.menu_frame)
        self.button_frame.pack(side='left')
        
        # 전체화면/창모드 토글 버튼
        self.fullscreen = False
        self.toggle_btn = ttk.Button(self.button_frame, text="전체화면", command=self.toggle_fullscreen)
        self.toggle_btn.pack(side='left', padx=2)
        
        # bbox 모드 버튼들
        self.mode_var = tk.StringVar(value="view")
        ttk.Radiobutton(self.button_frame, text="보기", variable=self.mode_var, 
                       value="view", command=self.change_mode).pack(side='left', padx=2)
        ttk.Radiobutton(self.button_frame, text="그리기", variable=self.mode_var, 
                       value="draw", command=self.change_mode).pack(side='left', padx=2)
        ttk.Radiobutton(self.button_frame, text="선택", variable=self.mode_var, 
                       value="select", command=self.change_mode).pack(side='left', padx=2)
        
        # bbox 삭제 버튼
        self.clear_btn = ttk.Button(self.button_frame, text="bbox 전체 삭제", 
                                  command=self.clear_bboxes)
        self.clear_btn.pack(side='left', padx=2)
        
        # 선택된 bbox 삭제 버튼
        self.delete_selected_btn = ttk.Button(self.button_frame, text="선택 bbox 삭제",
                                            command=self.delete_selected_bbox)
        self.delete_selected_btn.pack(side='left', padx=2)
        
        # Canvas 생성
        self.canvas = tk.Canvas(self.main_frame, highlightthickness=0)
        self.canvas.pack(expand=True, fill='both')
        
        # bbox 관련 변수
        self.start_x = None
        self.start_y = None
        self.current_bbox = None
        self.image_bboxes = {}  # 이미지 경로를 키로 사용하여 bbox 저장
        self.selected_bbox = None
        self.selected_bbox_id = None
        self.drag_start_x = None
        self.drag_start_y = None

        # YOLO 학습용 BBOX
        self.yolo_bboxes = {}
        
        # 이미지 관련 변수
        self.images = []
        self.current_index = 0
        self.current_image_tk = None
        self.image_position = (0, 0)  # 이미지의 좌상단 위치
        
        # 키 이벤트 바인딩
        # self.root.bind('<n>', self.next_image)
        self.root.bind('<n>', self.next_image_and_save)
        self.root.bind('<p>', self.previous_image)
        self.root.bind('<Escape>', self.exit_fullscreen)
        self.root.bind('<f>', self.toggle_fullscreen)
        self.root.bind('<Delete>', self.delete_selected_bbox)
        self.root.bind('<s>', self.select_mode)  
        self.root.bind('<d>', self.draw_mode)  
        self.root.bind('<a>', self.auto_detect)  

        # 마우스 이벤트 바인딩
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        
        # 시작 버튼
        self.start_btn = ttk.Button(self.menu_frame, text="이미지 선택", 
                                  command=self.load_images)
        self.start_btn.pack(side='left', padx=5)
    
        # 이미지 확대/축소 관련 변수 추가
        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0

        # 마우스 휠 이벤트 바인딩
        self.canvas.bind('<MouseWheel>', self.on_mousewheel)  # Windows
        self.canvas.bind('<Button-4>', self.on_mousewheel)    # Linux 위로 스크롤
        self.canvas.bind('<Button-5>', self.on_mousewheel)    # Linux 아래로 스크롤

        # 이미지 카운터 레이블
        self.counter_label = ttk.Label(self.menu_frame, text="0/0")
        self.counter_label.pack(side='right', padx=5)
    
        # YOLO 저장 관리자 초기화
        self.bbox_saver = YOLOBBoxSaver()
        
        # 클래스 선택을 위한 콤보박스 생성
        self.class_var = tk.StringVar()
        self.class_label = ttk.Label(self.button_frame, text="Class:")
        self.class_label.pack(side='left', padx=2)
        self.class_combo = ttk.Combobox(self.button_frame, textvariable=self.class_var, width=5)
        self.class_combo.pack(side='left', padx=2)
        
        # 콤보박스 값 설정
        self.class_combo['values'] = self.bbox_saver.get_classes()
        self.class_combo.set("0")  # 기본값 설정
        
        # 새 클래스 추가 버튼
        self.add_class_btn = ttk.Button(self.button_frame, text="새 클래스 추가", 
                                    command=self.add_new_class)
        self.add_class_btn.pack(side='left', padx=2)

        # 모델 자동 예측
        model_path = r"C:\Users\mbkim\Desktop\Claripi\ClariVBA\DL_VBA_Core\onnx\crop_spine.pt"  # 모델 경로
        # model_path = r'C:\Users\mbkim\Desktop\RibFrac_yolov\runs\detect\train\weights\best.pt'
        self.detector = YOLODetector(
            model_path=model_path,
            conf_threshold=0.5,
            min_height=10
        )

        # 패닝 관련 변수
        self.pan_start_x = None
        self.pan_start_y = None
        self.pan_start_image_pos = None

        # 중간 마우스 버튼 바인딩
        self.canvas.bind('<Button-2>', self.start_pan)
        self.canvas.bind('<B2-Motion>', self.pan)
        self.canvas.bind('<ButtonRelease-2>', self.end_pan)

        # 자동 검출 버튼
        self.detect_btn = ttk.Button(self.button_frame, text="자동 검출", 
                                    command=self.auto_detect)
        self.detect_btn.pack(side='left', padx=2)

        # 저장 버튼 (class_combo 생성 이후에 추가)
        self.save_btn = ttk.Button(self.button_frame, text="저장", 
                                command=self.save_current_image_bboxes)
        self.save_btn.pack(side='left', padx=2)

        # 클래스별 색상 설정
        self.class_colors = {
            "0": "#FF0000",  # 빨강
            "1": "#00FF00",  # 초록
            "2": "#0000FF",  # 파랑
            "3": "#FF00FF",  # 마젠타
            "4": "#00FFFF",  # 시안
            "5": "#FFFF00",  # 노랑
            "6": "#FF8000",  # 주황
            "7": "#8000FF",  # 보라
            "8": "#0080FF",  # 하늘색
            "9": "#FF0080",  # 분홍
        }


    def change_mode(self, event=None):
        self.mode = self.mode_var.get()
        if self.mode == "select":
            self.canvas.config(cursor="hand2")
        elif self.mode == "draw":
            self.canvas.config(cursor="crosshair")
        else:
            self.canvas.config(cursor="")
        self.deselect_bbox()

    def deselect_bbox(self):
        if self.selected_bbox_id:
            # 선택된 bbox의 빨간색으로 변경
            self.canvas.itemconfig(self.selected_bbox_id, outline='red', width=1)
            # 조절점 삭제
            self.delete_resize_handles()
            self.selected_bbox = None
            self.selected_bbox_id = None

    def create_resize_handles(self, bbox_coords):
        """bbox의 꼭짓점에 조절점 생성"""
        x1, y1, x2, y2 = bbox_coords
        handle_size = 6
        self.resize_handles = []
        
        # 꼭짓점 위치 계산
        corners = [
            (x1, y1, 'nw'),  # 좌상단
            (x2, y1, 'ne'),  # 우상단
            (x1, y2, 'sw'),  # 좌하단
            (x2, y2, 'se')   # 우하단
        ]
        
        for x, y, pos in corners:
            handle = self.canvas.create_rectangle(
                x - handle_size/2, y - handle_size/2,
                x + handle_size/2, y + handle_size/2,
                fill='white', outline='blue',
                tags=f'handle_{pos}'
            )
            self.resize_handles.append(handle)

    def delete_resize_handles(self):
        """조절점 삭제"""
        if hasattr(self, 'resize_handles'):
            for handle in self.resize_handles:
                self.canvas.delete(handle)
            self.resize_handles = []

    def delete_selected_bbox(self, event=None):
        if self.selected_bbox and self.selected_bbox_id:
            current_image = self.images[self.current_index]
            if current_image in self.image_bboxes:
                # 현재 선택된 bbox의 캔버스 좌표
                coords = self.canvas.coords(self.selected_bbox_id)

                # 이미지 크기와 스케일 정보
                image = Image.open(current_image)
                orig_width, orig_height = image.size

                if self.fullscreen:
                    screen_width = self.root.winfo_screenwidth() 
                    screen_height = self.root.winfo_screenheight()
                else:
                    screen_width = self.root.winfo_width()
                    screen_height = self.root.winfo_height() - self.menu_frame.winfo_height()

                scale_x = screen_width / orig_width
                scale_y = screen_height / orig_height
                scale_factor = min(scale_x, scale_y) * self.scale
                x_offset = self.image_position[0]
                y_offset = self.image_position[1]

                # 캔버스 좌표를 원본 이미지 좌표로 변환
                x1 = (coords[0] - x_offset) / scale_factor
                y1 = (coords[1] - y_offset) / scale_factor 
                x2 = (coords[2] - x_offset) / scale_factor
                y2 = (coords[3] - y_offset) / scale_factor

                # 저장된 bbox 목록에서 가장 가까운 것을 찾아 삭제
                closest_index = None
                min_diff = float('inf')

                for i, bbox in enumerate(self.image_bboxes[current_image]):
                    # class_id를 제외한 좌표 비교
                    diff = abs(bbox[1] - x1) + abs(bbox[2] - y1) + \
                            abs(bbox[3] - x2) + abs(bbox[4] - y2)
                    if diff < min_diff:
                        min_diff = diff
                        closest_index = i

                # 가장 가까운 bbox 삭제
                if closest_index is not None:
                    del self.image_bboxes[current_image][closest_index]
                    # yolo_bboxes도 함께 삭제
                    if current_image in self.yolo_bboxes:
                        del self.yolo_bboxes[current_image][closest_index]

                # 캔버스에서 bbox 삭제
                self.canvas.delete(self.selected_bbox_id)
                self.delete_resize_handles()
                self.selected_bbox = None
                self.selected_bbox_id = None

    def find_closest_bbox(self, x, y):
        """가장 가까운 bbox와 그 index를 찾아 반환"""
        if not self.images or self.current_index >= len(self.images):
            return None, None, None

        current_image = self.images[self.current_index]
        if current_image not in self.image_bboxes:
            return None, None, None

        # 현재 스케일 및 오프셋 계산
        image = Image.open(current_image)
        orig_width, orig_height = image.size
        
        if self.fullscreen:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
        else:
            screen_width = self.root.winfo_width()
            screen_height = self.root.winfo_height() - self.menu_frame.winfo_height()
        
        scale_x = screen_width / orig_width
        scale_y = screen_height / orig_height
        base_scale = min(scale_x, scale_y)
        scale_factor = base_scale * self.scale
        x_offset = self.image_position[0]
        y_offset = self.image_position[1]

        # 클릭한 위치의 캔버스 좌표를 원본 이미지 좌표로 변환
        clicked_x = (x - x_offset) / scale_factor
        clicked_y = (y - y_offset) / scale_factor

        closest_bbox = None
        closest_bbox_id = None
        closest_index = None

        # 저장된 bbox들과 비교
        for i, bbox_data in enumerate(self.image_bboxes[current_image]):
            _, x1, y1, x2, y2 = bbox_data
            
            # bbox의 캔버스 좌표 계산
            canvas_x1 = x_offset + (x1 * scale_factor)
            canvas_y1 = y_offset + (y1 * scale_factor)
            canvas_x2 = x_offset + (x2 * scale_factor)
            canvas_y2 = y_offset + (y2 * scale_factor)

            # 클릭 위치가 bbox 내부에 있는지 확인
            if (canvas_x1 <= x <= canvas_x2 and canvas_y1 <= y <= canvas_y2):
                # 캔버스의 해당 bbox 찾기
                for bbox_id in self.canvas.find_all():
                    if self.canvas.type(bbox_id) == "rectangle":
                        coords = self.canvas.coords(bbox_id)
                        if coords and abs(coords[0] - canvas_x1) < 1 and \
                                abs(coords[1] - canvas_y1) < 1 and \
                                abs(coords[2] - canvas_x2) < 1 and \
                                abs(coords[3] - canvas_y2) < 1:
                            return coords, bbox_id, i

        return None, None, None

    def clear_bboxes(self):
        if self.images and self.current_index < len(self.images):
            current_image = self.images[self.current_index]
            if current_image in self.image_bboxes:
                self.image_bboxes[current_image] = []
            self.show_current_image()  # 이미지 다시 그리기
    
    def on_mouse_down(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        
        if self.mode == "draw":
            if self.current_bbox:
                self.canvas.delete(self.current_bbox)

            class_name = self.class_var.get()
            class_id = self.class_name_to_idx.get(class_name, 0)  # 없으면 0을 기본값으로 사용
            color = self.get_bbox_color(class_id)       

            self.current_bbox = self.canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y,
                outline=color, width=1
            )

        elif self.mode == "select":
            # 조절점 클릭 확인
            clicked_handle = self.canvas.find_closest(self.start_x, self.start_y)
            if hasattr(self, 'resize_handles') and clicked_handle[0] in self.resize_handles:
                self.resizing = True
                self.resize_handle = clicked_handle[0]
                tags = self.canvas.gettags(clicked_handle[0])
                self.resize_corner = [tag for tag in tags if tag.startswith('handle_')][0]
                return

            # bbox 선택
            bbox, bbox_id, self.bbox_index = self.find_closest_bbox(self.start_x, self.start_y)
            if bbox:
                if self.selected_bbox_id:
                    self.deselect_bbox()
                self.selected_bbox = bbox
                self.selected_bbox_id = bbox_id
                self.canvas.itemconfig(bbox_id, outline='yellow', width=2)
                self.create_resize_handles(bbox)
                self.drag_start_x = self.start_x
                self.drag_start_y = self.start_y
            else:
                self.deselect_bbox()
                self.show_current_image()
    
    def on_mouse_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        
        if self.mode == "draw" and self.current_bbox:
            self.canvas.coords(self.current_bbox, 
                            self.start_x, self.start_y, cur_x, cur_y)
        elif self.mode == "select":
            if hasattr(self, 'resizing') and self.resizing:
                # 조절점 드래그로 크기 조절
                bbox_coords = list(self.canvas.coords(self.selected_bbox_id))
                
                if 'nw' in self.resize_corner:
                    bbox_coords[0] = cur_x
                    bbox_coords[1] = cur_y
                elif 'ne' in self.resize_corner:
                    bbox_coords[2] = cur_x
                    bbox_coords[1] = cur_y
                elif 'sw' in self.resize_corner:
                    bbox_coords[0] = cur_x
                    bbox_coords[3] = cur_y
                elif 'se' in self.resize_corner:
                    bbox_coords[2] = cur_x
                    bbox_coords[3] = cur_y
                
                # bbox 업데이트
                self.canvas.coords(self.selected_bbox_id, *bbox_coords)

                # 조절점 위치 업데이트
                self.delete_resize_handles()
                self.create_resize_handles(bbox_coords)
                
                # 이미지 크기와 스케일 정보
                image = Image.open(self.images[self.current_index])
                orig_width, orig_height = image.size
                
                if self.fullscreen:
                    screen_width = self.root.winfo_screenwidth()
                    screen_height = self.root.winfo_screenheight()
                else:
                    screen_width = self.root.winfo_width()
                    screen_height = self.root.winfo_height() - self.menu_frame.winfo_height()
                
                scale_x = screen_width / orig_width
                scale_y = screen_height / orig_height
                scale_factor = min(scale_x, scale_y) * self.scale
                x_offset = self.image_position[0]
                y_offset = self.image_position[1]

                # 캔버스 좌표를 원본 이미지 좌표로 변환
                x1 = (bbox_coords[0] - x_offset) / scale_factor
                y1 = (bbox_coords[1] - y_offset) / scale_factor
                x2 = (bbox_coords[2] - x_offset) / scale_factor
                y2 = (bbox_coords[3] - y_offset) / scale_factor

                # 이미지 경계를 벗어난 bbox 조정
                x1 = max(0, min(x1, orig_width))
                y1 = max(0, min(y1, orig_height))
                x2 = max(0, min(x2, orig_width))
                y2 = max(0, min(y2, orig_height))

                # 이미지 bboxes 목록 업데이트
                current_image = self.images[self.current_index]
                if current_image in self.image_bboxes:
                    # 현재 선택된 bbox의 인덱스 찾기
                    selected_idx = None
                    min_diff = float('inf')
                    
                    for i, bbox in enumerate(self.image_bboxes[current_image]):
                        # 저장된 bbox와 현재 선택된 bbox의 차이 계산
                        diff = abs(bbox[1] - x1) + abs(bbox[2] - y1) + \
                            abs(bbox[3] - x2) + abs(bbox[4] - y2)
                        if diff < min_diff:
                            min_diff = diff
                            selected_idx = i
                    
                    if self.bbox_index is not None:
                        # image_bboxes 업데이트
                        class_id = self.image_bboxes[current_image][self.bbox_index][0]
                        self.image_bboxes[current_image][self.bbox_index] = (
                            class_id, x1, y1, x2, y2
                        )
                        
                        # YOLO 포맷으로 변환
                        x_center = ((x1 + x2) / 2) / orig_width
                        y_center = ((y1 + y2) / 2) / orig_height
                        width = abs(x2 - x1) / orig_width
                        height = abs(y2 - y1) / orig_height
                        
                        # yolo_bboxes 업데이트
                        self.yolo_bboxes[current_image][self.bbox_index] = (
                            class_id,
                            x_center, y_center,
                            width, height
                        )

                    elif selected_idx is not None:
                        # image_bboxes 업데이트
                        class_id = self.image_bboxes[current_image][selected_idx][0]
                        self.image_bboxes[current_image][selected_idx] = (
                            class_id, x1, y1, x2, y2
                        )
                        
                        # YOLO 포맷으로 변환
                        x_center = ((x1 + x2) / 2) / orig_width
                        y_center = ((y1 + y2) / 2) / orig_height
                        width = abs(x2 - x1) / orig_width
                        height = abs(y2 - y1) / orig_height
                        
                        # yolo_bboxes 업데이트
                        self.yolo_bboxes[current_image][selected_idx] = (
                            class_id,
                            x_center, y_center,
                            width, height
                        )
                
                self.selected_bbox = tuple(bbox_coords)

            elif self.selected_bbox_id:  # bbox 이동
                dx = cur_x - self.drag_start_x
                dy = cur_y - self.drag_start_y
                self.canvas.move(self.selected_bbox_id, dx, dy)
                
                # 조절점도 함께 이동
                for handle in self.resize_handles:
                    self.canvas.move(handle, dx, dy)
                
                self.drag_start_x = cur_x
                self.drag_start_y = cur_y
                
                # 현재 캔버스 좌표
                new_coords = self.canvas.coords(self.selected_bbox_id)
                
                # 이미지 크기와 스케일 정보
                image = Image.open(self.images[self.current_index])
                orig_width, orig_height = image.size
                
                if self.fullscreen:
                    screen_width = self.root.winfo_screenwidth()
                    screen_height = self.root.winfo_screenheight()
                else:
                    screen_width = self.root.winfo_width()
                    screen_height = self.root.winfo_height() - self.menu_frame.winfo_height()
                
                scale_x = screen_width / orig_width
                scale_y = screen_height / orig_height
                scale_factor = min(scale_x, scale_y) * self.scale
                x_offset = self.image_position[0]
                y_offset = self.image_position[1]

                # 캔버스 좌표를 원본 이미지 좌표로 변환
                x1 = (new_coords[0] - x_offset) / scale_factor
                y1 = (new_coords[1] - y_offset) / scale_factor
                x2 = (new_coords[2] - x_offset) / scale_factor
                y2 = (new_coords[3] - y_offset) / scale_factor

                # 이미지 bboxes 목록 업데이트
                current_image = self.images[self.current_index]
                if current_image in self.image_bboxes:
                    # 현재 선택된 bbox의 인덱스 찾기
                    selected_idx = None
                    min_diff = float('inf')
                    
                    for i, bbox in enumerate(self.image_bboxes[current_image]):
                        # 저장된 bbox와 현재 선택된 bbox의 차이 계산
                        diff = abs(bbox[1] - (x1 + dx/scale_factor)) + \
                            abs(bbox[2] - (y1 + dy/scale_factor)) + \
                            abs(bbox[3] - (x2 + dx/scale_factor)) + \
                            abs(bbox[4] - (y2 + dy/scale_factor))
                        if diff < min_diff:
                            min_diff = diff
                            selected_idx = i

                    if self.bbox_index is not None:
                        # image_bboxes 업데이트
                        class_id = self.image_bboxes[current_image][self.bbox_index][0]
                        self.image_bboxes[current_image][self.bbox_index] = (
                            class_id, x1, y1, x2, y2
                        )
                        
                        # YOLO 포맷으로 변환
                        x_center = ((x1 + x2) / 2) / orig_width
                        y_center = ((y1 + y2) / 2) / orig_height
                        width = abs(x2 - x1) / orig_width
                        height = abs(y2 - y1) / orig_height
                        
                        # yolo_bboxes 업데이트
                        self.yolo_bboxes[current_image][self.bbox_index] = (
                            class_id,
                            x_center, y_center,
                            width, height
                        )                        
                    elif selected_idx is not None:
                        # image_bboxes 업데이트
                        class_id = self.image_bboxes[current_image][selected_idx][0]
                        self.image_bboxes[current_image][selected_idx] = (
                            class_id, x1, y1, x2, y2
                        )
                        
                        # YOLO 포맷으로 변환
                        x_center = ((x1 + x2) / 2) / orig_width
                        y_center = ((y1 + y2) / 2) / orig_height
                        width = abs(x2 - x1) / orig_width
                        height = abs(y2 - y1) / orig_height
                        
                        # yolo_bboxes 업데이트
                        self.yolo_bboxes[current_image][selected_idx] = (
                            class_id,
                            x_center, y_center,
                            width, height
                        )
                
                self.selected_bbox = tuple(new_coords)

    def on_mouse_up(self, event):
        if self.mode == "draw" and self.current_bbox:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            
            # 최소 크기 확인
            if abs(cur_x - self.start_x) > 5 and abs(cur_y - self.start_y) > 5:
                if self.images and self.current_index < len(self.images):
                    current_image = self.images[self.current_index]
                    if current_image not in self.image_bboxes:
                        self.image_bboxes[current_image] = []

                    if current_image not in self.yolo_bboxes:
                        self.yolo_bboxes[current_image] = []      

                    # 이전 선택 해제
                    if self.selected_bbox_id:
                        self.deselect_bbox()

                    # 이미지 크기와 스케일 정보
                    image = Image.open(current_image)
                    orig_width, orig_height = image.size
                    
                    if self.fullscreen:
                        screen_width = self.root.winfo_screenwidth()
                        screen_height = self.root.winfo_screenheight()
                    else:
                        screen_width = self.root.winfo_width()
                        screen_height = self.root.winfo_height() - self.menu_frame.winfo_height()
                    
                    scale_x = screen_width / orig_width
                    scale_y = screen_height / orig_height
                    scale_factor = min(scale_x, scale_y) * self.scale
                    x_offset = self.image_position[0]
                    y_offset = self.image_position[1]

                    # 캔버스 좌표를 원본 이미지 좌표로 변환
                    x1 = (min(self.start_x, cur_x) - x_offset) / scale_factor
                    y1 = (min(self.start_y, cur_y) - y_offset) / scale_factor
                    x2 = (max(self.start_x, cur_x) - x_offset) / scale_factor
                    y2 = (max(self.start_y, cur_y) - y_offset) / scale_factor

                    # 이미지 경계를 벗어난 bbox 조정
                    x1 = max(0, min(x1, orig_width))
                    y1 = max(0, min(y1, orig_height))
                    x2 = max(0, min(x2, orig_width))
                    y2 = max(0, min(y2, orig_height))

                    class_name = self.class_var.get()
                    class_id = self.class_name_to_idx.get(class_name, 0)  # 없으면 0을 기본값으로 사용

                    # bbox 추가
                    self.image_bboxes[current_image].append((
                        class_id,  # class_id
                        x1, y1, x2, y2
                    ))
                    
                    # YOLO 포맷으로 변환
                    x_center = ((x1 + x2) / 2) / orig_width
                    y_center = ((y1 + y2) / 2) / orig_height
                    width = abs(x2 - x1) / orig_width
                    height = abs(y2 - y1) / orig_height
                    
                    # YOLO 포맷 bbox 추가
                    self.yolo_bboxes[current_image].append((
                        class_id,
                        x_center, y_center,
                        width, height
                    ))

                    # bbox 인덱스 설정
                    self.bbox_index = len(self.image_bboxes[current_image]) - 1
                    
                    # 현재 선택된 bbox 좌표 저장
                    self.selected_bbox = (
                        min(self.start_x, cur_x),
                        min(self.start_y, cur_y),
                        max(self.start_x, cur_x),
                        max(self.start_y, cur_y)
                    )

            else:
                # 최소 크기보다 작은 경우 bbox 삭제
                self.canvas.delete(self.current_bbox)

            self.show_current_image()                
            self.current_bbox = None
            self.drag_data = {"x": 0, "y": 0}  # 드래그 데이터 초기화
            
        elif self.mode == "select":
            self.resizing = False
            self.resize_handle = None
            self.resize_corner = None

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes('-fullscreen', self.fullscreen)
        if self.fullscreen:
            self.toggle_btn.configure(text="창모드")
            self.menu_frame.pack_forget()
        else:
            self.toggle_btn.configure(text="전체화면")
            self.menu_frame.pack(fill='x', padx=5, pady=5)
        self.show_current_image()
    
    def update_counter(self):
        if self.images:
            self.counter_label.configure(text=f"{self.current_index + 1}/{len(self.images)}")
        else:
            self.counter_label.configure(text="0/0")
    
    def show_current_image(self):
        self.canvas.delete("all")  # 캔버스 초기화
        
        if self.images and 0 <= self.current_index < len(self.images):
            # 이미지 로드
            image = Image.open(self.images[self.current_index])
            
            if self.fullscreen:
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
            else:
                screen_width = self.root.winfo_width()
                screen_height = self.root.winfo_height() - self.menu_frame.winfo_height()
            
            # 원본 이미지 크기 저장
            orig_width, orig_height = image.size
            
            # 이미지 비율 유지하며 크기 조정
            scale_x = screen_width / orig_width
            scale_y = screen_height / orig_height
            base_scale = min(scale_x, scale_y)
            
            # 확대/축소 적용
            final_scale = base_scale * self.scale
            new_width = int(orig_width * final_scale)
            new_height = int(orig_height * final_scale)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # PhotoImage로 변환
            self.current_image_tk = ImageTk.PhotoImage(image)
            
            # 이미지를 캔버스 중앙에 배치
            # x = (screen_width - new_width) // 2
            # y = (screen_height - new_height) // 2
            
            # # 이미지 위치 저장
            # self.image_position = (x, y)

            # 이미지 위치 설정 - 패닝 위치 유지
            if not hasattr(self, 'image_position') or self.image_position is None:
                # 초기 위치 (중앙)
                x = (screen_width - new_width) // 2
                y = (screen_height - new_height) // 2
                self.image_position = (x, y)
            else:
                # 현재 패닝 위치 사용
                x, y = self.image_position
            
            # 이미지 그리기
            self.canvas.create_image(x, y, anchor='nw', image=self.current_image_tk)
            
            # bbox 스케일 조정을 위한 비율 계산
            scale_factor = base_scale * self.scale
            
            # 저장된 bbox 그리기
            current_image = self.images[self.current_index]
            if current_image in self.image_bboxes:
                for bbox in self.image_bboxes[current_image]:
                    # bbox 좌표는 이미 base_scale이 적용된 상태이므로
                    # self.scale만 추가로 적용
                    scaled_x1 = x +bbox[1] * scale_factor
                    scaled_y1 = y + bbox[2] * scale_factor
                    scaled_x2 = x + bbox[3] * scale_factor
                    scaled_y2 = y + bbox[4] * scale_factor
                    
                    # self.canvas.create_rectangle(
                    #     scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                    #     outline='red', width=1
                    # )
                    class_id = bbox[0]
                    color = self.get_bbox_color(class_id)
                    label_text = self.class_names[class_id]
                    
                    # bbox 그리기
                    bbox_id = self.canvas.create_rectangle(
                        scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                        outline=color, width=1
                    )

                    self.canvas.create_text(
                        scaled_x1, scaled_y1 - 5,
                        text=label_text,
                        fill=color,
                        anchor='sw',
                        font=('Arial', 8)
                    )

    
    def next_image(self, event=None):
        if self.images:
            self.current_index = (self.current_index + 1) % len(self.images)
            self.update_counter()
            self.show_current_image()

    def previous_image(self, event=None):
        if self.images:
            self.current_index = (self.current_index - 1) % len(self.images)
            self.update_counter()
            self.show_current_image()

    def exit_fullscreen(self, event=None):
        if self.fullscreen:
            self.toggle_fullscreen()

    # 새로운 메서드 추가
    def add_new_class(self):
        """새로운 클래스 추가"""
        class_name = simpledialog.askstring("새 클래스", "새로운 클래스 이름을 입력하세요:")
        if class_name:
            self.bbox_saver.add_class(class_name)
            self.class_combo['values'] = self.bbox_saver.get_classes()

    def save_current_image_bboxes(self, event=None):
        """현재 이미지의 bbox 데이터를 YOLO 형식으로 저장"""
        if not self.images or self.current_index >= len(self.images):
            return
        
        current_image = self.images[self.current_index]
        if current_image not in self.yolo_bboxes:
            return
        
        # # 저장 디렉토리 선택
        # # save_dir = filedialog.askdirectory(title="저장할 디렉토리 선택")
        # # if not save_dir:  # 사용자가 취소를 누른 경우
        # #     return
        # save_dir = self.file_paths

        # # 이미지 파일명 가져오기
        # image_filename = os.path.basename(current_image)
        
        # # 저장할 전체 경로 생성
        # print(f"save_dir : {save_dir}")
        # save_path = os.path.join(save_dir, image_filename)
        
        # # 캔버스 크기와 이미지 위치 정보 가져오기
        # canvas_width = self.canvas.winfo_width()
        # canvas_height = self.canvas.winfo_height()
        # image_x = self.image_position[0]
        # image_y = self.image_position[1]
        
        # # bbox 데이터 저장 (캔버스 크기와 이미지 위치 정보 전달)
        # if self.bbox_saver.save_bbox_data(save_path, 
        #                                 self.yolo_bboxes[current_image],
        #                                 canvas_size=(canvas_width, canvas_height),
        #                                 image_position=(image_x, image_y)):
        #     messagebox.showinfo("저장 완료", f"bbox 데이터가 저장되었습니다.\n저장 위치: {save_dir}")
        # else:
        #     messagebox.showerror("저장 실패", "bbox 데이터 저장 중 오류가 발생했습니다.")
        self.save_yolo_annotations()

    # 모델 예측
    def auto_detect(self, event=None):
        if not self.images or self.current_index >= len(self.images):
            messagebox.showwarning("경고", "이미지를 먼저 로드해주세요.")
            return
            
        try:
            current_image = self.images[self.current_index]
            # 현재 이미지 로드
            orig_image = cv2.imread(current_image)
            if orig_image is None:
                raise Exception("이미지를 불러올 수 없습니다.")
                
            # 기존 bbox 삭제
            self.clear_bboxes()
            
            # 이미지 bboxes 초기화
            self.image_bboxes[current_image] = []

            # 학습용 YOLO 포맷 bbox 초기화
            self.yolo_bboxes[current_image] = []

            # 검출 수행
            bbox_coordinates = self.detector.detect(orig_image)
            print(f"bbox_coordinates: {bbox_coordinates}")

            # PIL로 이미지 로드
            image = Image.open(current_image)
            
            # 화면 크기 계산
            if self.fullscreen:
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
            else:
                screen_width = self.root.winfo_width()
                screen_height = self.root.winfo_height() - self.menu_frame.winfo_height()
            
            # 이미지 크기 조정 (show_current_image와 동일)
            image.thumbnail((screen_width, screen_height), Image.Resampling.LANCZOS)
            
            # 조정된 이미지 크기
            image_width = image.size[0]
            image_height = image.size[1]

            # 원본 이미지와 조정된 이미지의 비율 계산
            scale_x = image_width / orig_image.shape[1]
            scale_y = image_height / orig_image.shape[0]

            orig_width = orig_image.shape[1]
            orig_height = orig_image.shape[0]
            
            for x1, y1, x2, y2, cls in bbox_coordinates:
                # 좌표를 조정된 이미지 크기에 맞게 스케일링하고 오프셋 적용
                canvas_x1 = int(x1 * scale_x)
                canvas_y1 = int(y1 * scale_y)
                canvas_x2 = int(x2 * scale_x)
                canvas_y2 = int(y2 * scale_y)

                # bbox 생성
                self.image_bboxes[current_image].append((
                    cls,  # model 이 예측한 cls
                    canvas_x1,  # x1
                    canvas_y1,  # y1
                    canvas_x2,  # x2
                    canvas_y2   # y2
                ))

                # YOLO 포맷: <class> <x_center> <y_center> <width> <height>
                x_center = ((x1 + x2) / 2) / orig_width
                y_center = ((y1 + y2) / 2) / orig_height
                width = (x2 - x1) / orig_width
                height = (y2 - y1) / orig_height

                # 학습용 YOLO bbox 저장
                self.yolo_bboxes[current_image].append((
                    cls,  # model 이 예측한 cls
                    x_center,  # 중심 x
                    y_center,  # 중심 y
                    width,     # 너비
                    height     # 높이
                ))

            # 화면 업데이트
            self.show_current_image()
            messagebox.showinfo("완료", f"{len(bbox_coordinates)}개의 객체가 검출되었습니다.")
            
        except Exception as e:
            messagebox.showerror("오류", f"검출 중 오류가 발생했습니다: {str(e)}")

    # 단축키 설정
    def select_mode(self, event=None):
        self.mode_var.set("select")
        self.change_mode() 

    def draw_mode(self, event=None):
        self.mode_var.set("draw")
        self.change_mode() 

    def next_image_and_save(self, event=None):
        self.save_current_image_bboxes()
        self.next_image()

    def on_mousewheel(self, event):
        """마우스 휠 이벤트 처리"""
        old_scale = self.scale
        current_x, current_y = self.image_position
        
        # 마우스 위치의 이미지 상대 좌표 계산
        img_x = (event.x - current_x) / old_scale
        img_y = (event.y - current_y) / old_scale
        
        # 스케일 조정
        if event.num == 5 or event.delta < 0:  # Windows/축소
            self.scale = max(self.min_scale, self.scale * 0.9)
        if event.num == 4 or event.delta > 0:  # Linux/확대
            self.scale = min(self.max_scale, self.scale * 1.1)
        
        if old_scale != self.scale:
            # 새로운 스케일에서의 캔버스 좌표 계산
            new_x = event.x - (img_x * self.scale)
            new_y = event.y - (img_y * self.scale)
            
            # 이미지 위치 업데이트
            self.image_position = (new_x, new_y)
            
            # 이동 거리 계산 및 캔버스 이동
            dx = new_x - current_x
            dy = new_y - current_y
            self.canvas.move("all", dx, dy)
            
            self.show_current_image()

    def start_pan(self, event):
        """패닝 시작"""
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="hand2")

        # 패닝 시작할 때의 이미지 위치 저장
        self.pan_start_image_pos = self.image_position

    def pan(self, event):
        """패닝 중"""
        if self.pan_start_x is not None:
            # 이동 거리 계산
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            # 모든 캔버스 항목 이동
            self.canvas.move("all", dx, dy)

            # 이미지 위치 업데이트
            x, y = self.pan_start_image_pos
            self.image_position = (x + dx, y + dy)
            
            # 시작점 업데이트
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.pan_start_image_pos = self.image_position

    def end_pan(self, event):
        """패닝 종료"""
        self.pan_start_x = None
        self.pan_start_y = None
        self.pan_start_image_pos = None
        if self.mode == "select":
            self.canvas.config(cursor="hand2")
        elif self.mode == "draw":
            self.canvas.config(cursor="crosshair")
        else:
            self.canvas.config(cursor="")

    def save_yolo_annotations(self):
        """
        현재 이미지의 bbox를 YOLO 포맷으로 저장
        YOLO 포맷: <class> <x_center> <y_center> <width> <height>
        모든 값은 0~1 사이로 정규화됨
        """
        try:
            if not self.images or self.current_index >= len(self.images):
                messagebox.showwarning("경고", "저장할 이미지가 없습니다.")
                return

            current_image = self.images[self.current_index]
            
            # 이미지가 없거나 bbox가 없는 경우
            if not current_image or current_image not in self.yolo_bboxes:
                messagebox.showwarning("경고", "저장할 bbox가 없습니다.")
                return

            # .txt 파일 경로 생성
            txt_path = os.path.splitext(current_image)[0] + '.txt'
            
            # bbox 정보 저장
            with open(txt_path, 'w', encoding='utf-8') as f:
                for bbox in self.yolo_bboxes[current_image]:
                    class_id, x_center, y_center, width, height = bbox
                    # 좌표값을 소수점 6자리까지 저장
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # 성공 메시지 표시
            messagebox.showinfo("완료", f"Annotation이 저장되었습니다.\n저장 위치: {txt_path}")

        except Exception as e:
            messagebox.showerror("오류", f"저장 중 오류가 발생했습니다: {str(e)}")

    def load_images(self):
        file_paths = filedialog.askopenfilenames(
            title="이미지 파일 선택",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_paths:
            self.process_file_paths(file_paths)

    def setup_drag_drop(self):
        # 드래그 앤 드롭을 위한 이벤트 바인딩
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.handle_drop)

    def handle_drop(self, event):
        # 드롭된 파일 경로를 처리
        file_paths = self.root.tk.splitlist(event.data)
        # 이미지 파일만 필터링
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        image_files = [f for f in file_paths if f.lower().endswith(valid_extensions)]
        
        if image_files:
            self.process_file_paths(image_files)

    def process_file_paths(self, file_paths):
        if file_paths:
            self.file_paths = os.path.dirname(file_paths[0])
            self.images = list(file_paths)
            self.current_index = 0
            self.update_counter()
            self.show_current_image()

    def get_bbox_color(self, class_id):
            """클래스 ID에 따른 색상 반환"""
            class_id_str = str(class_id)
            if class_id_str in self.class_colors:
                return self.class_colors[class_id_str]
            
            # 기본 색상: 새로운 클래스를 위한 색상 자동 생성
            if not hasattr(self, '_color_index'):
                self._color_index = len(self.class_colors)
            
            # HSV 색상 공간을 사용하여 새로운 색상 생성
            import colorsys
            hue = (self._color_index * 0.618033988749895) % 1  # 황금비를 사용한 색상 분포
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
            color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
            
            # 새로운 색상 저장
            self.class_colors[class_id_str] = color
            self._color_index += 1
            
            return color

# def main():
#     # root = tk.Tk()
#     root = TkinterDnD.Tk()  # 메인 윈도우 하나만 생성
#     app = ImageViewer(root)
#     root.mainloop()

# if __name__ == "__main__":
#     main()