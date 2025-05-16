import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinterdnd2 import DND_FILES, TkinterDnD
from save_txt import YOLOBBoxSaver
from model_predict import YOLODetector
from PIL import Image, ImageTk
import cv2
import os
import subprocess
import threading
import yaml
import shutil
import random
import time
from Utils import ImageViewer


class ModernUIApp(ImageViewer):
    def __init__(self, root):
        # ImageViewer의 초기화를 건너뛰고 직접 초기화
        self.root = root
        self.root.title("BBOX Label Studio")
        self.style = ttk.Style(theme='darkly')
        
        # 기본 변수 초기화
        self.initialize_base_variables()
        
        # UI 구성
        self.create_modern_ui()
        
        # UI 의존성 있는 변수들 초기화
        self.initialize_ui_dependent_variables()
        
        # 이벤트 바인딩
        self.setup_bindings()
        self.setup_drag_drop()

    def initialize_base_variables(self):
        """UI에 의존성이 없는 기본 변수들 초기화"""
        # directory 설정
        self.file_paths = None
        
        # 화면 모드
        self.fullscreen = False
        
        # 모드 설정
        self.mode = "view"
        self.mode_var = tk.StringVar(value="view")

        # bbox 관련 변수
        self.start_x = None
        self.start_y = None
        self.current_bbox = None
        self.image_bboxes = {}
        self.selected_bbox = None
        self.selected_bbox_id = None
        self.drag_start_x = None
        self.drag_start_y = None
        self.bbox_index = None
        
        # 리사이즈 관련
        self.resize_handles = []
        self.resizing = False
        self.resize_handle = None
        self.resize_corner = None

        # YOLO 학습용 BBOX
        self.yolo_bboxes = {}
        
        # 이미지 관련 변수
        self.images = []
        self.current_index = 0
        self.current_image_tk = None
        self.image_position = (0, 0)
        
        # 이미지 확대/축소 관련 변수
        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0

        # 패닝 관련 변수
        self.pan_start_x = None
        self.pan_start_y = None
        self.pan_start_image_pos = None

        # YOLO 저장 관리자 초기화
        self.bbox_saver = YOLOBBoxSaver()
        
        # 클래스 정보 초기화
        self.class_names = self.load_class_names()
        self.class_name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.class_idx_to_name = {idx: name for idx, name in enumerate(self.class_names)}
        
        # 클래스 변수 초기화 (클래스가 있는 경우 첫 번째 클래스, 없으면 빈 문자열)
        self.class_var = tk.StringVar(value=self.class_names[0] if self.class_names else "")

        
        # 모델 자동 예측
        try:
            self.model_path = r"D:\spine_detection\models\detection\.pt"
            self.detector = YOLODetector(
                model_path=self.model_path,
                conf_threshold=0.5,
                min_height=10
            )
            self.model_state = True
        except:
            # 모델 변수 초기화
            self.detector = None  # 모델은 나중에 로드
            self.model_path = None  # 모델 경로 저장      
            self.model_state = False      

    def initialize_ui_dependent_variables(self):
        """UI 요소가 생성된 후 초기화해야 하는 변수들"""
        # 키보드 바인딩
        self.root.bind('<n>', self.next_image_and_save)
        self.root.bind('<p>', self.previous_image)
        self.root.bind('<Escape>', self.exit_fullscreen)
        self.root.bind('<f>', self.toggle_fullscreen)
        self.root.bind('<Delete>', self.delete_selected_bbox)
        self.root.bind('<s>', self.select_mode)  
        self.root.bind('<d>', self.draw_mode)  
        self.root.bind('<a>', self.auto_detect)
        self.root.bind('<c>', self.cycle_class) # 클래스 순환을 위한 바인딩 추가
        self.root.bind('<l>', self.create_navigation_thread)  # 'l' 키 바인딩 추가
            

        # 마우스 이벤트 바인딩
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<MouseWheel>', self.on_mousewheel)
        self.canvas.bind('<Button-4>', self.on_mousewheel)
        self.canvas.bind('<Button-5>', self.on_mousewheel)
        
        # 패닝 바인딩
        self.canvas.bind('<Button-2>', self.start_pan)
        self.canvas.bind('<B2-Motion>', self.pan)
        self.canvas.bind('<ButtonRelease-2>', self.end_pan)
        
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

    def create_modern_ui(self):
        """모던 UI 생성"""
        # 메인 컨테이너
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=BOTH, expand=YES, padx=10, pady=10)
        
        self.create_toolbar()
        self.create_main_area()
        self.create_sidebar()
        self.create_statusbar()


    def create_toolbar(self):
        """상단 툴바 생성"""
        toolbar = ttk.Frame(self.main_container)
        self.menu_frame = toolbar
        toolbar.pack(fill=X, pady=(0, 10))
        
        # 파일 그룹
        file_group = ttk.LabelFrame(toolbar, text="File", padding=5)
        file_group.pack(side=LEFT, padx=5)
        
        ttk.Button(file_group, text="📂 Open", 
                  command=self.load_images,
                  style='primary.Outline.TButton').pack(side=LEFT, padx=2)
        
        ttk.Button(file_group, text="💾 Save", 
                  command=self.save_current_image_bboxes,
                  style='success.Outline.TButton').pack(side=LEFT, padx=2)
        
        # 모델 그룹 추가
        model_group = ttk.LabelFrame(toolbar, text="Model", padding=5)
        model_group.pack(side=LEFT, padx=5)
        
        ttk.Button(model_group, text="📦 Load Model", 
                  command=self.load_model,
                  style='info.Outline.TButton').pack(side=LEFT, padx=2)
        
        # 모델 상태 표시
  
        if not self.model_state:
            self.model_status = ttk.Label(model_group, text="Model: Not Loaded")
        else:
            model_name = os.path.basename(self.model_path)
            self.model_status = ttk.Label(model_group,text=f"Model: {model_name}")
   

        self.model_status.pack(side=LEFT, padx=5)

        # 도구 그룹
        tools_group = ttk.LabelFrame(toolbar, text="Tools", padding=5)
        tools_group.pack(side=LEFT, padx=5)
        
        ttk.Radiobutton(tools_group, text="🔍 View", 
                       variable=self.mode_var, value="view",
                       command=self.change_mode).pack(side=LEFT, padx=2)
        
        ttk.Radiobutton(tools_group, text="✏️ Draw", 
                       variable=self.mode_var, value="draw",
                       command=self.change_mode).pack(side=LEFT, padx=2)
        
        ttk.Radiobutton(tools_group, text="👆 Select", 
                       variable=self.mode_var, value="select",
                       command=self.change_mode).pack(side=LEFT, padx=2)
        
        # 작업 그룹
        action_group = ttk.LabelFrame(toolbar, text="Actions", padding=5)
        action_group.pack(side=LEFT, padx=5)
        
        ttk.Button(action_group, text="🤖 Auto Detect", 
                  command=self.auto_detect,
                  style='info.Outline.TButton').pack(side=LEFT, padx=2)
        
        ttk.Button(action_group, text="🗑️ Clear", 
                  command=self.clear_bboxes,
                  style='danger.Outline.TButton').pack(side=LEFT, padx=2)   

        # 학습 그룹
        train_group = ttk.LabelFrame(toolbar, text="Training", padding=5)
        train_group.pack(side=LEFT, padx=5)
        
        ttk.Button(train_group, text="🚀 Train YOLO", 
                  command=self.prepare_and_train,
                  style='warning.Outline.TButton').pack(side=LEFT, padx=2)
        
       # 로그 그룹 추가
        log_group = ttk.LabelFrame(toolbar, text="Logs", padding=5)
        log_group.pack(side=LEFT, padx=5)
        
        ttk.Button(log_group, text="📋 View Log", 
                  command=self.create_navigation_thread,
                  style='success.Outline.TButton').pack(side=LEFT, padx=2)
        

    def create_main_area(self):
        """메인 작업 영역 생성"""
        main_frame = ttk.Frame(self.main_container)
        main_frame.pack(fill=BOTH, expand=YES, pady=5)
        
        # 캔버스를 포함할 프레임
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=BOTH, expand=YES)
        
        # 캔버스
        self.canvas = tk.Canvas(canvas_frame, 
                            bg='#2c2c2c',
                            highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=YES)
        
        # 하단 네비게이션 프레임
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=X, side=BOTTOM, pady=5)
                
    def create_sidebar(self):
        """우측 사이드바 생성"""
        sidebar = ttk.Frame(self.main_container)
        sidebar.pack(side=RIGHT, fill=Y, pady=(10, 10), padx=(10,0))
        
        # 클래스 선택
        class_frame = ttk.LabelFrame(sidebar, text="Class Selection", padding=10)
        class_frame.pack(fill=X, pady=5)
        
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(class_frame, 
                                      textvariable=self.class_var,
                                      values=self.bbox_saver.get_classes(),
                                      state='readonly')
        
        self.class_combo.pack(fill=X, pady=5)
        self.class_combo.set(self.class_names[0])
        
        ttk.Button(class_frame, text="➕ Add Class",
                  command=self.add_new_class,
                  style='info.TButton').pack(fill=X)

        # image 선택
        image_mov_frame = ttk.LabelFrame(sidebar, text="Image Selection", padding=10)
        image_mov_frame.pack(fill=X, pady=5)

        # Previous 버튼
        ttk.Button(image_mov_frame, text="◀️ Previous",
                command=self.previous_image).pack(side=LEFT, padx=5)
        
        # 카운터 레이블
        self.counter_label = ttk.Label(image_mov_frame, text="0/0")
        self.counter_label.pack(side=LEFT, pady=5)
        
        # Next 버튼
        ttk.Button(image_mov_frame, text="Next ▶️",
                command=self.next_image).pack(side=LEFT, padx=5)
        
    def create_navigation_window(self):
        """네비게이션 윈도우 생성"""
        self.nav_window = tk.Toplevel(self.main_container)
        self.nav_window.title("Training Output") 
        self.nav_window.geometry("1000x600")
        
        # 메인 프레임 생성
        main_frame = ttk.Frame(self.nav_window)
        main_frame.pack(fill=BOTH, expand=YES, padx=5, pady=5)
        
        # 출력 프레임 생성 (LabelFrame 사용)
        output_frame = ttk.LabelFrame(main_frame, text="Training Log", padding=10)
        output_frame.pack(fill=BOTH, expand=YES)
        
        # 스크롤바 추가
        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # 출력 텍스트 영역
        self.nav_output_text = tk.Text(output_frame, 
                                wrap=tk.WORD,
                                yscrollcommand=scrollbar.set,
                                bg='black',
                                fg='white',
                                font=('Consolas', 9))
        self.nav_output_text.pack(fill=BOTH, expand=YES, padx=5, pady=5)
        scrollbar.config(command=self.nav_output_text.yview)
        
        # 하단 버튼 프레임
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=X, pady=(5,0))
        
        # Clear 버튼
        ttk.Button(button_frame, text="Clear Output",
                command=self.clear_all_outputs).pack(fill=X, pady=(5,0))
        
        # 기존 로그 동기화
        if hasattr(self, 'main_output_text'):
            existing_log = self.main_output_text.get(1.0, tk.END)
            if existing_log.strip():  # 빈 로그가 아닌 경우에만
                self.nav_output_text.insert(tk.END, existing_log)
                self.nav_output_text.see(tk.END)

        def update_nav_position():
            main_x = self.main_container.winfo_x()
            main_y = self.main_container.winfo_y()
            
            # 메인 윈도우 오른쪽에 위치하도록 수정
            nav_x = main_x + self.main_container.winfo_width() + 10
            nav_y = main_y  # 같은 높이에 위치
            self.nav_window.geometry(f"+{nav_x}+{nav_y}")
        
        # 초기 위치 설정
        update_nav_position()
        
        # 윈도우 크기 조절이나 이동 시 위치 업데이트
        self.main_container.bind('<Configure>', lambda e: update_nav_position())

    def create_statusbar(self):
        """하단 상태바 생성"""
        statusbar = ttk.Frame(self.main_container)
        statusbar.pack(fill=X, pady=(10, 0))
        
        # self.status_label = ttk.Label(statusbar, text="Ready")
        # self.status_label.pack(side=LEFT)
        
        # self.coords_label = ttk.Label(statusbar, text="Mouse: 0, 0")
        # self.coords_label.pack(side=RIGHT)

        # 학습 출력 창 추가
        output_frame = ttk.LabelFrame(statusbar, text="Training Output", padding=10)
        output_frame.pack(fill=BOTH, expand=YES, pady=5)
        
        # 스크롤바 추가
        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # 메인 창의 출력 텍스트 영역
        self.main_output_text = tk.Text(output_frame, 
                                    wrap=tk.WORD, 
                                    height=8,
                                    yscrollcommand=scrollbar.set,
                                    bg='black',
                                    fg='white',
                                    font=('Consolas', 9))
        
        self.main_output_text.pack(fill=BOTH, expand=YES)
        scrollbar.config(command=self.main_output_text.yview)
        
        # Clear 버튼
        ttk.Button(output_frame, text="Clear Output", 
                command=self.clear_all_outputs).pack(fill=X, pady=(5,0))

    def setup_bindings(self):
        """이벤트 바인딩"""
        # 키보드 바인딩
        self.root.bind('<n>', self.next_image_and_save)
        self.root.bind('<p>', self.previous_image)
        self.root.bind('<Delete>', self.delete_selected_bbox)
        self.root.bind('<s>', self.select_mode)
        self.root.bind('<d>', self.draw_mode)
        self.root.bind('<a>', self.auto_detect)
        
        # 마우스 이벤트
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<MouseWheel>', self.on_mousewheel)
        # self.canvas.bind('<Motion>', self.update_mouse_position)
        
        # 패닝
        self.canvas.bind('<Button-2>', self.start_pan)
        self.canvas.bind('<B2-Motion>', self.pan)
        self.canvas.bind('<ButtonRelease-2>', self.end_pan)

    def update_mouse_position(self, event):
        """마우스 위치 업데이트"""
        self.coords_label.config(text=f"Mouse: {event.x}, {event.y}")

    # def update_status(self, message):
    #     """상태바 메시지 업데이트"""
    #     self.status_label.config(text=message)

    def load_class_names(self):

        """classes.txt 파일에서 클래스 이름 로드"""
        try:
            classes_path = "classes.txt"  # classes.txt 파일 경로
            if os.path.exists(classes_path):
                with open(classes_path, 'r', encoding='utf-8') as f:
                    # 빈 줄을 제외하고 각 줄을 클래스 이름으로 사용
                    class_names = [line.strip() for line in f if line.strip()]
                return class_names
            return []  # 파일이 없으면 빈 리스트 반환
        except Exception as e:
            print(f"클래스 파일 로드 중 오류 발생: {e}")
            return []
            
    def add_new_class(self):

        """새로운 클래스 추가"""
        class_name = tk.simpledialog.askstring("새 클래스", "새로운 클래스 이름을 입력하세요:")
        if class_name and class_name not in self.class_names:
            # 클래스 리스트에 추가
            self.class_names.append(class_name)
            idx = len(self.class_names) - 1
            self.class_name_to_idx[class_name] = idx
            self.class_idx_to_name[idx] = class_name
            
            # UI 업데이트
            self.class_combo['values'] = self.class_names
            
            # classes.txt 파일 업데이트
            try:
                with open('classes.txt', 'w', encoding='utf-8') as f:
                    for name in self.class_names:
                        f.write(f"{name}\n")
                messagebox.showinfo("성공", f"새 클래스 '{class_name}' 추가됨")
            except Exception as e:
                messagebox.showerror("오류", f"클래스 파일 저장 중 오류 발생: {e}")

    def cycle_class(self, event=None):

        """다음 클래스로 순환"""
        if not self.class_names:  # 클래스가 없으면 반환
            return
            
        # 현재 선택된 클래스 이름
        current_class = self.class_var.get()
        
        # 현재 클래스의 인덱스 찾기
        try:
            current_idx = self.class_names.index(current_class)
        except ValueError:
            current_idx = -1
        
        # 다음 클래스 선택
        next_idx = (current_idx + 1) % len(self.class_names)
        next_class = self.class_names[next_idx]
        
        # 콤보박스 업데이트
        self.class_combo.set(next_class)
        
        # 상태바에 알림 (상태바가 있는 경우)
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Class changed to: {next_class}")

    def load_model(self):
            """모델 파일 선택 및 로드"""
            model_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[
                    ("PyTorch Models", "*.pt"),
                    ("ONNX Models", "*.onnx"),
                    ("All Files", "*.*")
                ]
            )
            
            if model_path:
                try:
                    # 기존 모델 정리
                    if self.detector is not None:
                        del self.detector
                    
                    # 새 모델 로드
                    self.detector = YOLODetector(
                        model_path=model_path,
                        conf_threshold=0.5,
                        min_height=10
                    )
                    
                    # 모델 경로 저장
                    self.model_path = model_path
                    
                    # 상태 업데이트
                    model_name = os.path.basename(model_path)
                    self.model_status.config(text=f"Model: {model_name}")
                    # self.update_status(f"Model loaded: {model_name}")
                    self.model_state = True
                    messagebox.showinfo("Success", "Model loaded successfully!")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                    self.model_status.config(text="Model: Load Failed")
                    # self.update_status("Model load failed")
                    self.model_state = False

    def read_train_settings(self, settings_path):
        """train_settings.txt 파일에서 학습 설정 읽기"""
        settings = {}
        try:
            # UTF-8 인코딩으로 파일 읽기
            with open(settings_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    # 빈 줄이나 주석 처리된 줄 무시
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        # '=' 구분자로 분리하되, 첫 번째 '='만 사용
                        if '=' not in line:
                            raise ValueError(f"Invalid format at line {line_number}: Missing '=' separator")
                        
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # 키가 비어있는지 확인
                        if not key:
                            raise ValueError(f"Empty key at line {line_number}")
                        
                        settings[key] = value
                        
                    except ValueError as ve:
                        messagebox.showwarning("Warning", f"Skipping line {line_number}: {str(ve)}")
                        continue
                    
            return settings if settings else None
            
        except UnicodeDecodeError:
            # UTF-8 디코딩 실패시 다른 인코딩 시도
            try:
                with open(settings_path, 'r', encoding='cp949') as f:
                    # 위와 동일한 처리 로직
                    for line_number, line in enumerate(f, 1):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        try:
                            if '=' not in line:
                                raise ValueError(f"Invalid format at line {line_number}: Missing '=' separator")
                            
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            if not key:
                                raise ValueError(f"Empty key at line {line_number}")
                            
                            settings[key] = value
                            
                        except ValueError as ve:
                            messagebox.showwarning("Warning", f"Skipping line {line_number}: {str(ve)}")
                            continue
                    
                return settings if settings else None
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read settings file: {str(e)}")
                return None
                
        except FileNotFoundError:
            messagebox.showerror("Error", f"Settings file not found: {settings_path}")
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read settings file: {str(e)}")
            return None
        
    def update_training_output(self, process):
        """비동기로 학습 출력 업데이트"""
        def read_output():
            try:
                # UTF-8로 인코딩 설정
                process.stdout.reconfigure(encoding='utf-8', errors='replace')
                
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        # 출력 내용을 메인 스레드에서 처리
                        self.root.after(0, self.append_to_output, output)
                
                # 프로세스 종료 상태 확인
                return_code = process.poll()
                if return_code == 0:
                    self.root.after(0, self.append_to_output, "\nTraining completed successfully!\n")
                else:
                    self.root.after(0, self.append_to_output, 
                                f"\nTraining ended with return code: {return_code}\n")
            
            except Exception as e:
                # 오류 메시지도 UTF-8로 처리
                error_msg = f"\nError reading output: {str(e)}\n"
                self.root.after(0, self.append_to_output, error_msg)
            
            finally:
                process.stdout.close()

                   
        # 별도 스레드에서 출력 읽기 실행
        thread = threading.Thread(target=read_output)
        thread.daemon = True
        thread.start()

    def append_to_output(self, text):
        """텍스트 위젯에 출력 추가 (메인 스레드에서 실행)"""
        try:
            # # UTF-8로 인코딩/디코딩하여 문자 인코딩 문제 해결
            # encoded_text = text.encode('utf-8', errors='ignore').decode('utf-8')
            # self.output_text.insert(tk.END, encoded_text)
            # self.output_text.see(tk.END)

            # 메인 창 출력
            if hasattr(self, 'main_output_text'):
                encoded_text = text.encode('utf-8', errors='ignore').decode('utf-8')
                self.main_output_text.insert(tk.END, encoded_text)
                self.main_output_text.see(tk.END)
            
            # 네비게이션 창이 열려있는 경우에만 출력
            if hasattr(self, 'nav_window') and self.nav_window.winfo_exists():
                encoded_text = text.encode('utf-8', errors='ignore').decode('utf-8')
                self.nav_output_text.insert(tk.END, encoded_text)
                self.nav_output_text.see(tk.END)

        except Exception as e:
            print(f"Error appending output: {str(e)}")

        # # 메인 창 출력
        # if hasattr(self, 'main_output_text'):
        #     self.main_output_text.insert(tk.END, text)
        #     self.main_output_text.see(tk.END)
        
        # # 네비게이션 창 출력
        # if hasattr(self, 'nav_output_text') and self.nav_window.winfo_exists():
        #     self.nav_output_text.insert(tk.END, text)
        #     self.nav_output_text.see(tk.END)

    def clear_all_outputs(self):
        """모든 출력 창 내용 지우기"""
        if hasattr(self, 'main_output_text'):
            self.main_output_text.delete(1.0, tk.END)
        
        if hasattr(self, 'nav_output_text') and self.nav_window.winfo_exists():
            self.nav_output_text.delete(1.0, tk.END)

    def start_training_process(self, train_command):
        """별도 스레드에서 학습 프로세스 실행"""
        def run_training():
            try:
                process = subprocess.Popen(
                    train_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    text=True,
                    encoding='utf-8'            
                )
                
                # 비동기로 출력 업데이트
                self.update_training_output(process)
                
            except Exception as e:
                self.root.after(0, self.append_to_output, f"학습 시작 오류: {str(e)}\n")
        
        # 학습 프로세스를 별도 스레드에서 실행
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()

    def prepare_and_train(self):
        """YOLO 학습 준비 및 실행"""
        # 작업 폴더 선택
        work_dir = filedialog.askdirectory(
            title="Select Training Data Directory",
            initialdir=self.file_paths if self.file_paths else "/"
        )
        
        if not work_dir:  # 사용자가 취소한 경우
            return
        
        # 설정 파일 확인
        settings_path = './train_settings.txt'
        if not os.path.exists(settings_path):
            messagebox.showwarning("Warning", 
                "train_settings.txt not found!\nPlease create train_settings.txt with training parameters.")
            return

        # 학습 설정 읽기
        settings = self.read_train_settings(settings_path)
        if not settings:
            return

        # 데이터셋 비율 입력 받기
        ratio_window = tk.Toplevel()
        ratio_window.title("Dataset Split Ratio")
        ratio_window.geometry("300x150")
        
        train_ratio = tk.StringVar(value="70")
        val_ratio = tk.StringVar(value="20")
        test_ratio = tk.StringVar(value="10")
        
        tk.Label(ratio_window, text="Train %:").pack()
        tk.Entry(ratio_window, textvariable=train_ratio).pack()
        tk.Label(ratio_window, text="Validation %:").pack()
        tk.Entry(ratio_window, textvariable=val_ratio).pack()
        tk.Label(ratio_window, text="Test %:").pack()
        tk.Entry(ratio_window, textvariable=test_ratio).pack()
        
        ratios = {}
            
        def confirm_ratios():
            try:
                train = float(train_ratio.get())
                val = float(val_ratio.get())
                test = float(test_ratio.get())
                
                if train + val + test != 100:
                    messagebox.showerror("Error", "Ratios must sum to 100%")
                    return
                    
                ratios.update({
                    'train': train / 100,
                    'val': val / 100,
                    'test': test / 100
                })
                ratio_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers")
        
        tk.Button(ratio_window, text="Confirm", command=confirm_ratios).pack()
        ratio_window.wait_window()
        
        if not ratios:  # 사용자가 취소했거나 창을 닫은 경우
            return

        # 데이터셋 구조 생성
        dataset_path = os.path.join(os.path.dirname(work_dir), 'dataset')
        os.makedirs(dataset_path, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dataset_path, split), exist_ok=True)
        
        # 이미지와 라벨 파일 목록 가져오기
        image_files = [f for f in os.listdir(work_dir) if f.endswith('.png') or f.endswith('.jpg')]

        # 파일들을 비율에 따라 분할
        random.shuffle(image_files)
        n_files = len(image_files)
        train_idx = int(n_files * ratios['train'])
        val_idx = train_idx + int(n_files * ratios['val'])
        
        splits = {
            'train': image_files[:train_idx],
            'val': image_files[train_idx:val_idx],
            'test': image_files[val_idx:]
        }

        # 파일 복사
        for split, files in splits.items():
            for img_file in files:
                # 이미지 파일 복사
                src_img = os.path.join(work_dir, img_file)
                dst_img = os.path.join(dataset_path, split, img_file)
                shutil.copy2(src_img, dst_img)
                
                # 라벨 파일 복사
                label_file = img_file.replace('.png', '.txt')
                if os.path.exists(os.path.join(work_dir, label_file)):
                    src_label = os.path.join(work_dir, label_file)
                    dst_label = os.path.join(dataset_path, split, label_file)
                    shutil.copy2(src_label, dst_label)

        # dataset.yaml 파일 생성
        yaml_content = {
            'path': dataset_path,
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'names': self.get_class_names(work_dir)  # 클래스 이름 가져오기
        }
        
        yaml_path = os.path.join(dataset_path, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        # 기본 명령어 구성
        train_command = f"yolo task=detect mode=train"
        
        # 설정 파일의 모든 매개변수 추가
        for key, value in settings.items():
            train_command += f" {key}={value}"

        # Train, Val, Test 비율 설정 입력 받기
        # data 폴더 생성 및 복사 (변경전: 모든 png, txt파일이 work_dir에 있음음)
        # dataset.yaml파일 생성

        # 모델과 데이터 경로는 항상 마지막에 추가
        train_command += f" data={os.path.join(dataset_path, 'dataset.yaml')}"

        # 학습 시작 확인
        if messagebox.askyesno("Start Training", 
                            "Start YOLO training with the following command?\n\n" + 
                            train_command):
            
            # 출력 창 초기화
            # self.output_text.delete(1.0, tk.END)
            self.clear_all_outputs()
            self.append_to_output("Starting training...\n\n")
            
            # 별도 스레드에서 학습 시작
            self.start_training_process(train_command)
    
    def get_class_names(self, work_dir):
        """라벨 파일에서 클래스 이름 목록 가져오기"""
        class_ids = set()
        
        # 모든 txt 파일 검사
        for file in os.listdir(work_dir):
            if file.endswith('.txt'):
                with open(os.path.join(work_dir, file), 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
        
        # 클래스 ID를 이름으로 변환 (class0, class1, ...)
        return {i: f'class{i}' for i in sorted(class_ids)}

    def create_navigation_thread(self, event=None):
        """네비게이션 윈도우 생성을 위한 쓰레드 시작"""
        # 이미 네비게이션 윈도우가 열려있는지 확인
        if hasattr(self, 'nav_window') and self.nav_window.winfo_exists():
            # 이미 열려있다면 포커스만 이동
            self.nav_window.lift()
            self.nav_window.focus_force()
            return
            
        # 새로운 네비게이션 윈도우 쓰레드 생성
        nav_thread = threading.Thread(target=self.create_navigation_window)
        nav_thread.daemon = True  # 메인 윈도우가 종료되면 함께 종료
        nav_thread.start()

def main():
    root = TkinterDnD.Tk()
    app = ModernUIApp(root)
    root.geometry("1200x800")
    root.mainloop()

if __name__ == "__main__":
    main()