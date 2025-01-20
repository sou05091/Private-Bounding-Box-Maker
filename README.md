# Private Bounding Box Maker

## 개요
Private Bounding Box Maker는 YOLO 형식의 객체 검출을 위한 바운딩 박스 라벨링 도구입니다. TkinterDnD를 활용하여 드래그 앤 드롭 기능을 지원하며, 직관적인 인터페이스로 이미지에 바운딩 박스를 쉽게 생성할 수 있습니다.

## 주요 기능
- 드래그 앤 드롭으로 이미지 파일 로드
- 마우스로 바운딩 박스 그리기
- YOLO 형식(.txt)으로 라벨 데이터 저장
- 이미지 확대/축소 기능
- 바운딩 박스 편집 및 삭제
- 이전/다음 이미지 탐색

## 설치 방법
```bash
# 필요한 패키지 설치
pip install tkinterdnd2
pip install Pillow
```

## 사용 방법
1. 프로그램 실행
```bash
python main.py
```

2. 이미지 로드
   - 드래그 앤 드롭으로 이미지 파일 또는 폴더 추가
   - File 메뉴에서 이미지 파일 또는 폴더 선택

3. 바운딩 박스 생성
   - 마우스 왼쪽 버튼으로 드래그하여 박스 그리기
   - 박스 생성 후 클래스 선택

4. 바운딩 박스 편집
   - 박스 선택 후 이동 또는 크기 조절
   - Delete 키로 선택된 박스 삭제

5. 저장
   - 자동 저장 기능 지원
   - 각 이미지에 대해 YOLO 형식의 .txt 파일 생성

## 단축키
- `→` : 다음 이미지
- `←` : 이전 이미지
- `+` : 확대
- `-` : 축소
- `Del` : 선택된 박스 삭제
- `Ctrl+S` : 수동 저장

## 출력 형식
YOLO 형식의 텍스트 파일 (.txt)
```
<class> <x_center> <y_center> <width> <height>
```
- 모든 값은 이미지 크기로 정규화 (0~1)
- 클래스는 0부터 시작하는 정수

## 시스템 요구사항
- Python 3.6 이상
- tkinterdnd2
- Pillow

## 라이선스
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 기여 방법
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
