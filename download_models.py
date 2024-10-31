import os
import requests
from tqdm import tqdm

def download_landmarks_model():
    """dlib의 얼굴 랜드마크 모델을 다운로드하는 함수"""
    
    # 모델 URL
    MODEL_URL = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    # 저장할 파일 이름
    MODEL_FILE = "shape_predictor_68_face_landmarks.dat.bz2"
    EXTRACTED_FILE = "shape_predictor_68_face_landmarks.dat"
    
    # 이미 파일이 존재하는지 확인
    if os.path.exists(EXTRACTED_FILE):
        print(f"{EXTRACTED_FILE}이 이미 존재합니다.")
        return
    
    print("랜드마크 모델 다운로드 중...")
    
    # 파일 다운로드
    response = requests.get(MODEL_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(MODEL_FILE, 'wb') as file, tqdm(
        desc=MODEL_FILE,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)
    
    # bz2 압축 해제
    import bz2
    print("압축 해제 중...")
    with bz2.BZ2File(MODEL_FILE) as fr, open(EXTRACTED_FILE, 'wb') as fw:
        data = fr.read()
        fw.write(data)
    
    # 압축 파일 삭제
    os.remove(MODEL_FILE)
    print(f"다운로드 완료: {EXTRACTED_FILE}")

if __name__ == "__main__":
    download_landmarks_model()
