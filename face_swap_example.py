# face_swap_example.py
import cv2
from face_swap import face_swap

def test_face_swap(source_path, target_path, result_path):
    """
    얼굴 합성을 테스트하는 함수
    
    Parameters:
        source_path: 소스 이미지 경로 (합성할 얼굴)
        target_path: 타겟 이미지 경로 (얼굴이 합성될 대상)
        result_path: 결과 이미지 저장 경로
    """
    # 이미지 로드
    print("이미지를 로드하는 중...")
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    
    if source_img is None:
        raise ValueError(f"소스 이미지를 찾을 수 없습니다: {source_path}")
    if target_img is None:
        raise ValueError(f"타겟 이미지를 찾을 수 없습니다: {target_path}")
    
    # 이미지 크기 출력
    print(f"소스 이미지 크기: {source_img.shape}")
    print(f"타겟 이미지 크기: {target_img.shape}")
    
    # 얼굴 합성 수행
    print("얼굴 합성을 시작합니다...")
    result = face_swap(source_img, target_img)
    
    if result is not None:
        # 결과 저장
        cv2.imwrite(result_path, result)
        print(f"결과가 저장되었습니다: {result_path}")
        
        # 결과 보여주기
        cv2.imshow('Source', source_img)
        cv2.imshow('Target', target_img)
        cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("얼굴을 찾을 수 없습니다. 다른 이미지를 시도해보세요.")

if __name__ == "__main__":
    # 예시 실행
    source_path = "source.png"  # 합성할 얼굴 이미지
    target_path = "target.png"  # 얼굴이 합성될 대상 이미지
    result_path = "result.png"  # 결과 이미지가 저장될 경로
    
    try:
        test_face_swap(source_path, target_path, result_path)
    except Exception as e:
        print(f"에러가 발생했습니다: {str(e)}")
