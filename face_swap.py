import cv2
import numpy as np
import dlib
from collections import OrderedDict

# 얼굴의 랜드마크 포인트 순서 정의
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

def get_landmarks(detector, predictor, img):
    """이미지에서 얼굴 랜드마크를 추출하는 함수"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    if len(rects) == 0:
        return None
    
    shape = predictor(gray, rects[0])
    coords = np.zeros((68, 2), dtype="int")
    
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    
    return coords

def get_face_mask(img, landmarks):
    """얼굴 영역의 마스크를 생성하는 함수"""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    points = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, points, 255)
    return mask

def find_delaunay_triangles(img, landmarks, mask):
    """들로네 삼각분할을 수행하는 함수"""
    # 이미지 영역 설정
    rect = (0, 0, img.shape[1], img.shape[0])
    
    # 삼각분할을 위한 subdivison 객체 생성
    subdiv = cv2.Subdiv2D(rect)
    
    # 점들을 float32 타입의 튜플로 변환
    points = [(float(x), float(y)) for x, y in landmarks]
    
    # 포인트 삽입
    for point in points:
        try:
            subdiv.insert((point[0], point[1]))
        except Exception as e:
            print(f"포인트 삽입 중 에러 발생: {point}")
            continue
    
    # 삼각형 리스트 얻기
    triangles = subdiv.getTriangleList()
    
    # 결과 삼각형 인덱스 리스트
    triangles_idx = []
    
    # 각 삼각형에 대해
    for t in triangles:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        
        # 마스크 내부의 삼각형만 선택
        if (mask[pt1[1], pt1[0]] > 0 and 
            mask[pt2[1], pt2[0]] > 0 and 
            mask[pt3[1], pt3[0]] > 0):
            
            # 각 점에 대한 인덱스 찾기
            idx1 = -1
            idx2 = -1
            idx3 = -1
            
            # 랜드마크에서 해당하는 점 찾기
            for i, point in enumerate(landmarks):
                if (int(point[0]) == int(t[0]) and 
                    int(point[1]) == int(t[1])):
                    idx1 = i
                elif (int(point[0]) == int(t[2]) and 
                      int(point[1]) == int(t[3])):
                    idx2 = i
                elif (int(point[0]) == int(t[4]) and 
                      int(point[1]) == int(t[5])):
                    idx3 = i
            
            if idx1 != -1 and idx2 != -1 and idx3 != -1:
                triangles_idx.append([idx1, idx2, idx3])
    
    return triangles_idx

def warp_triangle(src, dst, src_tri, dst_tri):
    """삼각형 영역을 워핑하는 함수"""
    # numpy array로 변환하고 float32 타입으로 설정
    src_tri = np.float32(src_tri)
    dst_tri = np.float32(dst_tri)
    
    # 대상 삼각형의 경계 상자 계산
    r = cv2.boundingRect(dst_tri)
    (x, y, w, h) = r
    
    # 경계 상자 기준으로 좌표 변환
    dst_tri_cropped = []
    src_tri_cropped = []
    
    for i in range(3):
        dst_tri_cropped.append([dst_tri[i][0] - x, dst_tri[i][1] - y])
        src_tri_cropped.append(src_tri[i])  # 소스 좌표는 그대로 사용
    
    dst_tri_cropped = np.float32(dst_tri_cropped)
    src_tri_cropped = np.float32(src_tri_cropped)
    
    # 타겟 크기의 빈 이미지 생성
    dst_crop = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 변환 행렬 계산 및 적용
    warp_mat = cv2.getAffineTransform(src_tri_cropped, dst_tri_cropped)
    try:
        cv2.warpAffine(src, warp_mat, (w, h), dst_crop, 
                       flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
    except Exception as e:
        print(f"워핑 중 에러 발생: {str(e)}")
        print(f"크기 정보 - w: {w}, h: {h}")
        return
    
    # 마스크 생성
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_cropped), 1.0, 16, 0)
    
    # 가우시안 블러로 마스크 경계 부드럽게
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # 마스크 적용
    dst_crop = dst_crop.astype(np.float32) / 255.0
    warped_triangle = (dst_crop * mask[:, :, np.newaxis]) * 255.0
    
    # 대상 영역에 결과 합성
    try:
        dst[y:y+h, x:x+w] = dst[y:y+h, x:x+w] * (1 - mask[:, :, np.newaxis])
        dst[y:y+h, x:x+w] = dst[y:y+h, x:x+w] + warped_triangle
    except Exception as e:
        print(f"합성 중 에러 발생: {str(e)}")
        print(f"좌표 정보 - x: {x}, y: {y}, w: {w}, h: {h}")
        return

def visualize_landmarks(img, landmarks, save_path=None):
    """얼굴 랜드마크를 시각화하는 함수"""
    # 이미지 복사
    vis_img = img.copy()
    
    # 모든 랜드마크 포인트 그리기
    for i, (x, y) in enumerate(landmarks):
        # 포인트 그리기
        cv2.circle(vis_img, (int(x), int(y)), 2, (0, 255, 0), -1)
        # 포인트 번호 표시
        cv2.putText(vis_img, str(i), (int(x) + 2, int(y) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # 각 부위별로 다른 색상으로 연결선 그리기
    colors = {
        "jaw": (255, 0, 0),      # 파랑
        "right_eyebrow": (0, 255, 0),   # 초록
        "left_eyebrow": (0, 255, 0),    # 초록
        "nose": (0, 255, 255),    # 노랑
        "right_eye": (255, 0, 255),     # 분홍
        "left_eye": (255, 0, 255),      # 분홍
        "mouth": (0, 0, 255)      # 빨강
    }
    
    # 각 부위별로 선 그리기
    for facial_feature, color in colors.items():
        start, end = FACIAL_LANDMARKS_68_IDXS[facial_feature]
        points = landmarks[start:end]
        for i in range(len(points) - 1):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
            cv2.line(vis_img, pt1, pt2, color, 1)
        
        # jaw를 제외하고는 마지막 점과 첫 점을 연결 (눈, 눈썹, 입)
        if facial_feature != "jaw":
            pt1 = (int(points[-1][0]), int(points[-1][1]))
            pt2 = (int(points[0][0]), int(points[0][1]))
            cv2.line(vis_img, pt1, pt2, color, 1)
    
    # 결과 저장
    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"랜드마크 시각화 이미지가 저장되었습니다: {save_path}")
    
    return vis_img

def face_swap(source_img, target_img):
    """얼굴 합성을 수행하는 메인 함수"""
    # dlib의 얼굴 검출기와 랜드마크 예측기 초기화
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # 얼굴 랜드마크 검출
    source_landmarks = get_landmarks(detector, predictor, source_img)
    target_landmarks = get_landmarks(detector, predictor, target_img)
    
    if source_landmarks is None or target_landmarks is None:
        print("얼굴을 찾을 수 없습니다!")
        return None
    
    # 랜드마크 시각화 및 저장
    source_vis = visualize_landmarks(source_img, source_landmarks, "source_landmarks.png")
    target_vis = visualize_landmarks(target_img, target_landmarks, "target_landmarks.png")
    
    # 얼굴 마스크 생성
    source_mask = get_face_mask(source_img, source_landmarks)
    target_mask = get_face_mask(target_img, target_landmarks)
    
    # 디버깅을 위해 마스크도 저장
    cv2.imwrite("source_mask.png", source_mask)
    cv2.imwrite("target_mask.png", target_mask)
    
    # 들로네 삼각분할
    triangles_idx = find_delaunay_triangles(target_img, target_landmarks, target_mask)
    
    if not triangles_idx:
        print("삼각분할에 실패했습니다!")
        return None
    
    # 결과 이미지 초기화
    result_img = np.float32(target_img.copy())
    
    # 삼각형 시각화를 위한 이미지
    triangulation_img = np.copy(target_img)
    
    # 각 삼각형 영역에 대해 워핑 수행
    for i, triangle in enumerate(triangles_idx):
        # 소스 이미지의 삼각형 좌표
        src_tri = []
        dst_tri = []
        
        for idx in triangle:
            src_tri.append(source_landmarks[idx])
            dst_tri.append(target_landmarks[idx])
            
        src_tri = np.array(src_tri, dtype=np.float32)
        dst_tri = np.array(dst_tri, dtype=np.float32)
        
        # 삼각형 시각화
        cv2.line(triangulation_img, 
                 tuple(map(int, dst_tri[0])), 
                 tuple(map(int, dst_tri[1])), 
                 (0, 255, 0), 1)
        cv2.line(triangulation_img, 
                 tuple(map(int, dst_tri[1])), 
                 tuple(map(int, dst_tri[2])), 
                 (0, 255, 0), 1)
        cv2.line(triangulation_img, 
                 tuple(map(int, dst_tri[2])), 
                 tuple(map(int, dst_tri[0])), 
                 (0, 255, 0), 1)
        
        # 삼각형 워핑
        warp_triangle(source_img, result_img, src_tri, dst_tri)
    
    # 삼각분할 시각화 저장
    cv2.imwrite("triangulation.png", triangulation_img)
    
    # 워핑 결과를 uint8로 변환
    result_warped = result_img.astype(np.uint8)
    cv2.imwrite("warped_before_blend.png", result_warped)
    
    # 얼굴 마스크 생성 및 중심점 계산
    face_mask = get_face_mask(result_warped, target_landmarks)
    
    # 마스크의 중심점 계산
    moments = cv2.moments(face_mask)
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    center = (center_x, center_y)
    
    # 마스크 확장 (더 자연스러운 블렌딩을 위해)
    kernel = np.ones((9,9), np.uint8)
    face_mask = cv2.dilate(face_mask, kernel, iterations=1)
    
    # seamless cloning 적용 (여러 모드 시도)
    try:
        # 일반 클로닝
        result_normal = cv2.seamlessClone(
            result_warped, target_img, face_mask, center, cv2.NORMAL_CLONE
        )
        cv2.imwrite("result_normal_clone.png", result_normal)
        
        # 혼합 클로닝
        result_mixed = cv2.seamlessClone(
            result_warped, target_img, face_mask, center, cv2.MIXED_CLONE
        )
        cv2.imwrite("result_mixed_clone.png", result_mixed)
        
        # 모노크롬 전송
        result_monochrome = cv2.seamlessClone(
            result_warped, target_img, face_mask, center, cv2.MONOCHROME_TRANSFER
        )
        cv2.imwrite("result_monochrome_clone.png", result_monochrome)
        
        # 기본적으로 MIXED_CLONE 결과 반환
        final_result = result_mixed
        
    except Exception as e:
        print(f"Seamless cloning 중 에러 발생: {str(e)}")
        print("기본 블렌딩으로 대체합니다.")
        
        # 에러 발생 시 기본 블렌딩 사용
        result_mask = cv2.GaussianBlur(face_mask, (5, 5), 0)
        result_mask = result_mask.astype(float) / 255.0
        final_result = (result_warped * result_mask[:, :, np.newaxis] + 
                       target_img * (1 - result_mask[:, :, np.newaxis]))
    
    # 디버그 정보 출력
    print("\n처리 과정:")
    print("1. 랜드마크 검출 완료")
    print(f"2. 소스 이미지에서 {len(source_landmarks)} 개의 포인트 검출")
    print(f"3. 타겟 이미지에서 {len(target_landmarks)} 개의 포인트 검출")
    print(f"4. {len(triangles_idx)} 개의 삼각형 생성")
    print("5. 워핑 완료")
    print("6. Seamless Cloning 적용")
    
    return final_result

if __name__ == "__main__":
    # 이미지 로드
    source_img = cv2.imread("source4.jpg")
    target_img = cv2.imread("target2.png")
    
    if source_img is None or target_img is None:
        print("이미지를 로드할 수 없습니다!")
        exit()
    
    print("이미지 크기:")
    print(f"Source: {source_img.shape}")
    print(f"Target: {target_img.shape}")
    
    # 얼굴 합성 수행
    result = face_swap(source_img, target_img)
    
    if result is not None:
        # 결과 저장
        cv2.imwrite("result.png", result)
        print("\n결과가 저장되었습니다.")
        
        # 모든 결과 보여주기
        cv2.imshow('1. Source Image', source_img)
        cv2.imshow('2. Target Image', target_img)
        cv2.imshow('3. Normal Clone', cv2.imread("result_normal_clone.png"))
        cv2.imshow('4. Mixed Clone', cv2.imread("result_mixed_clone.png"))
        cv2.imshow('5. Monochrome Clone', cv2.imread("result_monochrome_clone.png"))
        cv2.imshow('6. Final Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("얼굴 합성에 실패했습니다!")