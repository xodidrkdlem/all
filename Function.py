import math     
import time     # 영상에 보이는fps를 타이머가 계산하기 위해 사용됨.
import numpy as np  # 배열 작업용
import cv2      # 비디오/이미지 처리용
import matplotlib.pyplot as plt

def denoise_frame(frame):
    kernel = np.ones((3, 3), np.float32) / 9
# np.ones(3,3) 배열의 모양으로 1로 초기화 인데 이걸 각각/9를 한다. 왜???
    print(kernel)                              
#[[0.11111111 0.11111111 0.11111111]
#  [0.11111111 0.11111111 0.11111111]
#  [0.11111111 0.11111111 0.11111111]]
    denoised_frame = cv2.filter2D(frame, -1, kernel)   # 프레임에 필터 적용.
#cv2.filter2D(src, ddepth, kernel, dst=None, anchor=None, delta=None, borderType=None) -> dst
# src : 입력 영상, ddepth : 출력 영상 데이터 타입. (e.g) cv2.CV_8U, cv2.CV_32F, cv2.CV_64F,
#  -1을 지정하면 src와 같은 타입의 dst 영상을 생성합니다.
 #kernel: 필터 마스크 행렬. 실수형.   
    return denoised_frame  #노이즈 제거된 프레임 반환

# --------------------------------------------------------------------------------

# <확실한 부분>
# 이 함수의 기능: 노이즈가 제거된 프레임을 반환하는 함수
#  위의 frame부분에  영상이 들어감.
# 영상은 사진의 연속. 

# <궁금한부분> 
#  np.ones()  : 왜 커널이라는 변수에서 3,3으로 배열 모양을 정했고, 실수로 썼으며, 
#             : 왜 1로 초기화를 하였고 그걸 왜 9로 나눴는지

# filter2D()  :
# (입력 영상, 출력 영상 데이터 타입(-1 입력시 입력영상과 같은 타입으로 생성), 필터 마스크 행렬)
# 필터 마스크 행렬 뭐지?
#  cv2.fliter2D 함수를 사용해서 평균 값 필터를 적용해보겠습니다.
#  평균값 필터는 영상의 특정 좌표 값을 주변 픽셀 값들의 산술 평균으로 설정하는 필터입니다.
#  평균값으로 바꾸게 되면 전체적으로 화면이 뿌옇게 된다.
#  픽셀들 간의 그레이스스케일 값 변화가 줄어들어 날카로운 에지가 무뎌지고, 영상에 있는 잡음의 영향이 사라지는 효과가 있습니다.


#<추측 가능한 부분>
# 위 함수에서 가우시안 블러를 활용해서 화면을 부드럽게 만들어 노이즈를 줄인듯 하다.
# 해결: 맞음.


#--------------------------------------------------------------------------------------


def detect_edges(frame):                   #가장자리 감지하는 함수
    """ Function for detecting edges       # Canny Edge Detection으로 프레임의 에지를 감지하는 기능
    on frame with Canny Edge Detection """ 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 프레임을 회색조로 변환
    canny_edges = cv2.Canny(gray, 50, 150)  # 1:3 (threshold 비율,th1,th2) canny 에지 감지 기능 적용
    
    return canny_edges  # 가장자리 프레임 반환



#----------------------------------------------------------------------------------------- 
#<확실한 부분>
#위 함수는 Canny Edge Detection으로 프레임의 에지를 감지하는 기능
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#영상을 그레이 스케일로 변환.
#그레이 스케일로 바뀐 영상(사진)은 흰색(255,255,255)과 검은색(0,0,0) 두개의 색깔가짐.
#canny_edges  = cv2.Canny(gray,50,150) //(이미지 또는 영상,임계값1, 임계값 2)
#canny 엣지가 엣지검출을 하기에 제일적합.
#가우시안 블러 등 사용자가 지정한 canny 사용보단 opencv의 canny가 더 정확함. 함수를 만들지말고. 있는걸 쓰자.


#임계값(Th1)의 기능 : 다른엣지와의 엣지가 되기 쉬운 인접 부분의 엣지 여부를 판단.
#Th1의 값을 작게하여도 원래 엣지가 아니라고 판단된 부분이 갑자기 검출되지는 않음.
#임계값(Th1)은 Th2에서 검출된 엣지를 보고, 엣지와 엣지 사이의 선을 길게 늘리는 역할.
#★결론: 값이 커지면 엣지 검출 down, 값이 작아지면 엣지 검출 up
#!!!!!경고: 너무 줄이면 모든 부분을 엣지와 엣지사이의 연장선이라고 생각하고 전부 연결. 

#임계값(TH2)의 기능: 엣지인지 아닌지를 판단.값을 작게 할수록 너 많은 엣지 검출.
#★결론: 값이 커지면 엣지 검출 down, 값이 작아지면 엣지 검출 up


#<추측한 부분>
#Canny를 통해 비율이 1:3이 엣지 검출을 위한 최고의 비율인가를 의심해볼 필요가 있음.
#해결 1:3 ,1:2가 최고 비율
#-----------------------------------------------------------------------------------------


def region_of_interest(frame):      #원래 프레임에 관심영역 그리는 함수!!
    """ Function for drawing region of    
    interest on original frame """     

    height, width = frame.shape   #높이, 너비 = 
    mask = np.zeros_like(frame)   #0이 가득찬 상태&&frame과 비슷한 크기의 배열을 만듬.  

    # 아래 화면의 아래쪽으로 향한 초점을 맞춥니다.
    polygon = np.array([[
        (int(width*0.30), height),              # 왼쪽 아래 좌표/화면 왼쪽에서 끝(그래서 세로 길이 안정한것.
        (int(width*0.46),  int(height*0.72)),   # 왼쪽 위에 좌표/이 좌표와 아래 좌표를 건들면 높낮이 조절가능&
        (int(width*0.58), int(height*0.72)),    # 오른쪽 위에 좌표/사다리꼴 모양의 윗가로 부분 길이 조절가능.
        (int(width*0.82), height),              # 오른쪽 아래 좌표/화면 오른쪽에서 끝 
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)             #다각형을 그리고 그범위만큼 채우는 함수
    roi = cv2.bitwise_and(frame, mask)           #둘다 0이 아닌 경우에만 값을 통과시킨다는 뜻.
    
    return roi     
    
#--------------------------------------------------------------------------------------
#<확실한 부분>
#cv2.fillPoly(mask, polygon, 255)             #다각형을 그리고 그범위만큼 채우는 함수
##mask = mat형태의 이미지 파일 지정// pol = 다각형 포인트 어레이// 다각형색상(0~255)//라인타입,shift
#다각형을 여려개 만들어서 겹치는 부분 지우는 코드 존재(주소창에.fillpoly찾으면됨.)
#
#
#
#
#
#<궁금한 부분>
# fillpoly에서 shift에 해당하는 부분 기능은 뭘까? 
# bitwise_and()함수의 역할이 잘 이해가 안감.
# 해결: 겹친 부분만 roi로 출력(아래범위 늘리고 줄이고 가능)


#--------------------------------------------------------------------------------------
def histogram(frame):                              
    """ Function for histogram                     #히스토그램의 기능. 
    projection to find leftx and rightx bases """  
 #leftx 및 rightx 염기를 찾기 위한 투영???
 #투영 : 입체적인 물체를 평면 상에 그리는 방식
 
    
    histogram = np.sum(frame, axis=0)   #히스토그램 빌딩. 
    midpoint = np.int(histogram.shape[0]/2)   # 히스토그램에서 중간점 HD 
    left_x_base = np.argmax(histogram[:midpoint])    # 왼쪽 최대 크기를 계산
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint   #오른쪽 최대 크기를 계산 

    return left_x_base, right_x_base

#-------------------------------------------------------------------------------------

#히스토그램이란
# 표로 되어 있는 도수 분포를 정보 그림으로 나타낸 것이다.
#  더 간단하게 말하면, 도수분포표를 그래프로 나타낸 것이다
#np.sum()은 전체 합계를 구하는 함수. 모든 원소의 합계를 구하는 함수
# import numpy as np

# a = np.array([[1, 0, 3],     ㅣㅣ ㅣ  ㅣ ㅣ  ㅣ ㅣ
#              [2, 5, 4],      ㅣㅣ ㅣ  ㅣ ㅣ  ㅣ ㅣ  axis 축이 0이면 (6, 11,14),아래방향
#              [3, 6, 7]])     ㅣV  ㅣ  ㅣ ㅣ  ㅣ ㅣ 

# np.sum(a) 
# 결과: 31 (a의 원소들의 합)

# 모든 원소의 합계가 아닌, 특정 축을 기준으로만 합계를 구하기 위해서는,
# axis!!!!!! 인자를 지정해주시면 됩니다.
# axis가 0이면 세로(열)의 원소들 더하기
# axis가 1이면 가로(행)의 원소들 더하기
# 기본적으로 keepdims 인자가 지정되지 않은 상황에서는,
# n차원 array를 input으로 넣으면 n-1차원 array가 결과로 반환됩니다
#
#
#<확실한 부분>
#argmax: 가장큰 원소의 인덱스
#
#
#<궁금한 부분>
#histogram의 쓰임새는 무엇인가?
#
#
#------------------------------------------------------------------------------------

def warp_perspective(frame):           #프레임을 뒤틀어서 스카이뷰 각도에서 처리하는 기능
    """ Function for warping the frame  # 프레임을 와핑하여 스카이뷰 각도에서 처리하는 기능
    to process it on skyview angle """   
    
    height, width = frame.shape    # 이미지 사이즈
    offset = 50     # Offset for frame ration saving
    
    # 원본이미지 좌표
    source_points = np.float32([[int(width*0.46), int(height*0.72)], # 왼쪽 위에 좌표 
                      [int(width*0.58), int(height*0.72)],           # 오른쪽 위에 좌표
                      [int(width*0.30), height],                     # 왼쪽 아래 좌표
                      [int(width*0.82), height]])                    # 오른쪽 아래 좌표
    
    # 변경할 이미지 위치 좌표
    destination_points = np.float32([[offset, 0],                   # 왼쪽 위에 좌표
                      [width-2*offset, 0],                  # 오른쪽 위에 좌표
                      [offset, height],                      # 왼쪽 아래 좌표
                      [width-2*offset, height]])     # 오른쪽 아래 좌표
    
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)    # 스카이뷰 창의 이미지를 반전시키는 메트리스
    skyview = cv2.warpPerspective(frame, matrix, (width, height))    # 최종적으로 반전

    return skyview  # 스카이뷰 프레임 반환.

    #-------------------------------------------------------------------------------------------

def detect_lines(frame):                    #라인 감지 기능 Hough Lines polar를 통한.
    """ Function for line detection 
    via Hough Lines Polar """
    
    line_segments = cv2.HoughLinesP(frame, 1, np.pi/180 , 20, 
                                    np.array([]), minLineLength=40, maxLineGap=150)

 # houghLines() 안쓰는 이유: 모든 점에대해서 계산,시간이 너무 소요됨, 그만큼, 메모리 펑
 # HoughLinesP( , , , , , , )
    
    
    return line_segments    # 도로의 라인 분할

    #------------------------------------------------------------------------------------
 #<확인한 부분>
 # houghLines() 안쓰는 이유: 모든 점에대해서 계산,시간이 너무 소요됨, 그만큼, 메모리 펑
 # HoughLinesP을 쓰는 이유: 임의의 점을 이용하여 직선을 찾는 것이기 때문.(단,임계값을 작게해야만한다.)
 #장점: 시작점과 끝점을 return해주기 때문에 쉽게 화면에 표현할수있음.
 #질문: 직선만 찾으면 곡선은??
 #
 # HoughLinesP( image,rho ,theta ,threshold ,minLineLength ,maxLineGap )
# image – 8bit, single-channel binary image, canny edge를 선 적용.
# rho – r 값의 범위 (0 ~ 1 실수)
# theta – 𝜃 값의 범위(0 ~ 180 정수)

# !!!!!!!!!!!!!!!!!!!!
# threshold – 만나는 점의 기준, 숫자가 작으면 많은 선이 검출되지만 정확도가 떨어지고, 숫자가 크면 정확도가 올라감.
# !!!!!!!!!!!!!!!!!!!!

# minLineLength – 선의 최소 길이. 이 값보다 작으면 reject. 
# maxLineGap – 선과 선사이의 최대 허용간격. 이 값보다 작으며 reject.
#
#<궁금한 부분 >
# minLineLength – 선의 최소 길이. 이 값보다 작으면 reject. 
# maxLineGap – 선과 선사이의 최대 허용간격. 이 값보다 작으며 reject.
#위에 두개 사용하면 곡선 검출 가능??
#. minLineLength는 최소 직선 길이를,
#  maxLineGap은 직선이 어느 정도 간격을 두고 떨어져 있어도 하나의 직선으로 볼 것인지를 결정한다.
#
#------------------------------------------------------------------------------


def optimize_lines(frame, lines):   #라인 최적화 기능 및 도로에 하나의 실선 출력
    """ Function for line optimization and 
    outputing one solid line on the road """    
    
    height, width, _ = frame.shape  # 프레임 크기 가져오기
    
    
    if lines is not None:   # 줄이 없으면 메모리에서 줄을 가져옵니다.
        # 줄 구분을 위한 변수 초기화
        lane_lines = [] # 두줄 모두
        left_fit = []   # 왼쪽 줄의 경우
        right_fit = []  # 오른쪽 라인의 경우
        
        for line in lines:  # 라인 범위의 각 라인에 접근
            x1, y1, x2, y2 = line.reshape(4)    # 실제 줄을 좌표로 풀기.

            parameters = np.polyfit((x1, x2), (y1, y2), 1)  # 얻은 포인트에서 매개변수 가져오기.
            slope = parameters[0]       # 목록 매개변수의 첫번째 매개변수는 기울기.
            intercept = parameters[1]   # 두번째는 절편
            
            if slope < 0:   # 여기에서 선의 기울기를 확인한다.
                left_fit.append((slope, intercept))    #왼쪽(기울기,절편)
            else:   
                right_fit.append((slope, intercept))   #오른쪽(기울기,절편)

        if len(left_fit) > 0:       # 여기에서 왼쪽 줄에 맞는지 여부를 확인합니다.
            left_fit_average = np.average(left_fit, axis=0)     # 왼쪽 라인의 평균 맞춤
            lane_lines.append(map_coordinates(frame, left_fit_average)) # 매핑된 포인트의 결과를목록에 추가 lane_line
            
        if len(right_fit) > 0:       # 여기에서 오른쪽 줄에 맞는 여부를 확인함
            right_fit_average = np.average(right_fit, axis=0)   # 오른쪽 선의 평균 맞춤.
            lane_lines.append(map_coordinates(frame, right_fit_average))    # 매핑된 포인트의 결과를 목록에 추가 lane_line
        
    return lane_lines       # 실제 감지되고 최적화된 라인을 반환.

def map_coordinates(frame, parameters): #라인 구성을 위해 주어진 매개변수를 매핑하는 기능
    """ Function for mapping given      #매핑:하나의 값을 다른 값으로 대응시키는 것을 말한다
    parameters for line construction """  
    
    height, width, _ = frame.shape  # frame 사이즈 가져오기.
    slope, intercept = parameters   # 주어진 매개변수에서 기울기 및 절편 풀기
    
    if slope == 0:      # 기울기가 0인지 확인
        slope = 0.1     # Divisiob by Zero 오류를 줄이기 위해 처리합니다.
    
    y1 = height             # 프레임의 하단 좌표
    y2 = int(height*0.72)  # 프레임 중앙에서 아래쪽으로 점을 찍습니다.  
    x1 = int((y1 - intercept) / slope)  # 공식 (y절편)/기울기로 x1계산하기.
    x2 = int((y2 - intercept) / slope)  # 공식 (y절편)/기울기로 x2계산하기.
    
    return [[x1, y1, x2, y2]]   # 좌표 배열 반환


#-----------------------------------------------------------------------------------
def display_lines(frame, lines):                  #원본 프레임에 라인 표시하는 기능.
    """ Function for displaying lines             
    on the original frame """
    
    mask = np.zeros_like(frame)   # 프레임과 같은 차원의 0으로 채워진 배열 만들기.
    if lines is not None:       # 기존 라인이 있는지 확인
        for line in lines:      # 줄 목록을 반복합니다.
            for x1, y1, x2, y2 in line: # 좌표로 라인 풀기
                cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 5)    # 생성된 마스크에 선을 그림.
    
    mask = cv2.addWeighted(frame, 0.8, mask, 1, 1)    #원본 프레임과 마스크 병합
    
    return mask

    #------------------------------------------------------------------------------
#<확인 가능한 부분>
#cv2.line(img,pt1,pt2,color,thickness(선 두께))
#pt1 : 시작점  좌표,  pt2:종료점 좌표
#cv2.addWeighted() - 가중치 합, 평균 연산.
#cv2.addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=None) -> dst
# • src1: (입력) 첫 번째 영상
# • alpha: 첫 번째 영상 가중치
# • src2: 두 번째 영상. src1과 같은 크기 & 같은 타입
# • beta: 두 번째 영상 가중치
# • gamma: 결과 영상에 추가적으로 더할 값
# • dst: 가중치 합 결과 영상
# • dtype: 출력 영상(dst)의 타입
#cv2.add(같은 위치에 존재하는 픽셀값을 더하여 결과 영상의 픽셀값으로 설정,덧셈결과 255이상일 경우 255으로)
#
#
#<궁금한 부분>
#line 변수는 어디에?
#
#
#
#------------------------------------------------------------------------------
def display_heading_line(frame, up_center, low_center):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    
    x1 = int(low_center)
    y1 = height
    x2 = int(up_center)
    y2 = int(height*0.72)
    
    cv2.line(heading_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    
    return heading_image

    #--------------------------------------------------------
    #<궁금한 부분>
    #display_heading_line 함수는 어디에, 왜, 어떻게 사용되는지.
    #
    #
    #--------------------------------------------------------
    def get_floating_center(frame, lane_lines):  #차선을 바라보며 도로에서 조향각을 계산하는 기능
    """ Function for calculating steering angle   
    on the road looking at the lanes on it """
    
    height, width, _ = frame.shape # 프레임사이즈가져오기
    
    if len(lane_lines) == 2:    # 여기에서 2개의 줄이 감지되었는지 확입합니다.
        left_x1, _, left_x2, _ = lane_lines[0][0]   # 왼쪽 줄 풀기
        right_x1, _, right_x2, _ = lane_lines[1][0] # 오른쪽 줄 풀기(unpaking)
        #해석의 오류일지도? 파이썬에서의 paking, unpaking  paking하는 법 인수앞에*
        low_mid = (right_x1 + left_x1) / 2  # Calculate the relative position of the lower middle point
        up_mid = (right_x2 + left_x2) / 2   #하단 중간 지점의 상대 위치 계산

    else:       # 감지되지않은 줄 처리.
        up_mid = int(width*1.9)
        low_mid = int(width*1.9)
    
    return up_mid, low_mid       # Return shifting points(이동 지점 반환)
#-------------------------------------------------------
#<궁금한 부분>
#lane_lines 변수가 어디에있는지.
#up_mid = int(width*1.9),low_mid = int(width*1.9)에서 왜 1.9를 곱했는지       
#
def add_text(frame, image_center, left_x_base, right_x_base):  #회전방향 화면상 텍스트로 출력
    """ Function for text outputing
    Output the direction of turn"""

    lane_center = left_x_base + (right_x_base - left_x_base) / 2 # 두 라인 사이에서 차선 중심 찾기
    
    deviation = image_center - lane_center    # 편차 찾기.

    if deviation > 160:         # 편차에 따른 예측 회전
        text = "Smooth Left"
        memory_text = text
    elif deviation < 40 or deviation > 150 and deviation <= 160:
        text = "Smooth Right"
        memory_text = text
    elif deviation >= 40 and deviation <= 150:
        text = "Straight"
        memory_text = text
    else:
        text = memory_text
    
    cv2.putText(frame, "DIRECTION: " + text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) # Draw direction
    #cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    #        영상의 경우 프레임, 텍스트,텍스트 좌표, 폰트,폰트 크기,폰트 색상,라인 타입,
    # 2는 뭘 뜻하는건지 모름.
    return frame    # 방향이 있는 프레임 return
    #------------------------------------------------------------------------------
#<확인 가능한 부분>
#cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
#             이미지,텍스트,좌표,폰트.폰트크기,색상,두께,라인타입
#deviation의 값을 조정하면 뭐가 달라지는지
#
#<추측가능한 부분>
#deviation을 조정하면 기울기에 따른 방향 인식이 달라질수도 있다.


