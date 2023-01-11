# Import libraries
import math     # For math functions
import time     # For timer to calculate fps
import numpy as np  # For array operations
import cv2      # For video/image processing
import matplotlib.pyplot as plt
prev_angle = []

# 색상 검출
def color_hsv(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_yellow = np.array([24,20,20])          # 노랑색 범위
    upper_yellow = np.array([79,255,255])          

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res

# 노이즈 제거
def denoise_frame(frame):
    """ Function for denoising 
    image with Gaussian Blur """   
    
    # kernel = np.ones((5, 5), np.float32) / 9   # We used 3x3 kernel
    # denoised_frame = cv2.filter2D(frame, -1, kernel)   # Applying filter on frame
    denoised_frame = cv2.GaussianBlur(frame, (7,7),0)   # Applying filter on frame
    
    return denoised_frame   # Return denoised frame

# 에지 검출
def detect_edges(frame):
    """ Function for detecting edges 
    on frame with Canny Edge Detection """ 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    # canny_edges = cv2.Canny(gray, 50, 150)  # Apply Canny edge detection function with thresh ratio 1:3
    canny_edges = cv2.Canny(gray, 100, 300)  # Apply Canny edge detection function with thresh ratio 1:3
    # canny_edges = cv2.Canny(gray, 70, 140)
    return canny_edges  # Return edged frame

# 차선 검출할 영역 지정
def region_of_interest(frame):
    """ Function for drawing region of 
    interest on original frame """
    
    height, width = frame.shape
    mask = np.zeros_like(frame)
    # only focus lower half of the screen
    polygon = np.array([[
        (int(width*0.0), height),              # Bottom-left point # 왼쪽 아래
        (int(width*0.0),  int(height*0.10)),   # Top-left point # 왼쪽 위
        (int(width), int(height*0.10)),    # Top-right point # 오른쪽 위
        (int(width), height),              # Bottom-right point # 오른쪽 아래
    ]], np.int32)
    
    # 위에 지정한 크기의 다각형 내면을 255로 채우기
    cv2.fillPoly(mask, polygon, 255)
    # bitwise_and 함수로 255인 부분만 출력
    roi = cv2.bitwise_and(frame, mask)
    
    return roi

# 히스토그램 이용해서 x축 왼쪽, 오른쪽 베이스 좌표 찾기
def histogram(frame):
    """ Function for histogram 
    projection to find leftx and rightx bases """
    
    # np.sum() 참고: https://jimmy-ai.tistory.com/116
    histogram = np.sum(frame, axis=0)   # Build histogram
    midpoint = np.int(histogram.shape[0]/2)     # Find mid point on histogram
    # np.argmax(): 가장 큰 원소 인덱스 반환
    left_x_base = np.argmax(histogram[:midpoint])    # Compute the left max pixels  
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint    # Compute the right max pixels

    # plt.plot(histogram)
    # plt.show() 
    return left_x_base, right_x_base

# 스카이 뷰 각도로 처리
def warp_perspective(frame):
    """ Function for warping the frame 
    to process it on skyview angle """
    
    height, width = frame.shape    # Get image size
    offset = 200     # Offset for frame ration saving

    # Perspective points to be warped # 원본 이미지 좌표
    source_points = np.float32([[int(width*0.20), int(height*0.50)], # Top-left point
                      [int(width*0.90), int(height*0.50)],           # Top-right point
                      [int(width*0.0), height],                     # Bottom-left point
                      [int(width), height]])                    # Bottom-right point
    
    # Window to be shown # 변경할 이미지 좌표
    destination_points = np.float32([[offset, 0],           # Top-left point
                      [width-2*offset, 0],                  # Top-right point
                      [offset, height],                     # Bottom-left point
                      [width-2*offset, height]])     # Bottom-right point
    
    # getPerspectiveTransform(원본 이미지 좌표, 변경할 이미지 좌표)
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)    # Matrix to warp the image for skyview window
    skyview = cv2.warpPerspective(frame, matrix, (width, height))    # Final warping perspective 

    return skyview, matrix  # Return skyview frame

# 라인 감지
def detect_lines(frame):
    """ Function for line detection
    via Hough Lines Polar """
    
    # 직선(차선) 검출
    # HoughLinesP() 참고: https://deep-learning-study.tistory.com/212
    line_segments = cv2.HoughLinesP(frame, 1, np.pi/180 , 90, 
                                minLineLength=150, maxLineGap=10)
    # HoughLinesP(검출 이미지, 거리, 각도, 임계값, 최소 선 길이, 최대 선 간격)
    return line_segments    # Return line segment on road


# 라인 최적화. 도로에 하나의 실선 출력
def optimize_lines(frame, lines):
    """ Function for line optimization and 
    outputing one solid line on the road """

    if lines is not None:   # If there no lines we take line in memory
        # Initializing variables for line distinguishing
        lane_lines = [] # For both lines
        left_fit = []   # For left line
        right_fit = []  # For right line
        
        for line in lines:  # Access each line in lines scope
            x1, y1, x2, y2 = line.reshape(4)    # Unpack actual line by coordinates
            # print('x1, y1, x2, y2: ',x1, y1, x2, y2)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Take parameters from points gained
            slope = parameters[0]       # First parameter in the list parameters is slope
            
            intercept = parameters[1]   # Second is intercept

            if slope < 0:   # Here we check the slope of the lines 
                left_fit.append((slope, intercept))
            else:   
                right_fit.append((slope, intercept))

        if len(left_fit) > 0:       # Here we ckeck whether fit for the left line is valid
            left_fit_average = np.average(left_fit, axis=0)     # Averaging fits for the left line
            lane_lines.append(map_coordinates(frame, left_fit_average)) # Add result of mapped points to the list lane_lines
            
        if len(right_fit) > 0:       # Here we ckeck whether fit for the right line is valid
            right_fit_average = np.average(right_fit, axis=0)   # Averaging fits for the right line
            lane_lines.append(map_coordinates(frame, right_fit_average))    # Add result of mapped points to the list lane_lines
        
    else:
        lane_lines = None
    # print('lane_lines:', lane_lines)
    return lane_lines       # Return actual detected and optimized line 


def map_coordinates(frame, parameters):
    """ Function for mapping given 
    parameters for line construction """
    
    height, width, _ = frame.shape  # Take frame size
    slope, intercept = parameters   # Unpack slope and intercept from the given parameters
    
    if slope == 0:      # Check whether the slope is 0
        slope = 0.1     # handle it for reducing Divisiob by Zero error
    
    y1 = height             # Point bottom of the frame
    y2 = int(height*0.10)  # Make point from middle of the frame down  
    x1 = int((y1 - intercept) / slope)  # Calculate x1 by the formula (y-intercept)/slope
    x2 = int((y2 - intercept) / slope)  # Calculate x2 by the formula (y-intercept)/slope
    # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
    # print('parameters : ', parameters)
    
    return [[x1, y1, x2, y2]]   # Return point as array

# 화면에 라인 표시
def display_lines(frame, lines):
    """ Function for displaying lines 
    on the original frame """
    height, width, _ = frame.shape
    # image center line
    cv2.line(frame, ((width//2), height), ((width//2), (height//2)), (255,0,0), 2)
 
    # cv2.rectangle(frame, (int(width*0.2), 100), (500,500), (255,0,0),-1)

    mask = np.zeros_like(frame)   # Create array with zeros using the same dimension as frame
    if lines is not None:       # Check if there is a existing line
        for line in lines:      # Iterate through lines list
            for x1, y1, x2, y2 in line: # Unpack line by coordinates
                # print('x1, y1, x2, y2 : ', x1, y1, x2, y2)
                # 선 감지 안될 때
                if x1<-10000 or y1<-10000 or x2<-10000 or y2<-10000 or x1>10000 or y1>10000 or x2>10000 or y2>10000:
                    # cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    print('fail_line: ', line)
                    pass
                else:
                    cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 2)    # Draw the line on the created mask
                    # cv2.line(mask, (int(line[0][0]), int(line[0][1])), (int(line[0][2]), int(line[0][3])), (0, 255, 0), 2)    # Draw the line on the created mask
                    # for i in lines:
                        # cv2.line(mask, (int(i[0][0]), int(i[0][1])), (int(i[0][2]), int(i[0][3])), (0, 255, 0), 2)
    else:
        mask = cv2.addWeighted(frame, 0.8, mask, 1, 1) 
    # 원본 프레임과 마스크 병합
    mask = cv2.addWeighted(frame, 0.8, mask, 1, 1)    # Merge mask with original frame
    
    return mask

def direction(frame, lane_lines):
    height, width, _ = frame.shape
    cv2.rectangle(frame, (int(width*0.45), int(height*0.2)), (int(width*0.55), int(height*0.3)), (255,0,0),-1)
    
    print('lane_lines: ', lane_lines)

    left_a1 = int(width*0.45)
    right_a2 = int(width*0.55)

    text = 'Stop'
    
    if len(lane_lines) == 2: 
        left_x1, _, left_x2, _ = lane_lines[0][0]   # Unpacking left line
        right_x1, _, right_x2, _ = lane_lines[1][0] # Unpacking right line

        if left_x2 > left_a1:    # right
            print('right')
            text = 'Right'

        elif right_x2 < right_a2:    # left
            print('left')
            text = 'Left'

        else:
            text = 'Straight'
    elif len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]   # 한 라인만 나올때 left인지 right인지 구분
        if x1 < x2:    # right
            if x2 > left_a1:
                print('right')
                text = 'Right'
            else:
                text = 'Straight'
        else:          # left
            if x2 < right_a2:
                print('left')
                text = 'Left'
            else:
                text = 'Straight'

    else:
        text = 'Stop'

    cv2.putText(frame, text, (int(width*0.20), int(height*0.30)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return text

# 조향각 계산하기 위해 센터 좌표 구하기
def get_floating_center(frame, lane_lines):
    """ Function for calculating steering angle 
    on the road looking at the lanes on it """
    height, width, _ = frame.shape # Take frame size
    
    if len(lane_lines) == 2:    # Here we check if there is 2 lines detected
        left_x1, _, left_x2, _ = lane_lines[0][0]   # Unpacking left line
        right_x1, _, right_x2, _ = lane_lines[1][0] # Unpacking right line
        
        # 중간선에 위,아래 좌표..?
        low_mid = (right_x1 + left_x1) / 2  # Calculate the relative position of the lower middle point
        up_mid = (right_x2 + left_x2) / 2
    else:
        up_mid = int(width*1.9)
        low_mid = int(width*1.9)

    return up_mid, low_mid       # Return shifting points

# 화면에 중앙선 표시
def display_heading_line(frame, up_center, low_center):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    
    x1 = int(low_center)
    y1 = height
    x2 = int(up_center)
    y2 = int(height*0.20)
    
    # print('low_center: ', low_center)
    # print('up_center: ', up_center)
    # 선 검출 안될 때
    if x1<0 or x2<0 or x1>10000 or x2>10000:
        heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
        pass
    else:
        cv2.line(heading_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    
    return heading_image

# 구한 각도 값을 내가 원하는 범위 값으로 재구성?하기
def Map(angle, angle_min, angle_max, output_min, output_max):
    angle = (angle-angle_min)*(output_max-output_min)/(angle_max-angle_min)+output_min
    # print('angle: ', round(angle, 1))
    
    return angle


# 각도 구하고 회전 방향 출력
def add_text(frame, up_center, low_center,lane_lines, left_x_base, right_x_base):
    """ Function for text outputing     
    Output the direction of turn"""
    height, width, _ = frame.shape
    
    left_x1, _, left_x2, _ = lane_lines[0][0] 
    lane_center = left_x_base + (right_x_base - left_x_base) / 2 # Find lane center between two lines
    # image center 와 line center 사이 각도 구하기
    # np.arctan 참고: https://velog.io/@dldndyd01/OpenCV-%EC%9D%B4%EB%AF%B8%EC%A7%80-%ED%9A%8C%EC%A0%84%EC%A4%91%EC%95%99-%EA%B8%B0%EC%A4%80-%ED%9A%8C%EC%A0%84-cv2.getRotationMatrix2D-%EC%96%BC%EA%B5%B4-%ED%9A%8C%EC%A0%84-%EA%B0%81%EB%8F%84-%EA%B5%AC%ED%95%98%EA%B8%B0
    
    if up_center < 1000:
        if up_center > low_center:
            tan_theta = ((up_center-width/2)-(up_center-low_center))/(height/2)
        elif up_center < low_center:
            tan_theta = ((low_center-width/2)-(low_center-up_center))/(height/2)
        else :
            tan_theta = 0
    else :
        tan_theta = ((width/2-left_x1)-(left_x2-left_x1))/(height/2)
    theta = np.arctan(tan_theta)
    rotate_angle = theta *180//math.pi
    # print('rotate_angle', rotate_angle)

    # print('left_x_base', left_x_base)
    # print('right_x_base', right_x_base)

    if rotate_angle >= -10 and rotate_angle <= 23:
        text = "Straight"
        memory_text = text
    elif rotate_angle < -10:
        text = "Smooth Left"
        memory_text = text
    elif rotate_angle > 23:
        text = "Smooth Right"
        memory_text = text
    else:
        text = memory_text
    
    cv2.putText(frame, text, (int(width*0.10), int(height*0.20)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # Draw direction
    
    return frame, rotate_angle    # Retrun frame with the direction on it

# 색상 검출
def Color(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_yellow = np.array([24,20,20])          # 노랑색 범위
    upper_yellow = np.array([79,255,255])          

    lower_green = np.array([30, 60, 110])        # 초록색 범위
    upper_green = np.array([120, 255, 255])

    lower_red = np.array([0, 255, 100])        # 빨강색 범위
    upper_red = np.array([180, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)     # 110<->150 Hue(색상) 영역을 지정.
    mask1 = cv2.inRange(hsv, lower_green, upper_green)  # 영역 이하는 모두 날림 검정. 그 이상은 모두 흰색 두개로 Mask를 씌움.
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res1 = cv2.bitwise_and(frame, frame, mask=mask1)    # 흰색 영역에 초록색 마스크를 씌워줌.
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)

def weight_moving_average(w1,w2,w3,w4):   #w1= 0.5 //  w2 = 0.3 // w3 = 0.15 // w4 = 0.05
    global prev_angle
    if len(prev_angle) <4:
        WMA_angle = prev_angle[0]
        # pass
    elif len(prev_angle)>=4:    
        WMA_angle = prev_angle[0]*w4+prev_angle[1]*w3+prev_angle[2]*w2+prev_angle[3]*w1 
        
        prev_angle.pop(0)
        # print('WMA_angle: ',WMA_angle)
    return WMA_angle

# 실행
def test4_Import_lib(frame):
    """ Main orchestrator that defines
    whole process of program execution"""

    # ----------------함수 진행 순서----------------

    # 노이즈 제거 -> 에지 검출 -> 차선 검출할 영역 지정 -> 차선(직선) 감지 ->
    # 감지된 라인 최적화(하나의 실선으로) -> 양쪽 line 화면에 그리기 ->
    # line_center 위, 아래 좌표 구하기 -> line_center 화면에 그리기 ->
    # 각도 구하고, 방향 텍스트 화면에 띄우고, 최종 imshow하기 위한 프레임 return ->
    # 구한 각도 값을 내가 원하는 범위 값으로 재구성?하기

    hsv_color = color_hsv(frame)
    denoised_frame = denoise_frame(hsv_color)   # Denoise frame from artifacts
    # cv2.imshow('hsv_color', hsv_color)

    canny_edges = detect_edges(denoised_frame)  # Find edges on the frame

    roi_frame = region_of_interest(canny_edges)   # Draw region of interest
    cv2.imshow('roi_frame',roi_frame)
    # angle값이 아닌 deviation값을 return 할 경우
    warped_frame, minverce = warp_perspective(canny_edges)    # Warp the original frame, make it skyview

    ret, thresh = cv2.threshold(warped_frame, 160, 255, cv2.THRESH_BINARY)

    left_x_base, right_x_base = histogram(thresh)         # Take x bases for two lines

    lines = detect_lines(roi_frame)                 # Detect lane lines on the frame
    # cv2.imshow('lines', roi_frame)

    lane_lines = optimize_lines(frame, lines)       # Optimize detected line
    
    direction = direction(frame, lane_lines)
    # 직선 검출 됐을 때
    if lane_lines!=None:
             
        lane_lines_image = display_lines(frame, lane_lines) # Display solid and optimized lines
        
        up_center, low_center = get_floating_center(frame, lane_lines) # Calculate the center between two lines

        heading_line = display_heading_line(lane_lines_image, up_center, low_center)

        final_frame, rotate_angle = add_text(heading_line, up_center, low_center,lane_lines,left_x_base, right_x_base) # Predict and draw turn

        angle = Map(rotate_angle, -30, 30, 0, 1)
        prev_angle.append(angle)
        weight_angle = weight_moving_average(w1=0.5,w2=0.3,w3=0.15,w4=0.05)

        return final_frame, lane_lines  # Return final frame
    else:
        return frame, 'Stop'
