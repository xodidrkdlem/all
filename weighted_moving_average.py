prev_angle = []

def opencv():
    #angle값을 보낸다.
    #prev_angle.append(angle)
    pass


def weight_moving_average(w1,w2,w3,w4):   #w1= 0.5 //  w2 = 0.3 // w3 = 0.15 // w4 = 0.05
    global prev_angle  


    if len(prev_angle)<=1:
        WMA_angle = prev_angle[0]
        print('WMA_angle: ',WMA_angle)
    elif len(prev_angle)==2:
        WMA_angle = prev_angle[0]*w4+prev_angle[1]*w3+prev_angle[1]*w2+prev_angle[1]*w1
        print('WMA_angle: ',WMA_angle)
    elif len(prev_angle)==3:
        WMA_angle = prev_angle[0]*w4+prev_angle[1]*w3+prev_angle[2]*w2+prev_angle[2]*w1
        print('WMA_angle: ',WMA_angle)
    elif len(prev_angle)>=4:    
        WMA_angle = prev_angle[0]*w4+prev_angle[1]*w3+prev_angle[2]*w2+prev_angle[3]*w1 

        prev_angle.pop(0)
        print('WMA_angle: ',WMA_angle)
    return WMA_angle


if __name__ == "__main__":
    weight_moving_average()

    
