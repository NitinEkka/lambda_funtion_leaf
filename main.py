import base64
import boto3
import numpy as np
import cv2

s3 = boto3.client('s3')

def lambda_handler(event, context):
    bucket_name = event["pathParameters"]["bucket"]
    file_name = event["queryStringParameters"]["file"]
    fileobj = s3.get_objects(Bucket=bucket_name, Key=file_name)
    nparr = np.frombuffer(fileobj['Body'].read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    resizeimg = cv2.resize(img, (400,400))
    resizeimg_copy = resizeimg.copy()
    cv2.imshow("Resized Image", resizeimg)
    hsv = cv2.cvtColor(resizeimg, cv2.COLOR_BGR2HSV)

#   Black Mask
#   upper_black = np.array([360,255,50])
    lower_black = np.array([0,0,0])
    upper_black = np.array([360,255,100])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    # cv2.imshow("Black Mask", mask_black)
    # cv2.waitKey(0)

    contours, _ = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, cont in enumerate(sorted_contours[:3],1):
        x = cv2.drawContours(resizeimg, cont, -1, (0,255,0), 3)
        cv2.putText(resizeimg, str(i), (cont[0,0,0],cont[0,0,1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 4)
        # cv2.imshow("ContourDrawingBlack",x)
        # cv2.waitKey(0)

    def find_contour_areas(contours):
        areas = []
        for cnt in contours:
            cont_area = cv2.contourArea(cnt)
            areas.append(cont_area)
        return areas

    print("Contour areas before sorting", find_contour_areas(contours))
    print()

    sorted_contours_by_area = sorted(contours, key=cv2.contourArea, reverse=True)

    print('Contor areas after sorting', find_contour_areas(sorted_contours_by_area))
    print()    

    frame_area = find_contour_areas(sorted_contours_by_area)[0]
    print("Frame Area :" ,frame_area)

#   Green Mask

    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # cv2.imshow("Green Mask", mask_green)
    # cv2.waitKey(0)

    contoursG, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contoursG = sorted(contoursG, key=cv2.contourArea, reverse=True)

    for i, cont in enumerate(sorted_contoursG[:3],1):
        x = cv2.drawContours(resizeimg_copy, cont, -1, (0,255,0), 3)
        cv2.putText(resizeimg_copy, str(i), (cont[0,0,0],cont[0,0,1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 4)
        # cv2.imshow("ContourDrawingGreen",x)
        # cv2.waitKey(0)

    def find_contour_areas(contoursG):
        areas = []
        for cnt in contoursG:
            cont_area = cv2.contourArea(cnt)
            areas.append(cont_area)
        return areas

    print("Contour areas before sorting", find_contour_areas(contoursG))
    print()

    sorted_contours_by_area = sorted(contoursG, key=cv2.contourArea, reverse=True)

    print('Contor areas after sorting', find_contour_areas(sorted_contours_by_area))
    print()   

    leaf_area = find_contour_areas(sorted_contours_by_area)[0]
    print("Leaf Area :" ,leaf_area)

#   Actual Calculation



    total_area_BG = frame_area+leaf_area
    ratio = total_area_BG/leaf_area
    print("Ratio is : ", ratio)

    actual_frame_area = 8250

    actual_leaf_area = actual_frame_area/ratio
    print("Actual leaf Area : ",actual_leaf_area)

    return {
        "Leaf Area" : actual_leaf_area
    }