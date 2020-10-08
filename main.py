import numpy as np
import cv2 as cv2
import os
import math

ID = {}
Firstname = {}
Lastname = {}
Degree = []


def empty_cell(cell):
    x1 = int(cell.shape[0] * 0.2)
    x2 = int(cell.shape[0] * 0.8)
    y1 = int(cell.shape[1] * 0.2)
    y2 = int(cell.shape[1] * 0.8)
    cell = cell[x1:x2, y1:y2]
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 7)
    total_white = cv2.countNonZero(thresh)
    ratio = total_white / float((x2 - x1) * (y2 - y1))

    if ratio > 0.98:
        return True
    return False


def detect_degree(options):
    min_ratio = math.inf
    min_index = 0

    for index, opt in enumerate(options):
        x1 = int(opt.shape[0] * 0.2)
        x2 = int(opt.shape[0] * 0.8)
        y1 = int(opt.shape[1] * 0.2)
        y2 = int(opt.shape[1] * 0.8)
        opt = opt[x1:x2, y1:y2]

        gray = cv2.cvtColor(opt, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 5)
        total_white = cv2.countNonZero(thresh)

        ratio = total_white / float(thresh.shape[0] * thresh.shape[1])
        print(ratio)
        if (ratio < min_ratio):
            min_ratio = ratio
            min_index = index
    return min_index


def extracted_form_test(path):
    I = cv2.imread(path)
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(I, dictionary, parameters=parameters)

    aruco_list = {}

    for k in range(len(markerCorners)):
        temp_1 = markerCorners[k]
        temp_1 = temp_1[0]
        temp_2 = markerIds[k]
        temp_2 = temp_2[0]
        aruco_list[temp_2] = temp_1

    p1 = aruco_list[34][3]
    p2 = aruco_list[35][2]
    p3 = aruco_list[33][0]
    p4 = aruco_list[36][1]

    width = 500
    height = 550
    points2 = np.array([(0, 0),
                        (width, 0),
                        (0, height),
                        (width, height)]).astype(np.float32)

    points1 = np.array([p1, p2, p3, p4], dtype=np.float32)

    output_size = (width, height)
    H = cv2.getPerspectiveTransform(points1, points2)
    J = cv2.warpPerspective(I, H, output_size)

    gray = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 7)
    kernelOpen = np.ones((2, 2), np.uint8)
    open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelOpen)
    kernel = np.ones((2, 2), np.uint8)
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_x = 5
    max_x = 50
    dir = path.split(os.path.sep)[-1]
    folder_name = dir[:-4]
    image_number = 1
    count = 0
    index = 0
    degree = ["PHD", "MS", "BS"]
    info = ['ID', 'FN', 'LN']
    degree_option = []
    sorted_Y = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1])
    sorted_X = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for c in sorted_Y:
        x, y, w, h = cv2.boundingRect(c)
        if (x > min_x and x < max_x and y < 280 and w>10):
            square_w = w // 9 + 5
            h -= 5
            for i in range(0, 8):
                x_start = x + square_w * i
                cell = J[y:y + h, x_start:x_start + square_w]

                if not os.path.exists('cells/' + str(folder_name)):
                    os.makedirs('cells/' + str(folder_name))
                if not empty_cell(cell):
                    cv2.imwrite('cells/{}/{}.png'.format(str(folder_name), info[index] + str(image_number)), cell)
                    cv2.rectangle(J, (x_start, y), (x_start + square_w, y + h), (255, 255, 12), 2)
                    cv2.imshow('op', J)
                    cv2.waitKey()
                    image_number += 1
            index += 1
            image_number = 1
    for c in sorted_X:
        x, y, w, h = cv2.boundingRect(c)
        if (w > 16 and w < 22 and y > 300 and h / w > 0.95):
            cell = J[y:y + h, x:x + w]
            if not os.path.exists('cells/' + str(folder_name)):
                os.makedirs('cells/' + str(folder_name))
            cv2.imwrite('cells/{}/{}.png'.format(str(folder_name), degree[count]), cell)
            cv2.drawContours(J, c, -1, (255, 0, 0), 2)
            cv2.rectangle(J, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.imshow('op', J)
            cv2.waitKey()
            degree_option.append(cell)
            count += 1

    return degree[detect_degree(degree_option)]


test_dir = "tests/12.jpg"
Degree.append(extracted_form_test(test_dir))
print(Degree)
