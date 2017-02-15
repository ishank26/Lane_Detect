import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def white_mask(img, white_threshold=200):
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(img, lower_white, upper_white)
    white_img = cv2.bitwise_and(img, img, mask=white_mask)
    return white_img


#white_img= white_mask(img)


# mask = np.zeros(img.shape[:2], np.uint8)
# mask[200:850, 250:1090]= 255
# img_mask = cv2.bitwise_and(edge,edge,mask = mask)


def blur_img(img, kernel=7):
    ''' Add gaussian blur to image '''
    blur = cv2.GaussianBlur(img, (kernel, kernel), 0)
    #blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    return blur


def grayscale(img):
    ''' Grayscale input image '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def edge_detect(img, lower_thresh=20, upper_thresh=150):
    ''' Apply canny edge detector '''
    edges = cv2.Canny(img, lower_thresh, upper_thresh)
    return edges


def clahe_apply(img, tile_size=7, clip_limit=0.8):
    ''' 
    Apply Contrast Limited Adjacent Histogram Equalization 
    Works better than General Histogram Equalization on this dataset
    '''
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_size, tile_size))
    clahe_img = clahe.apply(img)
    return clahe_img


def region_of_interest(img, vertices):
    ''' apply mask to image other than region of interest (trapezoid) '''
    # define a blank mask
    mask = np.zeros_like(img)

    # define a 1 channel or 3 channel color to fill the mask
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # fill pixels inside the polygon with ignore_mask_color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # return image where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def separate_lines(lines):
    left_lines = []
    right_lines = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 - x1 == 0.:  # corner case
            slope = 9999.  # practically infinite slope
        else:
            slope = (float(y2) - y1) / (x2 - x1)
            if slope >= 0:
                right_lines.append([x1, y1, x2, y2, slope])
            else:
                left_lines.append([x1, y1, x2, y2, slope])

    return right_lines, left_lines


def reject_outliers(data, cutoff_range):
    if data:
        data = np.array(data)
        data = data[(data[:, 4] >= cutoff_range[0]) &
                    (data[:, 4] <= cutoff_range[1])]
        m = np.mean(data[:, 4], axis=0)
        threshold = np.std(data[:, 4], axis=0)
        data = data[(data[:, 4] <= m + threshold) &
                    (data[:, 4] >= m - threshold)]
        return data
    else:
        return


def extend_point(x1, y1, x2, y2, length):
    line_len = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    x = x2 + (x2 - x1) / line_len * length  # x'= x2 + cos(z) * length
    y = y2 + (y2 - y1) / line_len * length  # y'= y2 + sin(z) * length

    return x, y


def merge_lines(lines):
    lines = np.array(lines)[:, :4]  # Drop last column (slope)
    x1, y1, x2, y2 = np.mean(lines, axis=0)
    x1b, y1b = extend_point(x1, y1, x2, y2, -400)  # bottom point
    x2u, y2u = extend_point(x1, y1, x2, y2, 300)  # top point
    line = np.array([[x1b, y1b, x2u, y2u]])

    return np.array([line], dtype=np.int32)


def draw_lines(lines_img, remain_hough, color=[255, 0, 0], thickness=2):
    for line in remain_hough:
        for x1, y1, x2, y2 in line:
            cv2.line(lines_img, (x1, y1), (x2, y2), color, thickness)


def hough(img, rho, theta, threshold, min_line_len, max_line_gap):
    #hough = cv2.HoughLines(roi, 1, np.pi / 180, 40, 30, 200)
    hough_lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)  # returns [[[x1,y1,x2,y2]]] for each line

    return hough_lines


def weighted_img(img, initial_img, a=0.8, b=1., c=0.):  # initial_img * a + img * b + c
    return cv2.addWeighted(initial_img, a, img, b, c)


def process_lane(img):

    global height
    global width

    height = img.shape[0]
    width = img.shape[1]

    global vertices
    vertices = np.array([[
        [(3 * width / 4), height / 5],
        [width / 4,  height / 5],
        [40, height],
        [width - 40, height]
    ]], dtype=np.int32)

    roi = region_of_interest(img, vertices)

    img_gray = grayscale(roi)

    blur = blur_img(img_gray)

    clahe = clahe_apply(blur)

    edges = edge_detect(clahe)
    #edges= edge_detect(blur)

    rho = 3  # distance resolution in pixels of the Hough grid
    theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 20
    min_line_len = 10  # minimum number of pixels making up a line
    max_line_gap = 30    # maximum gap in pixels between connectable line segments

    hough_lines = hough(
        edges, rho, theta, threshold, min_line_len, max_line_gap)

    if hough_lines is not None:
        right_lines, left_lines = separate_lines(hough_lines)
    else:
        print "Bad data", img_name
        return

    # print right_lines
    # print left_lines

    # right lines
    if right_lines:
        right_lines = reject_outliers(right_lines, cutoff_range=(0.5, 2.5))
        right_lines = merge_lines(right_lines)
    else:
        print "Bad data right lines", img_name
        return

    # left lines
    if left_lines:
        left_lines = reject_outliers(left_lines, cutoff_range=(-0.85, -0.6))
        left_lines = merge_lines(left_lines)
    else:
        print "Bad data left lines", img_name
        return

    lines = np.concatenate((right_lines, left_lines))
    #lines_img = np.zeros(img.shape, dtype=np.uint8)
    lines_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(lines_img, lines, thickness=8)
    final_img = weighted_img(lines_img, img)

    return final_img


'''
f = open("output.txt", 'a')
sys.stdout = f

img_dir = 'e-rick-lanes-01/'
out_dir = 'e-rick_test1/'
for img_name in os.listdir(img_dir):
    image = cv2.imread(os.path.join(img_dir, img_name))
    print "Processing ", img_name
    proc = process_lane(image)
    cv2.imwrite(out_dir + "/" + img_name, proc)
print ".....Complete....."    

f.close()

'''


if __name__ == '__main__':

    ### try on image ######
    img = cv2.imread("frame0820.jpg")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = blur_img(img_gray)

    clahe = clahe_apply(blur)

    edges = cv2.Canny(clahe, 20, 150)

    final = process_lane(img)

    plt.figure(figsize=(75, 50))
    plt.subplot(131)
    #plt.imshow(img, cmap='gray')
    plt.imshow(clahe, cmap='gray')
    plt.title("CLAHE")
    plt.subplot(132)
    plt.imshow(edges, cmap='gray')
    plt.title("Edges")
    plt.subplot(133)
    plt.imshow(final)
    plt.title("Final")
    plt.show()
