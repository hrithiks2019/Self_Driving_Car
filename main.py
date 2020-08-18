import glob
import pickle
import time
import cv2
import numpy as np
import serial

def undistort_img():
    obj_pts = np.zeros((6 * 9, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    images = glob.glob('camera_cal/*.jpg')
    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            objpoints.append(obj_pts)
            imgpoints.append(corners)
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump(dist_pickle, open('camera_cal/cal_pickle.p', 'wb'))


def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = undistort(img)
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    h_channel = hls[:, :, 0]
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def perspective_warp(img,
                     dst_size=(1280, 720),
                     src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                     dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped


def inv_perspective_warp(img,
                         dst_size=(1280, 720),
                         src=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
                         dst=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped


def get_hist(img):
    hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
    return hist


left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []


def sliding_window(img, nwindows=9, margin=150, minpix=1, draw_windows=True):
    global left_a, left_b, left_c, right_a, right_b, right_c
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255
    histogram = get_hist(img)
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    window_height = np.int(img.shape[0] / nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if draw_windows:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (100, 255, 255), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (100, 255, 255), 3)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty


def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5 / 720
    xm_per_pix = 3.7 / 720
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    return (left_curverad, right_curverad, center)


def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    cv2.fillPoly(color_img, np.int_(points), (255, 5, 0))
    inv_perspective = inv_perspective_warp(color_img)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective


def calc_servo_pos(center_value):
    main_offset = abs(center_value)
    if main_offset < 10:
        return 'ALMOST CENTER', 'CENTER', 0
    elif 10 < main_offset < 100:
        if center_value > 0:
            return 'TURN SLIGHTLY TO LEFT', 'LEFT', 30
        else:
            return 'TURN SLIGHTLY TO RIGHT', 'RIGHT', 30
    elif main_offset > 100:
        if center_value > 0:
            return 'TURN HARDLY TO LEFT', 'LEFT', 75
        else:
            return 'TURN HARDLY TO RIGHT', 'RIGHT', 75
    else:
        return 'NONE', 'NONE', 'NONE'


def vid_pipeline(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray[350:1150, 400:750]
    img_ = pipeline(img)
    img_ = perspective_warp(img_)
    out_img, curves, lanes, ploty = sliding_window(img_, draw_windows=False)
    curverad = get_curve(img, curves[0], curves[1])
    lane_curve = np.mean([curverad[0], curverad[1]])
    left_most_end = np.mean((curves[0]))
    right_most_end = np.mean((curves[1]))
    img = draw_lanes(img, curves[0], curves[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    car_cascade = cv2.CascadeClassifier('Support/cars.xml')
    bike_cascade = cv2.CascadeClassifier('Support/bike.xml')
    bus_cascade = cv2.CascadeClassifier('Support/bus.xml')
    pedestrian_cascade = cv2.CascadeClassifier('Support/pedestrian.xml')
    cv2.putText(img, 'Lane Curvature: {:.0f} m'.format(lane_curve), (450, 50), font, 0.7, (0, 255, 255), 2)
    cv2.putText(img, 'Vehicle offset: {:.4f} m'.format(curverad[2]), (450, 80), font, 0.7, (0, 255, 255), 2)
    cv2.line(img, (640, 535), (640, 595), (0, 0, 255), 2)
    cars = car_cascade.detectMultiScale(gray, 1.5, 3)
    bikes = bike_cascade.detectMultiScale(gray, 1.19, 1)
    buses = bus_cascade.detectMultiScale(gray, 1.16, 1)
    pedestrians = pedestrian_cascade.detectMultiScale(gray, 1.3, 2)
    if (len(cars) > 0) or (len(bikes) > 0) or (len(buses) > 0) or (len(pedestrians) > 0):
        obstacles = []
        if len(cars) > 0:
            for (x, y, w, h) in cars:
                cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
                obstacles.append(f'{len(cars)}-car')
        if len(bikes) > 0:
            for (x, y, w, h) in bikes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 215), 2)
                obstacles.append(f'{len(bikes)}-Bike')
        if len(buses) > 0:
            for (x, y, w, h) in buses:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                obstacles.append(f'{len(buses)}-Bus')
        if len(pedestrians) > 0:
            for (a, b, c, d) in pedestrians:
                cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 210), 4)
                obstacles.append(f'{len(pedestrians)}-Pedestrian')
        cv2.putText(img, f"Obstacles Ahead: {str(obstacles).replace('[','').replace(']','')} in front",
                    (450, 140), font, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(img, "Obstacles Ahead: No Obstacles in Front", (450, 140), font, 0.7, (0, 255, 255), 2)
    return img, gray, left_most_end, right_most_end


right_curves, left_curves = [], []
video_path = 'TEST/Test.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    start_time = time.time()
    ret, img = cap.read()
    if ret:
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
        kp, detection_gray, left_most_end, right_most_end = vid_pipeline(img)
        lane_center_pos = int((right_most_end - left_most_end) / 2) + int(left_most_end)
        cv2.putText(kp, f'Processing Time: {round((time.time() - start_time), 4)} Seconds', (450, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.rectangle(kp, (200, 535), (200 + 880, 535 + 60), (0, 0, 255), 5)
        cv2.putText(kp, 'Steering Control', (530, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(kp, '.', (int(lane_center_pos), 570), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 25)
        offset = int(630 - lane_center_pos)
        steering_commands, side, PWM = calc_servo_pos(offset)
        cv2.putText(kp, f"Remarks: {steering_commands} ", (450, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print(steering_commands)
        cv2.putText(kp, f"Servo Commands: SIDE={side} PWM(DC)={PWM}% ", (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.imshow('Obstacle_Detection_FRONT', detection_gray)
        cv2.imshow('Lane Detection', kp)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
