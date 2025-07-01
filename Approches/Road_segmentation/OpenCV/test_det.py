#!/usr/bin/env python3
"""
Stable lane detection & tracking with dynamic curved fill,
using Kalman filter and polynomial smoothing, corrected np.eye calls.
"""
from __future__ import division
import cv2
import numpy as np
import math
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from scipy.linalg import block_diag
import yt_dlp

def region_of_interest(img, vertices):
    """
    Apply a polygonal mask to an image.
    Args:
        img: Input image as NumPy array.
        vertices: List of points defining ROI polygon (as np.int32).
    Returns:
        Masked image as NumPy array (same shape as img).
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, (255,255,255))
    return cv2.bitwise_and(img, mask)
def gamma_correction_auto(image, equalizeHist=False, vidsize=None):
    """
    Perform automatic gamma correction based on image luminance.
    Args:
        image: Input BGR image as NumPy array.
        equalizeHist: Whether to apply histogram equalization to channels.
        vidsize: Required video size (not used internally).
    Returns:
        Gamma-corrected BGR image as NumPy array.
    """
    if vidsize is None:
        raise ValueError("vidsize must be provided")
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    Yavg = yuv[:,:,0].astype(np.float64).mean()
    eps = 1.19209e-7
    gamma = 0.7 - (-0.3/np.log10(Yavg+eps))
    channels=[]
    for c in cv2.split(image):
        c = (c.astype(np.float32)/255.0)**gamma
        c = np.uint8(c*255)
        if equalizeHist:
            c = cv2.equalizeHist(c)
        channels.append(c)
    return cv2.merge(channels)


def compute_gamma_param(image, vidsize):
    """
    Compute the gamma parameter for luminance normalization.
    Args:
        image: Input BGR image as NumPy array.
        vidsize: Expected video size (unused).
    Returns:
        Gamma parameter as float.
    """
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    Yavg = yuv[:,:,0].mean()
    return 0.7 - (-0.3/np.log10(Yavg+1.19209e-7))

def IPM(image, src_pts, size=(700,600)):
    """
    Apply Inverse Perspective Mapping (bird's-eye transform) to image.
    Args:
        image: Input image as NumPy array.
        src_pts: List of four source points (np.float32).
        size: Output image size (width, height).
    Returns:
        Warped and rotated image as NumPy array.
    """
    dst = np.array([[0,0],[size[0]-1,0],[size[0]-1,size[1]-1],[0,size[1]-1]], np.float32)
    H,_ = cv2.findHomography(np.array(src_pts, np.float32), dst)
    warped = cv2.warpPerspective(image, H, size)
    return cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
class LaneTracker:
    """
    Kalman filter-based tracker for multiple lane lines.
    """
    def __init__(self, n_lanes, proc_q, meas_q, proc_cov_par, proc_type='white'):
        """
        Args:
            n_lanes: Number of lanes to track.
            proc_q: Process noise covariance scalar.
            meas_q: Measurement noise covariance scalar.
            proc_cov_par: Cross-lane process covariance parameter.
            proc_type: Process noise type ('white' or other).
        """
        self.n_lanes = n_lanes
        self.meas_size = 4*n_lanes
        self.state_size = 2*self.meas_size
        self.kf = cv2.KalmanFilter(self.state_size, self.meas_size, 0)
        self._init_matrices(proc_q, meas_q, proc_cov_par, proc_type)
        self.initialized = False
        self.state = np.zeros((self.state_size,1), np.float32)
        self.meas  = np.zeros((self.meas_size,1), np.float32)

    def _init_matrices(self, pq, mq, pcp, ptype):
        """
        Initialize Kalman filter matrices.
        Args:
            pq: Process noise covariance.
            mq: Measurement noise covariance.
            pcp: Process cross covariance.
            ptype: Process noise type.
        """
        self.kf.transitionMatrix = np.eye(self.state_size, dtype=np.float32)
        M = np.zeros((self.meas_size, self.state_size), np.float32)
        for i in range(self.meas_size):
            M[i, 2*i] = 1
        self.kf.measurementMatrix = M
        if ptype == 'white':
            block = np.array([[0.25,0.5],[0.5,1.0]], np.float32)
            self.kf.processNoiseCov = block_diag(*([block]*self.meas_size)) * pq
        else:
            self.kf.processNoiseCov = np.eye(self.state_size, dtype=np.float32) * pq
        for i in range(0, self.meas_size, 2):
            for j in range(1, self.n_lanes):
                idx = i + j*self.meas_size
                self.kf.processNoiseCov[i, idx] = pcp
                self.kf.processNoiseCov[idx, i] = pcp
        self.kf.measurementNoiseCov = np.eye(self.meas_size, dtype=np.float32) * mq
        self.kf.errorCovPre = np.eye(self.state_size, dtype=np.float32)

    def _update_dt(self, dt):
        """
        Update transition matrix to account for new time step.
        Args:
            dt: Time delta in seconds.
        """
        for i in range(0, self.state_size, 2):
            self.kf.transitionMatrix[i, i+1] = dt

    def predict(self, dt):
        """
        Predict next lane positions with the Kalman filter.
        Args:
            dt: Time delta in seconds.
        Returns:
            List of lane positions, or None if not initialized.
        """
        if not self.initialized:
            return None
        self._update_dt(dt)
        x = self.kf.predict()
        lanes = []
        for i in range(0, self.state_size, 8):
            lanes.append((x[i,0], x[i+2,0], x[i+4,0], x[i+6,0]))
        return lanes

    def update(self, detected):
        """
        Update Kalman filter state with detected lane positions.
        Args:
            detected: List of detected lane coordinates (may contain None).
        """
        if not self.initialized:
            if None not in detected:
                for lane, i in zip(detected, range(0, self.state_size, 8)):
                    self.state[i:i+8:2, 0] = lane
                self.kf.statePost = self.state.copy()
                self.initialized = True
        else:
            for lane, i in zip(detected, range(0, self.meas_size, 4)):
                if lane is not None:
                    self.meas[i:i+4, 0] = lane
            self.kf.correct(self.meas)

class LaneDetector:
    """
    Lane detector using Hough transform and geometric heuristics.
    """
    def __init__(self, road_horizon, prob_hough=True, span=0.35):
        """
        Args:
            road_horizon: Pixel row of the image defining horizon/ROI top.
            prob_hough: Use probabilistic Hough transform if True.
            span: Fraction of image height for lane segment span.
        """
        self.road_horizon = road_horizon
        self.prob_hough = prob_hough
        self.vote = 50
        self.roi_theta = 0.3
        self.span = span
        self.left_segs  = []
        self.right_segs = []

    def _base_distance(self, x1, y1, x2, y2, w):
        """
        Compute signed horizontal distance of base point from center.
        Args:
            x1, y1, x2, y2: Line segment endpoints.
            w: Image width.
        Returns:
            Distance (negative for left, positive for right).
        """
        if x1 == x2:
            return (w/2) - x1
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m*x1
        bx = -c/m
        return (w/2) - bx

    def _scale_line(self, x1, y1, x2, y2, h):
        """
        Scale (extrapolate) a line segment to span a fixed vertical region.
        Args:
            x1, y1, x2, y2: Endpoints.
            h: Image height.
        Returns:
            Extrapolated line as tuple (x1, y1, x2, y2).
        """
        y_top = self.road_horizon
        y_bot = min(self.road_horizon + self.span*h, h)
        m = (y2 - y1) / (x2 - x1 + 1e-6)
        if y1 < y2:
            x1 += (y_top - y1)/m; y1 = y_top
            x2 += (y_bot - y2)/m; y2 = y_bot
        else:
            x2 += (y_top - y2)/m; y2 = y_top
            x1 += (y_bot - y1)/m; y1 = y_bot
        return (x1, y1, x2, y2)

    def detect(self, frame):
        """
        Detect left and right lanes in the image using Hough transform.
        Args:
            frame: Input BGR image as NumPy array.
        Returns:
            List [left_lane, right_lane], each as tuple or None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi  = gray[self.road_horizon:frame.shape[0], :]
        edges = cv2.Canny(cv2.medianBlur(roi,5), 60, 120)
        lines = (cv2.HoughLinesP(edges, 1, np.pi/180, self.vote,
                 minLineLength=30, maxLineGap=100) if self.prob_hough else None)
        self.left_segs.clear(); self.right_segs.clear()
        if lines is None:
            return [None, None]
        shifted = lines + np.array([0, self.road_horizon, 0, self.road_horizon])
        w = frame.shape[1]
        best = {'left': (None, float('inf')), 'right': (None, float('inf'))}
        for x1, y1, x2, y2 in shifted[:,0]:
            theta = abs(math.atan2(y2-y1, x2-x1))
            if theta < self.roi_theta: continue
            dist = self._base_distance(x1,y1,x2,y2, w)
            side = 'left' if dist < 0 else 'right'
            (self.left_segs if side=='left' else self.right_segs).append((x1,y1,x2,y2))
            if abs(dist) < abs(best[side][1]):
                best[side] = ((x1,y1,x2,y2), dist)
        left  = (self._scale_line(*best['left'][0], frame.shape[0])  if best['left'][0]  else None)
        right = (self._scale_line(*best['right'][0],frame.shape[0]) if best['right'][0] else None)
        return [left, right]

def fit_curve(segs, deg=2):
    """
    Fit a polynomial curve to a list of line segments.
    Args:
        segs: List of line segments (each a tuple of 4 ints).
        deg: Polynomial degree (default quadratic).
    Returns:
        Fitted polynomial (np.poly1d) or None if input empty.
    """
    if not segs:
        return None
    pts = np.vstack([[x1,y1] for x1,y1,_,_ in segs] +
                    [[x2,y2] for _,_,x2,y2 in segs])
    ys, xs = pts[:,1], pts[:,0]
    coeffs = np.polyfit(ys, xs, deg)
    return np.poly1d(coeffs)

def main():
    """
    Main function: runs lane detection and tracking demo from YouTube video stream.
    Performs video capture, preprocessing, lane detection, Kalman tracking, curve fitting,
    visualization, and interactive display.
    """
    url = 'https://www.youtube.com/watch?v=eBcgG_BhjCY'
    with yt_dlp.YoutubeDL({'format':'best','quiet':True}) as ydl:
        info = ydl.extract_info(url, download=False)
        stream_url = info['url']
    cap = cv2.VideoCapture(stream_url)
    vidsize = (640,480,3)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    road_horizon = int(H*3.1/5)
    roi_pts = [(0,H),(0,road_horizon),(W,road_horizon),(W,H)]

    lt = LaneTracker(2,0.1,15,0,'white')
    ld = LaneDetector(road_horizon,True,0.25)

    left_poly = None
    right_poly = None
    alpha = 0.2  # smoothing factor

    ticks = cv2.getTickCount()
    while True:
        prev = ticks; ticks = cv2.getTickCount()
        dt = (ticks-prev)/cv2.getTickFrequency()

        ret, frame = cap.read()
        if not ret: break

        gamma = gamma_correction_auto(frame, False, vidsize)
        cropped = region_of_interest(gamma, [np.array(roi_pts, np.int32)])

        preds = lt.predict(dt)
        lanes = ld.detect(cropped)

        if preds:
            for x1,y1,x2,y2 in preds:
                mx = (x1+x2)/2
                if mx < W/2:
                    ld.left_segs.append((x1,y1,x2,y2))
                else:
                    ld.right_segs.append((x1,y1,x2,y2))

        new_left = fit_curve(ld.left_segs, 2)
        new_right= fit_curve(ld.right_segs,2)

        if new_left is not None:
            if left_poly is None:
                left_poly = new_left
            else:
                left_poly = np.poly1d(alpha*new_left.coeffs + (1-alpha)*left_poly.coeffs)
        if new_right is not None:
            if right_poly is None:
                right_poly = new_right
            else:
                right_poly = np.poly1d(alpha*new_right.coeffs + (1-alpha)*right_poly.coeffs)

        overlay = np.zeros_like(frame)
        if left_poly is not None and right_poly is not None:
            ys = np.linspace(road_horizon, H, 200)
            lp = np.vstack([left_poly(ys), ys]).T.astype(int)
            rp = np.vstack([right_poly(ys), ys]).T.astype(int)
            poly = np.vstack([lp, rp[::-1]])
            cv2.fillPoly(overlay, [poly], (0,255,0))

        overlay[:road_horizon,:] = 0
        out = cv2.addWeighted(frame,1.0,overlay,0.3,0)
        bev = IPM(overlay, roi_pts)

        lt.update(lanes)

        status = [
            f"Γ={compute_gamma_param(frame,vidsize):.2f}",
            f"proc=0.1, meas=15.0",
            f"Hough thr={ld.vote}, θmin={ld.roi_theta:.2f}"
        ]
        y = 30
        for s in status:
            cv2.putText(out, s, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            y += 25

        cv2.imshow('Lane Tracking', out)
        # cv2.imshow("Bird's-Eye View", bev)
        # cv2.imshow("Gamma", gamma)
        # cv2.imshow("ROI", cropped)
        # cv2.imshow("Overlay", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
