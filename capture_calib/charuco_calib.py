import glob
import os
import cv2
import skimage.transform
from cv2 import aruco
import numpy as np
from matplotlib import pyplot as plt
import h5py



def save_h5py_file(name, my_dict):
    h = h5py.File(name, 'w')
    for k, v in my_dict.items():
        h.create_dataset(k, data=np.array([v]).squeeze())
    h.close()



class CharucoMarker(object):
    def __init__(self, board_size=[15, 11.194], marker_division=[12, 9]):
        """
        :param board_size: squareLength and markerLength. e,g.[5, 1] means 5mm and 1mm
        :param marker_division: The number of marker in vertical and horizontal direction. e.g. [5, 7]
        """
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
        self.board = aruco.CharucoBoard_create(marker_division[0], marker_division[1], board_size[0], board_size[1], self.aruco_dict)
        # variables for single camera calibration
        self.all_ids_single = []
        self.all_corners_2d_single = []
        self.all_corners_3d_single = []
        self.valid_img_index_single = []
        # variables for stereo camera calibration
        self.all_ids_stereo = []
        self.all_corners_2d_l, self.all_corners_2d_r = [], []
        self.all_corners_3d_stereo = []
        self.valid_img_index_stereo = []
        # calibration terms
        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-9)
        self.flags = cv2.CALIB_FIX_ASPECT_RATIO  # + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_PRINCIPAL_POINT


    def detect_corners_one_img(self, img):
        """
        detect charuco markers from one img
        :param img: image [H, W, c] read from opencv
        :return: 2D corners and their ids
        """
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners, ids, _ = aruco.detectMarkers(img_gray, self.aruco_dict)
        if len(corners) > 0:
            for corner in corners:
                cv2.cornerSubPix(img_gray, corner, (3, 3), (-1, -1), criteria=criteria)
            _, corners2, ids2 = aruco.interpolateCornersCharuco(corners, ids, img_gray, self.board)
            if corners2 is not None and ids2 is not None and len(corners2) > 3:
                return corners2, ids2
        return None, None


    def find_2D_3D_matches(self, corners, ids):
        """
        :param corners: detected 2d corner points
        :param ids: each id for 2d point
        :return: 3D corner, 2D corner
        """
        obj_point3d, img_point2d = [], []
        for i in range(len(ids)):
            point3d = self.board.chessboardCorners[ids[i][0]]  # shape [3]
            obj_point3d.append(point3d)
            img_point2d.append(corners[i])
        return np.array(obj_point3d), np.array(img_point2d)


    def verify_single_calib(self, cam_params, data_dir, size=None, verbose=True):
        error_list = []
        tol_error = 0
        intrinsic, dist, rvecs, tvecs = cam_params['M'], cam_params['d'], cam_params['R'], cam_params['t']
        img_dir_list = sorted(glob.glob(os.path.join(data_dir, '*.png')))

        # loop through all valid frames
        for i in range(len(self.valid_img_index_single)):
            points3d, points2d = self.all_corners_3d_single[i], self.all_corners_2d_single[i]
            # calculate error
            points2d_project, _ = cv2.projectPoints(points3d, rvecs[i], tvecs[i], intrinsic, distCoeffs=dist)
            error = cv2.norm(points2d, points2d_project, cv2.NORM_L2) / len(points3d)
            error_list.append(error)
            tol_error = tol_error + error
            # plot
            if verbose:
                img = cv2.imread(img_dir_list[self.valid_img_index_single[i]])
                if size is not None:
                    img = cv2.resize(img, size)
                print('re-projection error of the {}-th image: {:.4f}'.format(i, error))
                for j in range(len(points2d_project)):
                    cv2.circle(img, (int(points2d_project[j][0, 0]), int(points2d_project[j][0, 1])), 2, (255, 0, 0), thickness=-1)
                    cv2.circle(img, (int(points2d[j][0, 0]), int(points2d[j][0, 1])), 3, (0, 255, 0), thickness=2)
                    cv2.imshow('re-projection output', cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
                    cv2.waitKey(3)

        # results
        print('average re-projection error is: ', tol_error / len(self.valid_img_index_single))
        return np.array(error_list)


    def get_params_for_undistorted_img(self, camerapara, w, h):
        intrinsic, dist, rvecs, tvecs = camerapara
        new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(intrinsic, dist, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(intrinsic, dist, None, new_cam_mat, (w, h), 5)
        new_dist = np.zeros_like(dist)
        # how to use after: then new_cam_mat is the camera intrinsic for the image dst without any distortion
        # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        return new_cam_mat, new_dist, roi, mapx, mapy


    def calib_single_camera(self, data_dir, f=None, pixel_size=None, size=None):
        """
        :param data_dir: captured data path
        :param f: focal length in mm of the camera
        :param pixel_size: pixel size in mm/pixel
        :param size: downsize the image to size to speed up
        :return: camera parameters
        """
        # downsize the resolution for speed
        img_dir_list = sorted(glob.glob(os.path.join(data_dir, '*.png')))
        if size is not None:
            img_size = size
            scale = cv2.imread(img_dir_list[0]).shape[1] / img_size[0]
        else:
            img_size = cv2.imread(img_dir_list[0], 0).shape
            img_size = (img_size[1], img_size[0])
            scale = 1.0

        # loop all the images
        for i in range(len(img_dir_list)):
            img = cv2.resize(cv2.imread(img_dir_list[i]), img_size)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img_gray = cv2.equalizeHist(img_gray)
            corners, ids = self.detect_corners_one_img(img_gray)
            if corners is not None:
                obj_3d_pts, img_2d_pts = self.find_2D_3D_matches(corners, ids)
                self.all_ids_single.append(ids)
                self.all_corners_2d_single.append(img_2d_pts)
                self.all_corners_3d_single.append(obj_3d_pts)
                self.valid_img_index_single.append(i)

        ## calibrating
        if f is not None and pixel_size is not None:
            cam_mat_init = np.array([[f / pixel_size, 0, img_size[0] / 2],
                                     [0, f / pixel_size, img_size[1] / 2],
                                     [0., 0., 1.]])
            flags = self.flags + cv2.CALIB_USE_INTRINSIC_GUESS
        else:
            cam_mat_init = None
            flags = self.flags
        res, cam_mat, dist, rvecs, tvecs = cv2.calibrateCamera(self.all_corners_3d_single, self.all_corners_2d_single, img_size, cam_mat_init,
                                                               np.zeros((5, 1)), flags=flags, criteria=self.criteria)
        print('single camera calibration done, residual is {:.5f}'.format(res))
        cam_mat = cam_mat * scale  # rescaled the cam_mat back to the original resolution
        cam_mat[2, 2] = 1

        calib_data = {'residual': res, 'M': cam_mat, 'd': dist, 'R': rvecs, 't': tvecs}
        return calib_data


    def malloc_single_calib(self):
        self.all_ids_single = []
        self.all_corners_2d_single = []
        self.all_corners_3d_single = []
        self.valid_img_index_single = []


    def generate_map_for_rectification(self, calib_data, size):
        """
        Usage: rectified_l = cv2.remap(img_l, map_x_l, map_y_l, interpolation=cv2.INTER_LINEAR)
        """
        M1, M2, d1, d2 = np.array(calib_data['M1']), np.array(calib_data['M2']), np.array(calib_data['d1']), np.array(calib_data['d2'])
        R, t = np.array(calib_data['R']), np.array(calib_data['t'])
        R1, R2, P1, P2, Q = cv2.stereoRectify(cameraMatrix1=M1, cameraMatrix2=M2, distCoeffs1=d1, distCoeffs2=d2, R=R, T=t,
                                           flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, imageSize=size, newImageSize=size)[0:5]
        map_x_l, map_y_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, size, cv2.CV_32FC1)
        map_x_r, map_y_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, size, cv2.CV_32FC1)
        rectify_dict = {'map_x_l': map_x_l, 'map_y_l': map_y_l, 'map_x_r': map_x_r, 'map_y_r': map_y_r, 'Q': Q}
        return rectify_dict


    def draw_matches(self, img1, img2):
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(img1, None)
        kp2, des2 = akaze.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
        return img3


    def vertify_stereo_calib(self, calib_data, data_dir_l, data_dir_r):
        img_dir_list_l = sorted(glob.glob(os.path.join(data_dir_l, '*.png')))
        img_dir_list_r = sorted(glob.glob(os.path.join(data_dir_r, '*.png')))
        map_x_l, map_y_l, map_x_r, map_y_r = calib_data['map_x_l'], calib_data['map_y_l'], calib_data['map_x_r'], calib_data['map_y_r']
        for i, (name_l, name_r) in enumerate(zip(img_dir_list_l, img_dir_list_r)):
            gray_l, gray_r = cv2.imread(name_l, 0), cv2.imread(name_r, 0)
            l_rect = cv2.remap(gray_l, map_x_l, map_y_l, interpolation=cv2.INTER_LINEAR)
            r_rect = cv2.remap(gray_r, map_x_r, map_y_r, interpolation=cv2.INTER_LINEAR)
            res = self.draw_matches(l_rect, r_rect)
            res = cv2.resize(res, (res.shape[1] // 2, res.shape[0] // 2))
            res = skimage.transform.rescale(res, 0.3)
            cv2.imshow('rectified pairs', res)
            cv2.waitKey(1000)


    def calib_stereo_camera(self, data_dir_l, data_dir_r, f_l=None, f_r=None, pixel_size_l=None, pixel_size_r=None, size=None):
        # respective intrinsic calibration to initialize the stereo calibration
        calib_data_l = self.calib_single_camera(data_dir_l, f_l, pixel_size_l)
        self.malloc_single_calib()
        calib_data_r = self.calib_single_camera(data_dir_r, f_r, pixel_size_r)
        self.malloc_single_calib()
        cam_l, dist_l = calib_data_l['M'], calib_data_l['d']
        cam_r, dist_r = calib_data_r['M'], calib_data_r['d']

        # detect corners
        img_dir_list_l = sorted(glob.glob(os.path.join(data_dir_l, '*.png')))
        img_dir_list_r = sorted(glob.glob(os.path.join(data_dir_r, '*.png')))
        if size is None:
            size = cv2.imread(img_dir_list_l[0], 0)
            size = (size.shape[1], size.shape[0])

        for i, (name_l, name_r) in enumerate(zip(img_dir_list_l, img_dir_list_r)):
            gray_l, gray_r = cv2.imread(name_l, 0), cv2.imread(name_r, 0)
            # gray_l, gray_r = cv2.equalizeHist(gray_l), cv2.equalizeHist(gray_r)
            corners_l, ids_l = self.detect_corners_one_img(gray_l)
            corners_r, ids_r = self.detect_corners_one_img(gray_r)
            if corners_l is not None and corners_r is not None:
                # only use the corners detected by both imgs
                shared_ids = np.intersect1d(ids_l.flatten(), ids_r.flatten()).reshape(-1, 1)
                corners_l = [corners_l[i] for i in np.where((ids_l == shared_ids[:, None]).all(-1))[1]]
                corners_r = [corners_r[i] for i in np.where((ids_r == shared_ids[:, None]).all(-1))[1]]
                obj_pts_l, img_pts_l = self.find_2D_3D_matches(corners_l, shared_ids)
                obj_pts_r, img_pts_r = self.find_2D_3D_matches(corners_r, shared_ids)
                assert np.linalg.norm(obj_pts_l - obj_pts_r) <= 1e-5, 'Stereo calibration fault: object pts are not identical in left and right images!'
                # append results
                self.all_corners_3d_stereo.append(obj_pts_l)
                self.all_corners_2d_l.append(img_pts_l)
                self.all_corners_2d_r.append(img_pts_r)
                self.all_ids_stereo.append(shared_ids)
                self.valid_img_index_stereo.append(i)

        # stereo calibration
        stereo_flags = self.flags + cv2.CALIB_USE_INTRINSIC_GUESS
        res, M1, d1, M2, d2, R, t, E, F = cv2.stereoCalibrate(self.all_corners_3d_stereo, self.all_corners_2d_l, self.all_corners_2d_r, cam_l, dist_l,
                                                              cam_r, dist_r, size, criteria=self.criteria, flags=stereo_flags)
        print('stereo calibration done, residual is {:.5f}'.format(res))
        calib_data = {'residual': res, 'M1': M1, 'M2': M2, 'd1': d1, 'd2': d2, 'R': R, 't': t, 'E': E, 'F': F}
        rectify_dict = self.generate_map_for_rectification(calib_data, size)
        calib_data.update(rectify_dict)
        return calib_data


##-----------------------------------------------------------------------------------------


def calibrate_single_camera(board_size, marker_division, data_dir, target_size, f=None, pixel_size=None, verbose=True):
    """ f and pixel size can be used to calculate an initial guess of the intrinsic matrix """
    calibrator = CharucoMarker(board_size, marker_division)
    calib_data = calibrator.calib_single_camera(data_dir, f=f, pixel_size=pixel_size, size=target_size)
    calibrator.verify_single_calib(calib_data, data_dir, size=target_size, verbose=verbose)
    return calib_data


def calibrate_stereo_camera(board_size, marker_division, data_dir_l, data_dir_r, f_l=None, f_r=None, pixel_size_l=None, pixel_size_r=None, size=None):
    """ f and pixel size can be used to calculate an initial guess of the intrinsic matrix """
    calibrator = CharucoMarker(board_size, marker_division)
    calib_data = calibrator.calib_stereo_camera(data_dir_l, data_dir_r, f_l, f_r, pixel_size_l, pixel_size_r, size)
    calibrator.vertify_stereo_calib(calib_data, data_dir_l, data_dir_r)
    return calib_data


##--------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # board_size = [15, 11.194]  # square_length, marker_length
    # marker_division = [12, 9]  # squares_x, squares_y
    board_size = [7.5, 5.6]  # square_length, marker_length
    marker_division = [25, 18]  # squares_x, squares_y

    ## single camera calibration
    # data_dir = './data/stereo_calib/flir'
    # target_size = None
    # res = calibrate_single_camera(board_size, marker_division, data_dir, target_size, f=25, pixel_size=0.0034)

    ## stereo camera calibration
    data_dir = '../data/stereo_calib_flir'
    data_dir_l = os.path.join(data_dir, 'left')
    data_dir_r = os.path.join(data_dir, 'right')
    res = calibrate_stereo_camera(board_size, marker_division, data_dir_l, data_dir_r, f_l=25, f_r=25,
                                  pixel_size_l=0.0034, pixel_size_r=0.0034)
    save_h5py_file(os.path.join(data_dir, 'stereo_calib_data.h5'), res)


