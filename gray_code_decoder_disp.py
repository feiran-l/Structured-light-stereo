import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os
import h5py
import skimage
import open3d as o3d
from open3d import geometry as o3dg



def numpy_to_o3d(pcd_np):
    valid_ids = (~np.isnan(pcd_np).any(axis=1)) * (~np.isinf(pcd_np).any(axis=1))
    valid_pcd = pcd_np[valid_ids]
    print('there are {} points'.format(valid_pcd.shape[0]))
    tmp = o3dg.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(valid_pcd)
    return tmp


def load_h5py_to_dict(data_dir):
    res = {}
    calib_data_h5 = h5py.File(data_dir, 'r')
    for k, v in calib_data_h5.items():
        res[k] = np.array(v)
    return res


def generate_rectify_data(calib_data, size):
    M1, M2, d1, d2 = calib_data['M1'], calib_data['M2'], calib_data['d1'], calib_data['d2']
    R, t = calib_data['R'], calib_data['t']
    flag = cv2.CALIB_ZERO_DISPARITY
    R1, R2, P1, P2, Q = cv2.stereoRectify(cameraMatrix1=M1, cameraMatrix2=M2, distCoeffs1=d1, distCoeffs2=d2, R=R, T=t,
                                          flags=flag, alpha=-1, imageSize=size, newImageSize=size)[0:5]
    map_x_l, map_y_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, size, cv2.CV_32FC1)
    map_x_r, map_y_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, size, cv2.CV_32FC1)
    return map_x_l, map_y_l, map_x_r, map_y_r, P1, P2, Q


def rectify(img, map_x, map_y):
    res = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    return res


##--------------------------------------------------------------------


if __name__ == '__main__':
    proj_w, proj_h = 1920, 1080  # resolution of the projector
    white_thred, black_thred = 5, 40  # white-thred for pos-neg decode, black_thred for white-black shading detection
    img_size = (2048, 1500)
    img_dir = './data/bag'
    calib_data_dir = './data/stereo_calib_flir/stereo_calib_data.h5'

    # load camera data
    calib_data = load_h5py_to_dict(calib_data_dir)
    map_x_l, map_y_l, map_x_r, map_y_r, P1, P2, Q = generate_rectify_data(calib_data, size=img_size)

    # decoder
    graycode = cv2.structured_light_GrayCodePattern.create(width=proj_w, height=proj_h)
    graycode.setWhiteThreshold(white_thred)
    graycode.setBlackThreshold(black_thred)
    num_required_imgs = graycode.getNumberOfPatternImages()
    print('num of imgs is {}'.format(num_required_imgs))

    # load data and rectify:
    rect_list_l, rect_list_r = [], []
    for i in range(num_required_imgs + 2):
        img_l = cv2.imread(os.path.join(img_dir, './left/{}.png'.format(i)), 0)
        img_r = cv2.imread(os.path.join(img_dir, './right/{}.png'.format(i)), 0)
        l_rect, r_rect = rectify(img_l, map_x_l, map_y_l), rectify(img_r, map_x_r, map_y_r)
        cv2.imshow('rectified data', skimage.transform.rescale(np.concatenate([l_rect, r_rect], axis=1), 0.2))
        cv2.waitKey(1)
        rect_list_l.append(l_rect)
        rect_list_r.append(r_rect)
    cv2.destroyAllWindows()

    # decoding
    pattern_list = np.array([rect_list_l[:-2], rect_list_r[:-2]])
    white_list = np.array([rect_list_l[-2], rect_list_r[-2]])
    black_list = np.array([rect_list_l[-1], rect_list_r[-1]])
    ret, disp_l = graycode.decode(pattern_list, np.zeros_like(pattern_list[0]), black_list, white_list)
    disp_l = disp_filter(disp_l)  # post-processing the disp map

    plt.imshow(disp_l)
    plt.title('left disparity map')
    plt.show()

    # disp to pcd
    cam_pts_l, cam_pts_r = [], []
    for i in range(disp_l.shape[0]):
        for j in range(disp_l.shape[1]):
            if disp_l[i, j] != 0:
                cam_pts_l.append([j, i])
                cam_pts_r.append([j + disp_l[i, j], i])
    cam_pts_l, cam_pts_r = np.array(cam_pts_l)[:, np.newaxis, :], np.array(cam_pts_r)[:, np.newaxis, :]
    pts4D = cv2.triangulatePoints(P1, P2, np.float32(cam_pts_l), np.float32(cam_pts_r)).T
    pts3D = pts4D[:, :3] / pts4D[:, -1:]

    # save ply
    tmp = numpy_to_o3d(pts3D)
    cl, ind = tmp.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    tmp = tmp.select_by_index(ind)
    o3d.io.write_point_cloud('./pcd.ply', tmp)
