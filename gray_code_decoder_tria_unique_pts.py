import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
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


def compute_shadow_mask(black_img, white_img, threshold):
    """ shaded area are with value 0 """
    shadow_mask = np.zeros_like(black_img)
    shadow_mask[white_img > black_img + threshold] = 1
    return shadow_mask


def intersect_mtlb(a, b):
    """
    MATLAB's intersect(a, b) returns:
    c: common values of a and b, sorted
    ia: the first position of each of them in a
    ib: the first position of each of them in b
    """
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]


def find_ids_of_unique_intersect_rows(A, B):
    """ Given A and B of shape (N, d), find respective ids of A and B such that A[id1, :] = B[id2, :] """
    s1 = np.array([','.join(item) for item in A.astype(str)])
    s2 = np.array([','.join(item) for item in B.astype(str)])
    _, ia, ib = intersect_mtlb(s1, s2)
    assert np.linalg.norm(A[ia] - B[ib]) <= 1e-8, "intersect ids are not correct!"
    return ia, ib


##--------------------------------------------------------------------


if __name__ == '__main__':
    proj_w, proj_h = 1920, 1080
    white_thred, black_thred = 5, 40 # white-thred for pos-neg decode, black_thred for white-black shading detection
    img_size = (2048, 1500)
    img_dir = './data/bag'
    calib_data_dir = './data/stereo_calib_flir/stereo_calib_data.h5'

    # load camera data
    calib_data = load_h5py_to_dict(calib_data_dir)
    map_x_l, map_y_l, map_x_r, map_y_r, P1, P2, Q = generate_rectify_data(calib_data, size=img_size)

    # decoder
    graycode = cv2.structured_light_GrayCodePattern.create(width=proj_w, height=proj_h)
    graycode.setWhiteThreshold(white_thred)
    num_required_imgs = graycode.getNumberOfPatternImages()
    print("num of necessary pattern imgs is {}".format(num_required_imgs))

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

    # find camera-projection correspondences
    shadow_mask_l = compute_shadow_mask(black_list[0], white_list[0], black_thred)
    shadow_mask_r = compute_shadow_mask(black_list[1], white_list[1], black_thred)
    cam_l, proj_l, cam_r, proj_r = [], [], [], []
    for i in range(shadow_mask_l.shape[0]):
        for j in range(shadow_mask_l.shape[1]):
            # left
            if shadow_mask_l[i, j] != 0:
                error, projPixel = graycode.getProjPixel(pattern_list[0], j, i)
                if not error:
                    cam_l.append([j, i])
                    proj_l.append(list(projPixel))
            # right
            if shadow_mask_r[i, j] != 0:
                error, projPixel = graycode.getProjPixel(pattern_list[1], j, i)
                if not error:
                    cam_r.append([j, i])
                    proj_r.append(list(projPixel))
    cam_l, proj_l, cam_r, proj_r = np.array(cam_l), np.array(proj_l), np.array(cam_r), np.array(proj_r)

    # fine pixel intersection between two cameras
    id_l, id_r = find_ids_of_unique_intersect_rows(proj_l, proj_r)
    cam_l, cam_r = cam_l[id_l][:, np.newaxis, :], cam_r[id_r][:, np.newaxis, :]

    np.save('./gt_l', cam_l.squeeze())
    np.save('./gt_r', cam_r.squeeze())

    # triangulate to pcd
    pts4D = cv2.triangulatePoints(P1, P2, np.float32(cam_l), np.float32(cam_r)).T
    pts3D = pts4D[:, :3] / pts4D[:, -1:]

    # save ply
    pcd = numpy_to_o3d(pts3D)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    pcd = pcd.select_by_index(ind)
    o3d.io.write_point_cloud('./pcd.ply', pcd)
