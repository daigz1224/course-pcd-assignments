# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct

from scipy.spatial import KDTree

import octree as octree
import kdtree as kdtree
from result_set import KNNResultSet, RadiusNNResultSet


np.seterr(all='raise')

def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    # root_dir = '/Users/renqian/cloud_lesson/kitti' # 数据集路径
    root_dir = './data' # 数据集路径
    cat = os.listdir(root_dir)
    iteration_num = len(cat)

    print("scipy ---------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        begin_t = time.time()
        root = KDTree(db_np, leaf_size)
        construction_time_sum += time.time() - begin_t

        begin_t = time.time()
        query = db_np[0, :]
        result_set = KNNResultSet(capacity=k)
        distance, indices = root.query(x=query, k=k)
        output = ''
        for i, item in enumerate(zip(indices, distance)):
            output += '%d - %.2f\n' % (item[0], item[1])
        # print(output)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        indices = root.query_ball_point(query, radius)
        output = ''
        for i, index in enumerate(indices):
            output += '%d - %.2f\n' % (index, np.linalg.norm(db_np[index] - query))
        # print(output)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Scipy: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum*1000/iteration_num,
                                                                     knn_time_sum*1000/iteration_num,
                                                                     radius_time_sum*1000/iteration_num,
                                                                     brute_time_sum*1000/iteration_num))
    sci_knn = knn_time_sum
    sci_radius = radius_time_sum
    sci_brute = brute_time_sum


    print("octree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        begin_t = time.time()
        root = octree.octree_construction(db_np, leaf_size, min_extent)
        construction_time_sum += time.time() - begin_t

        query = db_np[0,:]

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        octree.octree_knn_search(root, db_np, result_set, query)
        # print(result_set)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        octree.octree_radius_search_fast(root, db_np, result_set, query)
        # print(result_set)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum*1000/iteration_num,
                                                                     knn_time_sum*1000/iteration_num,
                                                                     radius_time_sum*1000/iteration_num,
                                                                     brute_time_sum*1000/iteration_num))
    oct_knn = knn_time_sum
    oct_radius = radius_time_sum
    oct_brute = brute_time_sum

    print("kdtree1 --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np, leaf_size)
        construction_time_sum += time.time() - begin_t

        query = db_np[0,:]

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
        # print(result_set)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        kdtree.kdtree_radius_search(root, db_np, result_set, query)
        # print(result_set)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Kdtree1: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))
    kd1_knn = knn_time_sum
    kd1_radius = radius_time_sum
    kd1_brute = brute_time_sum

    print("kdtree2 --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np, leaf_size, axis_method='by_variance')
        construction_time_sum += time.time() - begin_t

        query = db_np[0,:]

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
        # print(result_set)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        kdtree.kdtree_radius_search(root, db_np, result_set, query)
        # print(result_set)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Kdtree2: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))
    kd2_knn = knn_time_sum
    kd2_radius = radius_time_sum
    kd2_brute = brute_time_sum

    print('==================== BENCHMARK =====================')
    print("Scipy   knn/brute =  %.3f,   radius/brute = %.3f" % (sci_knn/sci_brute, sci_radius/sci_brute))
    print("Octree  knn/brute =  %.3f,   radius/brute = %.3f" % (oct_knn/oct_brute, oct_radius/oct_brute))
    print("Kdtree1 knn/brute =  %.3f,   radius/brute = %.3f" % (kd1_knn / kd1_brute, kd1_radius / kd1_brute))
    print("Kdtree2 knn/brute =  %.3f,   radius/brute = %.3f" % (kd2_knn / kd2_brute, kd2_radius / kd2_brute))

if __name__ == '__main__':
    main()