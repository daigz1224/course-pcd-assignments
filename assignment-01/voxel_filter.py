# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, method='random'):
    filtered_points = []
    # 作业3
    # 屏蔽开始

    point_cloud = np.array(point_cloud, dtype=np.float64) # int32 overflow!
    X = point_cloud[:, 0]
    Y = point_cloud[:, 1]
    Z = point_cloud[:, 2]

    # compute the min or max of the point set
    x_max, x_min = np.max(X), np.min(X)
    y_max, y_min = np.max(Y), np.min(Y)
    z_max, z_min = np.max(Z), np.min(Z)

    # compute the dimension of the voxel grid
    Dx = np.ceil((x_max - x_min) / leaf_size)
    Dy = np.ceil((y_max - y_min) / leaf_size)
    Dz = np.ceil((z_max - z_min) / leaf_size)

    # compute voxel index for each point
    hx = np.floor(((X - x_min) / leaf_size))
    hy = np.floor(((Y - y_min) / leaf_size))
    hz = np.floor(((Z - z_min) / leaf_size))
    idx = np.array(hx + hy * Dx + hz * Dz * Dy, dtype=np.float64)

    # sort the points according to the index
    point_cloud_idx = np.insert(point_cloud, 0, values=idx, axis=1)
    point_cloud_idx = point_cloud_idx[np.lexsort(point_cloud_idx[:,::-1].T)]

    # iterate the sorted points
    point_cloud_idx[:, 0].astype(np.int32)
    n, k = 0, point_cloud_idx[0, 0]
    if method == 'random':
        for i in range(point_cloud_idx.shape[0]):
            if point_cloud_idx[i, 0] != k:
                point_rand = np.random.randint(n, i)
                filtered_points.append(
                    point_cloud_idx[point_rand, 1:4])
                n, k = i, point_cloud_idx[i, 0]
    elif method == 'centroid':
        for i in range(point_cloud_idx.shape[0]):
            if point_cloud_idx[i, 0] != k:
                filtered_points.append(
                    np.mean(point_cloud_idx[n:i, 1:4], axis=0))
                n, k = i, point_cloud_idx[i, 0]
    else:
        raise NotImplementedError

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    data = np.loadtxt('./data/airplane_0001.txt', delimiter=',')
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(data[:,:3])
    # file_name = "/Users/renqian/Downloads/program/cloud_data/11.ply"
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 转成open3d能识别的格式
    # point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    # filtered_cloud = voxel_filter(point_cloud_pynt.points, 100.0)
    filtered_cloud = voxel_filter(np.array(point_cloud_o3d.points), 0.07, method='centroid')
    if len(filtered_cloud) > 0:
        point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
        # 显示滤波后的点云
        o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()

