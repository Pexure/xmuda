from xmuda.data.nuscenes.nuscenes_dataloader import NuScenesSCN
import numpy as np
import os.path as osp

preprocess_dir = "/home/xyyue/xiangyu/nuscenes_unzip/xmuda_lidarseg_preprocess"
nuscenes_dir = "/home/xyyue/xiangyu/nuscenes_unzip"
split = ('train_usa',)
# pselab_paths = ('/home/docker_user/workspace/outputs/xmuda/nuscenes/day_night/xmuda/pselab_data/train_night.npy',)
dataset = NuScenesSCN(split=split,
                      preprocess_dir=preprocess_dir,
                      nuscenes_dir=nuscenes_dir,
                      scale=1,
                      # pselab_paths=pselab_paths,
                      merge_classes=True,
                      use_image=True,
                      )
import open3d as o3d
pcd = o3d.geometry.PointCloud()
v3d = o3d.utility.Vector3dVector

for i in range(5):
    data = dataset[i]
    seg_points = data['coords']
    pcd.points = v3d(seg_points)  # seg_points=(N, 3)
    o3d.io.write_point_cloud(osp.join('tmp/xmuda_train_usa', f'{i}.pcd'), pcd)
    print(i, len(seg_points))
    print()
