from xmuda.data.nuscenes.nuscenes_dataloader import NuScenesSCN
import numpy as np

preprocess_dir = "/home/xyyue/xiangyu/nuscenes_unzip/xmuda_lidarseg_preprocess"
nuscenes_dir = "/home/xyyue/xiangyu/nuscenes_unzip"
split = ('train_day',)
# pselab_paths = ('/home/docker_user/workspace/outputs/xmuda/nuscenes/day_night/xmuda/pselab_data/train_night.npy',)
dataset = NuScenesSCN(split=split,
                      preprocess_dir=preprocess_dir,
                      nuscenes_dir=nuscenes_dir,
                      # pselab_paths=pselab_paths,
                      merge_classes=True,
                      use_image=True,
                      noisy_rot=0.1,
                      flip_x=0.5,
                      rot_z=2 * np.pi,
                      transl=True,
                      fliplr=0.5,
                      color_jitter=(0.4, 0.4, 0.4)
                      )
for i in range(10):
    data = dataset[i]
    seg_label = data['seg_label']
    print(i, len(seg_label))
