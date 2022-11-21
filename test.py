import torch
import torch.utils
import torch.utils.cpp_extension
import clip
import numpy
import os
print(torch.utils.cpp_extension.CUDA_HOME)
print(torch.cuda.is_available())
print(clip.available_models())
#clip.load("ViT-B/32", device='cpu')

pc_path = './data/ShapeNet55/shapenet_pc'
points = numpy.load(os.path.join(pc_path, '04330267-a95a8fc9fa81aadf40332412c5d013fb.npy')).astype(numpy.float32)

        # points = self.random_sample(points, self.sample_points_num)
#points = self.pc_norm(points)
points = torch.from_numpy(points).float()
print(points.shape)