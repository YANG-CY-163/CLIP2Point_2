from copy import deepcopy
from traceback import print_tb

import clip
import torch
import torch.nn as nn
import utils
from render import Renderer, Selector
from utils import NTXentLoss

from .model_transv1 import PointTransformerCls
from .global_depth import GlobalDepth
#clip_model, _ = clip.load("ViT-B/32", device='cpu')
clip_model = utils.load_clip("./ViT-B-32.pt", device='cpu')


class MultiImage(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.views = args.views
        self.selector =  Selector(self.views, args.dim, args.model)
        self.renderer = Renderer(points_radius=0.02)
        self.point_model = PointTransformerCls(args)  ##args.xxx
        self.image_model = deepcopy(clip_model.visual)
        self.weights = nn.Parameter(torch.ones([]))
        self.criterion = NTXentLoss(temperature = 0.07)
        self.global_d = GlobalDepth(num_views=args.views, in_features=512)
    
    def infer(self, images):
        azim, elev, dist = self.selector(points)
        # imgs = self.renderer(points, azim, elev, dist, self.views)
        # b, n, c, h, w = imgs.size()
        # imgs = imgs.reshape(b * n, c, h, w)
        # imgs = self.point_model(imgs)
        point_feat = self.point_model(points)
        point_feats = point_feat / point_feat.norm(dim=-1, keepdim=True)
        return point_feats
    
    def forward(self, images):
        batch_size = images.shape[0]
        b, n, c, h, w = images.size()
        images = images.reshape(b * n, c, h, w)
        image_feat = self.image_model(images.squeeze(1)).detach()
      
        image_feat = self.global_d(image_feat)

        loss = self.criterion(text_feat, image_feat)
        return loss
