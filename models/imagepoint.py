from copy import deepcopy
from traceback import print_tb

import clip
import torch
import torch.nn as nn
import utils
from render import Renderer, Selector
from utils import NTXentLoss

from .model_transv1 import PointTransformerCls

#clip_model, _ = clip.load("ViT-B/32", device='cpu')
clip_model = utils.load_clip("./ViT-B-32.pt", device='cpu')


class ImagePoint(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.views = args.views
        self.selector =  Selector(self.views, args.dim, args.model)
        self.renderer = Renderer(points_radius=0.02)
        self.point_model = PointTransformerCls(args)  ##args.xxx
        self.image_model = deepcopy(clip_model.visual)
        self.weights = nn.Parameter(torch.ones([]))
        self.criterion = NTXentLoss(temperature = 0.07)
    
    def infer(self, points):
        azim, elev, dist = self.selector(points)
        # imgs = self.renderer(points, azim, elev, dist, self.views)
        # b, n, c, h, w = imgs.size()
        # imgs = imgs.reshape(b * n, c, h, w)
        # imgs = self.point_model(imgs)
        point_feat = self.point_model(points)
        point_feats = point_feat / point_feat.norm(dim=-1, keepdim=True)
        return point_feats
    
    def forward(self, points, images):
        batch_size = points.shape[0]
        
        image_feat = self.image_model(images.squeeze(1)).detach()
        
        point_feat = self.point_model(points) #    batch_size/4 * 512
        
        loss = self.criterion(point_feat, image_feat)
        return loss
