from copy import deepcopy
import random
from traceback import print_tb

import clip
import torch
import torch.nn as nn
from models.point2views import Point2Views
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
        self.projection = Point2Views(512,dim=3)
    
    def infer(self, points):
        azim, elev, dist = self.selector(points)
        
        rand_idx = random.randint(0, 9)
        a,e,d = azim[:,rand_idx], elev[:,rand_idx], dist[:,rand_idx]
        # imgs = self.renderer(points, azim, elev, dist, self.views)
        # b, n, c, h, w = imgs.size()
        # imgs = imgs.reshape(b * n, c, h, w)
        # imgs = self.point_model(imgs)
        a = a.unsqueeze(-1)
        e = e.unsqueeze(-1)
        d = d.unsqueeze(-1)
        point_feat = self.point_model(points)
        point_feat = self.projection(point_feat,a,e,d)
        # project to 10 views   
        #point_feat = self.projection(point_feat,azim,elev,dist)
        # b*10*512
        #
        point_feats = point_feat / point_feat.norm(dim=-1, keepdim=True)
        return point_feats
    
    def forward(self, points, images,a,e,d):
        batch_size = points.shape[0]
        #print(images.squeeze(1).shape)
        # b, n, c, h, w = images.size()
        # images = images.reshape(b * n, c, h, w)
        # image_feat = self.image_model(images.squeeze(1)).detach()
        # image_feat = torch.sum(image_feat.reshape(b,n,-1),dim=1)
        #print(image_feat.shape)
        point_feat = self.point_model(points) #    batch_size/gpu_num  * 512
        point_feat = self.projection(point_feat,a,e,d)
        image_feat = self.image_model(images.squeeze(1)).detach()
        loss = self.criterion(point_feat, image_feat)
        return loss
