from copy import deepcopy

import clip
import torch
import torch.nn as nn
#from lightly.loss.ntx_ent_loss import NTXentLoss
import utils
from render import Renderer, Selector
from utils import NTXentLoss, read_state_dict

from .global_depth import GlobalDepth

from .model_transv1 import PointTransformerCls

#clip_model, _ = clip.load("ViT-B/32", device='cpu')
clip_model = utils.load_clip("./ViT-B-32.pt", device='cpu')


class DepthPoint(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.views = args.views
        self.selector =  Selector(self.views, args.dim, args.model)
        self.renderer = Renderer(points_radius=0.02)
        self.point_model = PointTransformerCls(args)
        self.depth_model = deepcopy(clip_model.visual) #
        # 
        ckpt = './ckpt/best_eval.pth'
        self.depth_model.load_state_dict(read_state_dict(ckpt))
        
        self.weights = nn.Parameter(torch.ones([]))
        self.criterion = NTXentLoss(temperature = 0.07)
        self.global_d = GlobalDepth(num_views=args.views, in_features=512)
    
    def infer(self, points):
        azim, elev, dist = self.selector(points)
        imgs = self.renderer(points, azim, elev, dist, self.views)
        b, n, c, h, w = imgs.size()
        imgs = imgs.reshape(b * n, c, h, w)
        imgs = self.depth_model(imgs)
        img_feats = imgs / imgs.norm(dim=-1, keepdim=True)
        return img_feats
    
    def forward(self, points, images,a,e,d):
        batch_size = points.shape[0]
        #depths = self.renderer(points, a, e, d, 1, aug=True, rot=True)
        point_feat = self.point_model(points)
        
        depths = self.renderer(points, a, e, d, self.views)  #view num=?
        b, n, c, h, w = depths.size()
        depths = depths.reshape(b * n, c, h, w)
        depths = self.depth_model(depths)  # multi-view
        
        #     obtain global
        depth_feat = self.global_d(depths)
        point_loss = self.criterion(depth_feat, point_feat)
        return point_loss
        #return point_loss + depth_loss / (self.weights ** 2) + torch.log(self.weights + 1), image_loss, depth_loss
