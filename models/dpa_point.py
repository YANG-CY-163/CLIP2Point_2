from copy import deepcopy

import clip
import torch.nn as nn
from models.depthpoint import DepthPoint

import utils
from models.adapter_point import SimplifiedAdapter_point
from models.imagepoint import ImagePoint
from render import Renderer, Selector
from utils import read_state_dict, read_state_dict_net

from .adapter import SimplifiedAdapter

#clip_model, _ = clip.load("ViT-B/32", device='cpu')
clip_model = utils.load_clip("./ViT-B-32.pt", device='cpu')


class DPA_Point(nn.Module):
    def __init__(self, args, eval=False):
        super().__init__()
        self.views = args.views
        self.selector =  Selector(self.views, args.dim, args.model)
        self.renderer = Renderer(points_radius=0.02)
        self.ori_model = deepcopy(clip_model.visual)
        #self.model = ImagePoint(args)
        self.model = DepthPoint(args)
        if not eval and args.ckpt is not None:
            print('loading from %s' % args.ckpt)
            self.model.load_state_dict(read_state_dict_net(args.ckpt))
       
        self.adapter1 = SimplifiedAdapter_point(num_views=args.views, in_features=512)
        self.adapter2 = SimplifiedAdapter(num_views=args.views, in_features=512)
    
    def forward(self, points):
        azim, elev, dist = self.selector(points)
        imgs = self.renderer(points, azim, elev, dist, self.views, rot=True)
        b, n, c, h, w = imgs.size()
        imgs = imgs.reshape(b * n, c, h, w)
        img_feat1 = self.adapter1(self.model.point_model(points))
        img_feat2 = self.adapter2(self.ori_model(imgs))
        img_feats = (img_feat1 + img_feat2) * 0.5
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        return img_feats
