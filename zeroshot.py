import argparse
from copy import deepcopy

import clip
from models.imagepoint import ImagePoint
import torch
import utils
from datasets import ModelNet40Align, ModelNet40Ply
from render.render import Renderer
from render.selector import Selector
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import read_state_dict

#clip_model, _ = clip.load("ViT-B/32", device='cpu')
clip_model = utils.load_clip("./ViT-B-32.pt", device='cpu')



def inference(args):    
    prompts = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower pot', 'glass box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night stand', 'person', 'piano', 'plant', 'radio', 'range hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv stand', 'vase', 'wardrobe', 'xbox']
    prompts = ['image of a ' + prompts[i] for i in range(len(prompts))]
    prompts = clip.tokenize(prompts)
    prompts = clip_model.encode_text(prompts)
    prompts_feats = prompts / prompts.norm(dim=-1, keepdim=True)
    dataset = ModelNet40Ply()
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, num_workers=4, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = deepcopy(clip_model.visual).to(device)
    model = ImagePoint(args).to(device)
    if args.ckpt is not None:
        #checkpoint = torch.load(args.ckpt)  
        #model.load_state_dict(torch.load(args.ckpt,map_location=device))
        #model.load_state_dict(checkpoint['net'])
        model.load_state_dict(read_state_dict(args.ckpt))
    selector = Selector(args.views, 0).to(device)
    render = Renderer(points_per_pixel=1, points_radius=0.02).to(device)
    prompt_feats = prompts_feats.to(device)

    model.eval()
    with torch.no_grad():
        correct_num = 0
        total = 0
        for (points, label) in tqdm(dataloader):
            points = points.to(device)
            # c_views_azim, c_views_elev, c_views_dist = selector(points)
            # images = render(points, c_views_azim, c_views_elev, c_views_dist, args.views, rot=True)
            # b, n, c, h, w = images.shape
            # images = images.reshape(-1, c, h, w)
            # image_feats = model(images)
            # image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
            # logits = image_feats @ prompt_feats.t()
            # logits = logits.reshape(b, n, -1)
            # logits = torch.sum(logits, dim=1)
            point_feats = model.infer(points)
                #          32*512         512*40
            logits = point_feats @ prompt_feats.t()
            probs = logits.softmax(dim=-1)
            index = torch.max(probs, dim=1).indices
            correct_num += torch.sum(torch.eq(index.detach().cpu(), label)).item()
            total += len(label)
    test_acc = correct_num / total
    print(test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zero-shot Point Cloud Classification')
    parser.add_argument('--views', type=int, default=10)
    parser.add_argument('--ckpt', type=str, default='./pre_results/vit32-image-pointtransv1/best_test.pth')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--dim', type=int, default=0, choices=[0, 512], 
                        help='0 if the view angle is not learnable')
    parser.add_argument('--model', type=str, default='PointNet', metavar='N',
                        choices=['DGCNN', 'PointNet'],
                        help='Model to use, [pointnet, dgcnn]')
    # point transformer
    parser.add_argument('--num_points', type=int, default=1024,
                        help='number of points ')
    parser.add_argument('--nblocks', type=int, default=4,
                        help='number of transformer blocks ')
    parser.add_argument('--nneighbor', type=int, default=16,
                        help='number of neighbor in knn')
    parser.add_argument('--input_dim', type=int, default=3,
                        help='input dim')
    parser.add_argument('--transformer_dim', type=int, default=512,
                        help='transformer_dim')
    args = parser.parse_args()

    inference(args)
