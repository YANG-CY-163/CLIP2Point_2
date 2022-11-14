import argparse
import os

import clip
import torch
import torch.nn as nn
import torch_optimizer as optim
import utils
from datasets import ModelNet40Align, ShapeNetRender
from models import CLIP2Point, DepthPoint, ImagePoint
from pointnet2_ops import pointnet2_utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import IOStream, read_state_dict

#clip_model, _ = clip.load("ViT-B/32", device='cpu')
clip_model = utils.load_clip("./ViT-B-32.pt", device='cpu')


def _init_(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + '/' + args.exp_name):
        os.makedirs(path + '/' + args.exp_name)


def train(args, io):
    test_prompts = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower pot', 'glass box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night stand', 'person', 'piano', 'plant', 'radio', 'range hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv stand', 'vase', 'wardrobe', 'xbox']
    val_prompts = ['airplane', 'ashcan', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellular telephone', 'chair', 'clock', 'computer keyboard', 'dishwasher', 'display', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'loudspeaker', 'mailbox', 'microphone', 'microwave', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote control', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'vessel', 'washer']
    test_prompts = ['image of a ' + test_prompts[i] for i in range(len(test_prompts))]
    val_prompts = ['image of a ' + val_prompts[i] for i in range(len(val_prompts))]
    test_prompts_ = clip.tokenize(test_prompts)
    test_prompt_feats = clip_model.encode_text(test_prompts_)
    test_prompt_feats = test_prompt_feats / test_prompt_feats.norm(dim=-1, keepdim=True)
    test_prompt_feats = test_prompt_feats
    val_prompts_ = clip.tokenize(val_prompts)
    val_prompt_feats = clip_model.encode_text(val_prompts_)
    val_prompt_feats = val_prompt_feats / val_prompt_feats.norm(dim=-1, keepdim=True)
    val_prompt_feats = val_prompt_feats  #55*512
    
    train_dataloader = DataLoader(ShapeNetRender(partition='train'), batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(ShapeNetRender(partition='test'), batch_size=args.test_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(ModelNet40Align(), batch_size=args.test_batch_size, num_workers=4)
    gpu_num = torch.cuda.device_count()
    gpus = [i for i in range(gpu_num)]
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
    # =================================== INIT MODEL ==========================================================
    summary_writer = SummaryWriter("pre_results/%s/tensorboard" % (args.exp_name))

    #model = ImagePoint(args)   ## model change
    #model = DepthPoint(args)
    model = CLIP2Point(args)
    
    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])  # 多卡训练修改
    model = model.to(device)
    # load after to device,  没有多卡训练的module in key冲突，但一定要指定map_location
    # if args.ckpt is not None:
    #     model.load_state_dict(torch.load(args.ckpt,map_location=device))
    #
    for name, param in model.named_parameters():
        if 'image_model' in name:
            param.requires_grad_(False)
    val_prompt_feats = val_prompt_feats.to(device)
    test_prompt_feats = test_prompt_feats.to(device)
    # ==================================== TRAINING LOOP ======================================================
    optimizer = optim.Lamb(model.parameters(), lr=0.006, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2 * len(train_dataloader),
        T_mult=1,
        eta_min=max(1e-2 * 1e-3, 1e-6),
        last_epoch=-1,
    )

    n_epochs = args.epoch
    max_val_acc = 0
    max_test_acc = 0
    start_epoch = -1
    ## resume
    if args.RESUME:
        #path_checkpoint 
        checkpoint = torch.load(args.ckpt,map_location=device)  
        #model.load_state_dict(torch.load(args.ckpt,map_location=device))
        model.load_state_dict(checkpoint['net'])  
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])  
        start_epoch = checkpoint['epoch']  
        scheduler.load_state_dict(checkpoint['lr_schedule'])
        max_test_acc = checkpoint['test_acc']
        max_val_acc = checkpoint['val_acc']

    ##
    for epoch in range(start_epoch+1,n_epochs):
        model.train()
        loss_sum = 0
        depth_sum = 0
        image_sum = 0
       
        for (image, points, a, e, d) in tqdm(train_dataloader):
            optimizer.zero_grad()
            image = image.to(device)
            points = points.to(device)
            a = a.unsqueeze(-1).to(device)
            e = e.unsqueeze(-1).to(device)
            d = d.unsqueeze(-1).to(device)
            loss,image_loss,depth_loss = model(points, image,a,e,d)
            loss = torch.mean(loss)
            
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
        # Validation and Testing
        model.eval()
        with torch.no_grad():
            correct_num = 0
            total = 0
            for (points, label) in tqdm(val_loader):
                b = points.shape[0]  # test batch size
                #print(points.shape)
                points = points.to(device)
                # infer input : point
                # 32*512      
                point_feats = model.module.infer(points)
                #                          512*55
                logits = point_feats @ val_prompt_feats.t()

                # only for multi-views
                logits = logits.reshape(b, args.views, -1)  # 10 view for each   e.g.32*10*55
                logits = torch.sum(logits, dim=1)  # 10 views 相加

                probs = logits.softmax(dim=-1)
                #print(probs.shape)
                index = torch.max(probs, dim=1).indices
                correct_num += torch.sum(torch.eq(index.detach().cpu(), label)).item()
                total += len(label)
        val_acc = correct_num / total

        with torch.no_grad():
            correct_num = 0
            total = 0
            for (points, label) in tqdm(test_loader):
                b = points.shape[0]
                print(b)
                points = points.to(device)
                fps_idx = pointnet2_utils.furthest_point_sample(points, 1024)# bs*1024
                points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
                #print(points.shape)  # 32*1024*3 
                point_feats = model.module.infer(points)
                #          32*512         512*40
                logits = point_feats @ test_prompt_feats.t()
                logits = logits.reshape(b, args.views, -1)
                logits = torch.sum(logits, dim=1)
                #print(logits.shape)
                probs = logits.softmax(dim=-1)
                index = torch.max(probs, dim=1).indices
                correct_num += torch.sum(torch.eq(index.detach().cpu(), label)).item()
                total += len(label)
        test_acc = correct_num / total

       
        mean_loss = loss_sum / len(train_dataloader)
        io.cprint('epoch%d total_loss: %.4f,  balance_weights: %.4f, val_acc: %.4f, test_acc: %.4f' % (epoch + 1, mean_loss,  model.module.weights, val_acc, test_acc))
        summary_writer.add_scalar('train/loss', mean_loss, epoch + 1)
       
        summary_writer.add_scalar('train/weights', model.module.weights, epoch + 1)
        summary_writer.add_scalar("val/acc", val_acc, epoch + 1)
        summary_writer.add_scalar("test/acc", test_acc, epoch + 1)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            'lr_schedule': scheduler.state_dict(),
            'val_acc': max_val_acc,
            'test_acc': max_test_acc
            }
            #torch.save(model.state_dict(), '%s/%s/best_val.pth' % ('pre_results', args.exp_name))
            torch.save(checkpoint, '%s/%s/best_val.pth' % ('pre_results', args.exp_name))
            io.cprint('save the best val acc at %d' % (epoch + 1))
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            'lr_schedule': scheduler.state_dict(),
            'val_acc': max_val_acc,
            'test_acc': max_test_acc
            }
            torch.save(checkpoint, '%s/%s/best_test.pth' % ('pre_results', args.exp_name))
            io.cprint('save the best test acc at %d' % (epoch + 1))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='vit32', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--views', type=int, default=10)
    parser.add_argument('--ckpt', type=str, default='./pre_results/vit32-image-pointtransv1/best_val.pth')
    parser.add_argument('--dim', type=int, default=0, choices=[0, 512], help='0 if the view angle is not learnable')
    parser.add_argument('--model', type=str, default='PointNet', metavar='N',
                        choices=['DGCNN', 'PointNet'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epoch', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--RESUME', type=bool, default=False, metavar='N',
                        help='if start or resume ')

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

    _init_('pre_results')
    io = IOStream('pre_results' + '/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    train(args, io)
