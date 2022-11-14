import warnings
from typing import Any, List, Union

import clip
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from plyfile import PlyData


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def read_state_dict_net(path):
    ckpt = torch.load(path)['net']
    base_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    # for key in list(base_ckpt.keys()):
    #     if key.startswith('point_model.'):
    #         base_ckpt[key[len('point_model.'):]] = base_ckpt[key]
    #     del base_ckpt[key]
    return base_ckpt

def read_state_dict(path):
    ckpt = torch.load(path)
    base_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    for key in list(base_ckpt.keys()):
        if key.startswith('point_model.'):
            base_ckpt[key[len('point_model.'):]] = base_ckpt[key]
        del base_ckpt[key]
    return base_ckpt


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def load_clip(model_path: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    
    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

   
    model = clip.model.build_model(state_dict or model.state_dict()).to(device)
    if str(device) == "cpu":
        model.float()
    return model

class NTXentLoss(nn.Module):
    """Implementation of the Contrastive Cross Entropy Loss."""
   
    def __init__(self,
                 temperature: float = 0.5,
                 memory_bank_size: int = 0,
                 gather_distributed: bool = False):
        super().__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))

    def forward(self,
                out0: torch.Tensor,
                out1: torch.Tensor):
        

        device = out0.device
        batch_size, _ = out1.shape  # 8 512

        # normalize the output to length 1
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        out0_large = out0
        out1_large = out1
        diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

        # calculate similiarities
        # here n = batch_size and m = batch_size * world_size
        # the resulting vectors have shape (n, m)
        logits_00 = torch.einsum('nc,mc->nm', out0, out0_large) / self.temperature
        logits_01 = torch.einsum('nc,mc->nm', out0, out1_large) / self.temperature
        logits_10 = torch.einsum('nc,mc->nm', out1, out0_large) / self.temperature
        logits_11 = torch.einsum('nc,mc->nm', out1, out1_large) / self.temperature
            
        # remove simliarities between same views of the same image
        logits_00 = logits_00[~diag_mask].view(batch_size, -1)
        logits_11 = logits_11[~diag_mask].view(batch_size, -1)

        # concatenate logits
        # the logits tensor in the end has shape (2*n, 2*m-1)
        logits_0100 = torch.cat([logits_01, logits_00], dim=1)
        logits_1011 = torch.cat([logits_10, logits_11], dim=1)
        logits = torch.cat([logits_0100, logits_1011], dim=0)

        # create labels
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        rank = dist.get_rank() if dist.is_initialized() else 0
        labels = labels + rank * batch_size
        labels = labels.repeat(2)

        loss = self.cross_entropy(logits, labels)

        return loss