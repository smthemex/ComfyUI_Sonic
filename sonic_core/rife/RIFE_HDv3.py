import torch
from .IFNet_HDv3 import *
import torch.nn.functional as F
    
class RIFEModel:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.flownet = IFNet().to(self.device).eval()

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()


    def load_model(self, path, rank=-1):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location ='cpu')))
                

    def inference(self, img0, img1, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4/scale, 2/scale, 1/scale]
        flow, mask, merged = self.flownet(imgs, scale_list)
        return merged[2]