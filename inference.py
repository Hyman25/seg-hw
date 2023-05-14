from collections import defaultdict
import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from Unet import UNET
from Uresnet import UnetResNet
from FPN import FPN
from tqdm import tqdm
from dataset import KITTIDataset
from PIL import Image
import argparse
import time
from utils import setup_seed, tensor2numpy, save_img


def save_infer(imgs, mask, pred, cnt, save_path):
    for i in range(imgs.size(0)):
        save_img(tensor2numpy(imgs[i].permute((1,2,0))), os.path.join(save_path, f'{cnt+i}_road.png'))
        save_img(tensor2numpy(mask[i].permute((1,2,0))), os.path.join(save_path, f'{cnt+i}_mask.png'))
        save_img(tensor2numpy(pred[i].permute((1,2,0))), os.path.join(save_path, f'{cnt+i}_pred.png'))


def inference(model, data_loader, infer_num, device, save_path):
    model = model.eval()
    cnt = 0
    with torch.no_grad():
        for imgs, mask in data_loader:
            mask = mask.to(device)
            imgs = imgs.to(device)
            pred = model(imgs.float())
            pred_mask = torch.zeros_like(pred)
            pred_mask[pred>0.5] = 1.
            save_infer(imgs, mask, pred_mask, cnt, save_path)
            cnt += imgs.size(0)
            if cnt >= infer_num:
                break


def save_args(args):
    os.makedirs(f'{args.name}')
    with open(os.path.join(args.name, 'args.txt'), 'w') as f:
        args_dict = args.__dict__
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in args_dict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', '-n', type=str, default='', help='specify an experiment name')
    parser.add_argument('--img_size', '-s', type=int, default=256)
    parser.add_argument('--batch_size', '-bs', type=int, default=8)
    parser.add_argument('--gpu_id', '-g', type=int, default=5)
    parser.add_argument('--model', '-m', type=str, default='unet', choices=['unet', 'ures', 'fpn'])
    parser.add_argument('--data_path', '-d', type=str, default='/mnt/diskc/hh/datasets/KITTI_Road/training')
    parser.add_argument('--resume_path', '-r', type=str, default=None, help='resume from path')
    parser.add_argument('--infer_num', type=int, default=8)
    args = parser.parse_args()

    if not args.name:
        args.name = args.model
    args.name = './results/' + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '_' + args.name
    save_args(args)
    setup_seed(args.seed)

    return args


if __name__ == '__main__':
    args = parse_args()
    
    DEVICE = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 创建模型
    print(f'creating model: {args.model}')
    if args.model.lower() == 'unet':
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    elif args.model.lower() == 'ures':
        model = UnetResNet(in_channels=3, out_channels=1).to(DEVICE)
    elif args.model.lower() == 'fpn':
        model = FPN(in_channels=3, out_channels=1).to(DEVICE)
    else:
        raise NotImplementedError
    
    # 加载数据
    print(f'creating dataloader: {args.data_path}')
    test_set  = KITTIDataset(data_path=args.data_path, split='test', img_size=args.img_size, aug=False)
    test_dataloader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    start_epoch = -1
    history = defaultdict(list)
    best_loss = math.inf
    best_epoch = -1

    # 加载模型
    if args.resume_path:
        print(f'load model from: {args.resume_path}')
        state = torch.load(args.resume_path, map_location='cpu')
        start_epoch = state['epoch']
        model.load_state_dict(state['model'])
        print(f'model resumed: epoch {start_epoch}')

    # 推理
    inference(model, test_dataloader, args.infer_num, DEVICE, save_path=args.name)