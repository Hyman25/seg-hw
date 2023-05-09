from collections import defaultdict
import os
import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from loss import IoULoss, BCEDiceLoss
from Unet import UNET
from Uresnet import UnetResNet
from FPN import FPN
from tqdm import tqdm
from dataset import KITTIDataset
import matplotlib.pyplot as plt
import argparse
import time
from utils import setup_seed

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    losses = []
    for train_input, train_mask in tqdm(data_loader, ncols=120):
        train_mask = train_mask.to(device)
        train_input=train_input.to(device)
        outputs=model(train_input.float())
        loss = loss_fn(outputs.float(), train_mask.float())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []
    with torch.no_grad():
        for val_input, val_mask in data_loader:
            val_mask = val_mask.to(device)
            val_input=val_input.to(device)
            outputs=model(val_input.float())
            loss = loss_fn(outputs.float(), val_mask.float())
            losses.append(loss.item())
    return np.mean(losses)


def plot_loss(history, path):
    plt.figure()
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.title('Training history')
    plt.ylim(0, 0.6)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss.png'))


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
    parser.add_argument('--name', '-n', type=str, default='', help='specify an experiment name')
    parser.add_argument('--epoch', '-e', type=int, default=1500)
    parser.add_argument('--img_size', '-s', type=int, default=256)
    parser.add_argument('--batch_size', '-bs', type=int, default=8)
    parser.add_argument('--gpu_id', '-g', type=int, default=5)
    parser.add_argument('--loss', '-l', type=str, default='dice', choices=['iou', 'dice'])
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--model', '-m', type=str, default='unet', choices=['unet', 'ures', 'fpn'])
    parser.add_argument('--data_path', '-d', type=str, default='/mnt/diskc/hh/datasets/KITTI_Road/training')
    parser.add_argument('--resume_path', '-r', type=str, default=None, help='resume from path')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    setup_seed()

    args = parse_args()
    args.name = './checkpoints/' + args.name + '_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
    save_args(args)

    DEVICE = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 创建模型
    if args.model.lower() == 'unet':
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    elif args.model.lower() == 'ures':
        model = UnetResNet(in_channels=3, out_channels=1).to(DEVICE)
    elif args.model.lower() == 'fpn':
        model = FPN(in_channels=3, out_channels=1).to(DEVICE)
    else:
        raise NotImplementedError

    # 损失函数: iou loss / bce with dice loss
    if args.loss.lower() == 'iou':
        loss_fn = IoULoss().to(DEVICE)
    elif args.loss.lower() == 'dice':
        loss_fn = BCEDiceLoss().to(DEVICE)
    else:
        raise NotImplementedError
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)    
    
    # 加载数据
    train_set = KITTIDataset(data_path=args.data_path, split='train', img_size=args.img_size)
    val_set   = KITTIDataset(data_path=args.data_path, split='val', img_size=args.img_size, aug=False)
    test_set  = KITTIDataset(data_path=args.data_path, split='test', img_size=args.img_size, aug=False)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    start_epoch = -1
    history = defaultdict(list)
    best_loss = math.inf
    best_epoch = -1

    # 加载模型
    if args.resume_path:
        state = torch.load(args.resume_path, map_location='cpu')
        start_epoch = state['epoch']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optim'])
        history = state['history']
        best_loss = np.min(np.array(history['val_loss']))

    # 训练
    for epoch in range(start_epoch+1, args.epoch):
        train_loss = train_epoch(model,train_dataloader,loss_fn, optimizer, DEVICE)
        val_loss = eval_model(model,val_dataloader,loss_fn, DEVICE)        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch}/{args.epoch}, Train loss {train_loss}, Val loss {val_loss}')

        if (epoch+1) % args.print_freq == 0:
            plot_loss(history, args.name)

        # 最佳结果
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            print(f'best epch: {best_epoch}')
            torch.save({
                'epoch': best_epoch,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'history': history,
                }, os.path.join(args.name, 'best_model_state.bin'))
        
        # 最后结果
        if (epoch+1) % args.save_freq == 0:
            torch.save({
                'epoch': args.epoch,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'history': history,
                }, os.path.join(args.name, 'latest_model_state.bin'))
            np.save(os.path.join(args.name, 'train_loss.npy'), np.array(history['train_loss']))
            np.save(os.path.join(args.name, 'val_loss.npy'), np.array(history['val_loss']))

    print(f'best epch: {best_epoch}')