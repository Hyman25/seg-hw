import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from utils import AUG_SIZE


class KITTIDataset(Dataset):
    def __init__(self, data_path, split='train', img_size=256, aug=False):
        
        self.imgs_path = os.path.join(data_path, 'image_2')
        self.mask_path = os.path.join(data_path, 'gt_image_2')
        
        self.img_size = img_size
        self.aug = aug
        self.aug_size = AUG_SIZE if aug else 1

        self.imgs = sorted(os.listdir(self.imgs_path))
        self.mask = sorted(os.listdir(self.mask_path))
        self.mask = [t for t in self.mask if 'road' in t]

        train_size = int(len(self.imgs) * 0.7)
        val_size   = int(len(self.imgs) * 0.1)
        test_size  = len(self.imgs) - train_size - val_size
        
        if split=='train':
            self.imgs = self.imgs[:train_size]
            self.mask = self.mask[:train_size]
        elif split=='val':
            self.imgs = self.imgs[train_size: train_size+val_size]
            self.mask = self.mask[train_size: train_size+val_size]
        elif split=='test':
            self.imgs = self.imgs[-test_size:]
            self.mask = self.mask[-test_size:]

        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5,border_mode=cv2.BORDER_CONSTANT),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.RGBShift(r_shift_limit=1, g_shift_limit=1, b_shift_limit=1, p=0.5),
        ])
    
    def augment(self, img, mask):
        aug = self.augmentation_pipeline(image=img, mask=mask)
        return aug['image'], aug['mask']

    def convert_to_binary_mask(self, mask):
        road_label = np.array([255, 0, 255]) # 标注路面RGB值
        cond = np.all(mask == road_label, axis = 2)
        mask = mask * cond[..., np.newaxis]
        mask = np.dot(mask[..., :3], [0.2989, 0.5870, 0.1140])
        # img = Image.fromarray(mask.astype(np.uint8))
        # img.save('testmask.png')
        mask = np.expand_dims(mask, axis=-1)
        mask[mask != 0.0]=1.0
        return mask

    def __len__(self):
        return len(self.imgs) * self.aug_size
    
    def __getitem__(self, index):
        index = index // self.aug_size
        
        img  = Image.open(os.path.join(self.imgs_path, self.imgs[index])).resize((self.img_size, self.img_size))
        img  = np.array(img).astype(np.float32) / 255.0
        mask = Image.open(os.path.join(self.mask_path, self.mask[index])).resize((self.img_size, self.img_size))
        mask = self.convert_to_binary_mask(np.array(mask)).astype(np.float32)

        if self.aug:
            img, mask = self.augment(img, mask)

        img  = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        return img, mask
    
# test
if __name__ == '__main__':
    a = KITTIDataset('/mnt/diskc/hh/datasets/KITTI_Road/training')
    b = DataLoader(a, batch_size=4)
    print(len(a))
    for img, mask in b:
        print(img.size(), mask.size())
        break