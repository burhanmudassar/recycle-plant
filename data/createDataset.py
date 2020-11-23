import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from torchvision.transforms import transforms
from torchvision.transforms.transforms import RandomAffine, RandomPerspective, RandomRotation, RandomVerticalFlip, ToPILImage

CLASSES = ("glass", "paper", "cardboard", "plastic", "metal", "trash")

testTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(104, 117, 123), std=(57.4, 57.1, 58.4)),
     
])

augmentTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=5, shear=10),
        transforms.RandomRotation(degrees=[90,270]),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(),
        # transforms.RandomCrop((300,300)),
])

trainTransform = transforms.Compose([
        *augmentTransform.transforms,
        *testTransform.transforms[1:] # Exclude PIL image conversion
        # transforms.Resize((224, 224)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=(104, 117, 123), std=(57.4, 57.1, 58.4)),
])



class TrashData(Dataset):

    def __init__(self, root, set, transform=None):
        self.root = root
        self.path = os.path.join(self.root, 'dataset-resized')
        self.nb_classes = len(CLASSES)
        self.transform = transform
        self.ids, self.targets = self.readsplitfile(set)

        print(f"Count of classes in {set}")
        for clsId, cls in enumerate(CLASSES):
            print(f"{cls} : {np.sum(self.targets == clsId)}")

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.ids)

    def readsplitfile(self, set='train'):
        filepaths = []
        classes = []
        with open(os.path.join(self.root, set+'list.txt'), 'r') as f:
            for x in f.readlines():
                line = x.split()
                f_ = line[0]
                c_ = int(line[1])
                filepaths += [os.path.join(self.path, CLASSES[c_-1], f_)]
                classes += [c_-1]

        return filepaths, np.array(classes)

    def getitem(self,index):
        path = self.ids[index]
        target = self.targets[index]

        im = cv2.imread(path)[:,:,::-1]
        if im is None:
            print("Invalid path")
            raise ValueError

        if self.transform:
            im = self.transform(im)

        if isinstance(im, torch.Tensor):
            return im, target
        else:
            return np.asarray(im), target



