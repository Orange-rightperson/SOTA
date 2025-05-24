import os
import torch
from PIL import Image
from collections import namedtuple
import numpy as np

class Fishyscapes(torch.utils.data.Dataset):
    FishyscapesClass = namedtuple('FishyscapesClass', ['name', 'id', 'train_id', 'hasinstances',
                                                       'ignoreineval', 'color'])
    # --------------------------------------------------------------------------------
    # A list of all Lost & Found labels
    # --------------------------------------------------------------------------------
    labels = [
        FishyscapesClass('in-distribution', 0, 0, False, False, (144, 238, 144)),
        FishyscapesClass('out-distribution', 1, 1, False, False, (255, 102, 102)),
        FishyscapesClass('unlabeled', 2, 255, False, True, (0, 0, 0)),
    ]

    train_id_in = 0
    train_id_out = 1
    num_eval_classes = 20
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, split='Static', root="", transform=None):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.split = split  # ['Static', 'LostAndFound']
        self.images = []  # list of all raw input images
        self.targets = []  # list of all ground truth TrainIds images
        self.segs = []
        filenames = os.listdir(os.path.join(root, self.split, 'original'))
        root = os.path.join(root, self.split)
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.png':
                f_name = os.path.splitext(filename)[0]
                filename_base_img = os.path.join("original", f_name)
                filename_base_labels = os.path.join("labels", f_name)
                filename_base_seg = os.path.join("/home/gyang/Codes/RPL/data/fishyscapes/Static_mix/labels/", f_name+'.png')
                self.images.append(os.path.join(root, filename_base_img + '.png'))
                self.targets.append(os.path.join(root, filename_base_labels + '.png'))
                self.segs.append(filename_base_seg)
        self.images = sorted(self.images)
        self.targets = sorted(self.targets)
        self.segs = sorted(self.segs)

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.segs[i]).convert('L')
        #seg = Image.open(self.segs[i]).convert('L')
        #target = np.array(target)
        #seg = np.array(seg)
        #seg[target == 1] = 20
        #target = Image.fromarray(seg)
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'LostAndFound Split: %s\n' % self.split
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()

