import torch
import os
import cv2
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize


def round_to_nearest_multiple(x, p):
    return int(((x - 1) // p + 1) * p)


def read_image(path):

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    return img


class TYJT(Dataset):
    """
    The Dataset folder is assumed to follow the following structure. In the given root folder, there must be two
    sub-folders:
    - fishyscapes_lostandfound: contains the mask labels.
    - laf_images: contains the images taken from the Lost & Found Dataset
    """

    def __init__(self, path, transforms):
        super().__init__()

        #self.hparams = hparams
        self.transforms = transforms

        self.images = []
        #self.labels = []

        #labels_path = os.path.join(
        #    hparams.dataset_root, 'fishyscapes_lostandfound')
        img_files = os.listdir(path)
        #label_files.sort()
        for img in img_files:
            #print(lbl)
            #self.labels.extend([os.path.join(labels_path, lbl)])
            #img_name = lbl
            self.images.extend([os.path.join(path, img)])

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = read_image(self.images[index])
        #label = read_image(self.labels[index])

        #label = label[:, :, 0]
        
        aug = self.transforms(image=image, mask=image)
        image = aug['image']
        label = aug['mask']

        return image,image# None#, label.type(torch.LongTensor)

    def __len__(self):
        return self.num_samples

