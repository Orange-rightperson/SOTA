import torch
import numpy as np
from PIL import Image as PILImage
import torch.nn.functional as F
import random
from pytorch_lightning import LightningModule
from contextlib import nullcontext
from omegaconf import OmegaConf
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from sklearn.cluster import KMeans
class NLSSampler:
    def __init__(self,  thresh, ignore_mask, method="fps", num_points=50):
        self.method = method
        self.num_points = num_points
        self.masked = True
        self.ignore_mask = None
        if ignore_mask:
            self.ignore_mask = torch.load(ignore_mask)
        self.uncertainty_thresh = thresh

    def __call__(self, x, uncert, inlier_output):
        ood_query_coords = list()

        uncertainties = []
        road_prediction = []
        road_prediction.append((inlier_output.max(0)[1] == 0).unsqueeze(0))
        uncertainties.append(uncert)
        # print(uncert[0].shape)
        # exit()
        uncert = uncert[0]
        #thresh = torch.min(uncert) + 0.6*(torch.max(uncert) - torch.min(uncert))
        #print(self.ignore_mask.shape)
        if self.ignore_mask is not None:
            uncert = torch.where(
                ~self.ignore_mask.to(uncert.device),
                uncert,
                torch.full_like(uncert, -30),
            )
        sample_idx = torch.nonzero(uncert >= self.uncertainty_thresh)
        if sample_idx.sum() < 1:
            sample_idx = torch.nonzero(uncert >= -15)
        sample_idx = torch.flip(sample_idx, [1])
        padding = torch.ones_like(sample_idx[:, :1])
        sample_idx = torch.cat((sample_idx, padding), dim=1)
        if self.method == "fps":
            sample_idx = sample_idx.float().contiguous().cuda()
            fps_idx = furthest_point_sample(sample_idx[None, ...], self.num_points)[
                0
            ].long()
            sample_idx = sample_idx[fps_idx].unsqueeze(1)
        elif self.method == "kmeans":
            predictor = KMeans(n_clusters=self.num_points).fit(sample_idx.cpu())
            sample_idx = (
                torch.Tensor(predictor.cluster_centers_).long().unsqueeze(1).cuda()
            )
        elif self.method == "concon":
            nls_image = (uncert.cpu().numpy() >= self.uncertainty_thresh).astype(
                np.uint8
            )
            num_labels, labels = cv2.connectedComponents(image=nls_image)
            sample_idx = list()
            for each_label in range(1, num_labels):
                sample_idx.append(labels == each_label)
            sample_idx = np.stack(sample_idx)
        elif self.method == "random":
            sample_idx = sample_idx[
                np.random.choice(
                    list(range(len(sample_idx))),
                    size=self.num_points,
                    replace=False,
                )
            ].unsqueeze(1)

        ood_query_coords.append(sample_idx)
            # self.current_step += 1
        uncertainties = torch.cat(uncertainties, dim=0)
        return ood_query_coords, uncertainties, torch.cat(road_prediction, dim=0)
