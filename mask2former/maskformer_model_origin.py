# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
#from .modeling.sam_lora import LoRA_Sam
import torch
import cv2 
import numpy as np
from torch import nn
from torch.nn import functional as F
import cv2
import numpy as np
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from torchvision import transforms
from .modeling.criterion import SetCriterion
from .modeling.sam_matcher import HungarianMatcher, FixedMatcher
from .modeling.sampler import NLSSampler
from .modeling.segment_anything import SamPredictor, sam_model_registry
from .modeling.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from .modeling.sfg import SelectiveFusionGate
from .modeling.sfg_pormpt import SelectiveFusionGatePrompt
from .modeling.sam_lora import LoRA_Sam
from copy import deepcopy
from scipy.ndimage import distance_transform_edt





class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
class ConvNet(nn.Module):
    def __init__(self, mask_in_chans, embed_dim):
        super(ConvNet, self).__init__()
        # 定义卷积层，将通道数从 20 -> 256，尺寸从 256x512 -> 64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

    def forward(self, x):
        x = self.conv1(x)  # 第一次卷积
        return x


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        force_region_partition: bool,
        outlier_supervision: bool,
        open_panoptic: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(
            pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(
            pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.force_region_partition = force_region_partition
        self.outlier_supervision = outlier_supervision
        self.open_panoptic = open_panoptic
        self.ignore_mask_path = "ckpts/ignore_mask.pth"
        self.test_proposal_sampler=NLSSampler(thresh=-0.15, ignore_mask=None, method="fps", num_points=50)
       # self.test_proposal_sampler_1=NLSSampler(thresh=-0.15, ignore_mask=None, method="kmeans", num_points=50)
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference
        sam_checkpoint = "ckpts/sam_vit_h_4b8939.pth"
        sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.set_requires_grad(sam_model, False)
        self.sam = LoRA_Sam(sam_model, 512).sam
        #self.im_encoder = LoRA_Sam(deepcopy(sam_model), 512).lora_vit
        self.predictor = SamPredictor(self.sam)
        self.automask_generator = SamAutomaticMaskGenerator(
            model=self.sam, min_mask_region_area=10
        )   
        self.nms_thresh = 0.7
        self.ignore_mask =None# torch.load(self.ignore_mask_path).cuda()
        
        toPIL = transforms.ToPILImage()
        self.minimal_uncert_value_instead_zero = -1.0

        #self.reshape_net = ConvNet()
        sfg_feat_num =2
        sfg_filter_num = 1
        sfg_intermediate_channels = 32
        sfg_filter_type = "conv2d"
        self.fusion_module = SelectiveFusionGate(
            feat_num = sfg_feat_num,
            filter_num=sfg_filter_num,
            intermediate_channels=sfg_intermediate_channels,
            filter_type=sfg_filter_type
        )
        self.prompt_fusion = SelectiveFusionGate(
            feat_num = sfg_feat_num,
            filter_num=sfg_filter_num,
            intermediate_channels=sfg_intermediate_channels,
            filter_type=sfg_filter_type
        )#SelectiveFusionGatePrompt(
        #                    feat_num = sfg_feat_num,
        #                    filter_num=sfg_filter_num,
        #                    intermediate_channels=sfg_intermediate_channels,
        #                    filter_type=sfg_filter_type
        #)
        mask_in_chans = 16
        embed_dim = 256
        self.num = 2
        self.mask_downscaling = ConvNet(mask_in_chans, embed_dim)#deepcopy(self.sam.prompt_encoder.mask_downscaling)
        self.ood_attention = ConvNet(mask_in_chans, embed_dim)#deepcopy(self.sam.prompt_encoder.mask_downscaling)
        #self.road_attention = ConvNet(mask_in_chans, embed_dim)
        #self.no_mask_embed = nn.Embedding(1, embed_dim)
        self.road_attention = deepcopy(self.sam.prompt_encoder.mask_downscaling)
        #self.road_embedding = [nn.Embedding(1, embed_dim) for i in range(self.num)]
        #self.road_attention = nn.Sequential(
        #    nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
        #    LayerNorm2d(mask_in_chans // 4),
        #    nn.GELU(),
        #    nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
        #    LayerNorm2d(mask_in_chans),
        #    nn.GELU(),
        #    nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
            #nn.AdaptiveAvgPool2d((self.num,self.num))
            #nn.Flatten(2)
        #)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        ood_weight = 1.0
        contrastive_weight = 1.0
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        # object_weight = cfg.MODEL.MASK_FORMER.OBJECT_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        smoothness_weight = cfg.MODEL.MASK_FORMER.SMOOTHNESS_WEIGHT
        sparsity_weight = cfg.MODEL.MASK_FORMER.SPARSITY_WEIGHT
        gambler_weight = cfg.MODEL.MASK_FORMER.GAMBLER_WEIGHT
        outlier_weight = cfg.MODEL.MASK_FORMER.OUTLIER_WEIGHT
        densehybrid_weight = cfg.MODEL.MASK_FORMER.DENSE_HYBRID_WEIGHT
        # building criterion
        matcher_type = cfg.MODEL.MASK_FORMER.MATCHER
        if matcher_type == "HungarianMatcher":
            matcher = HungarianMatcher(
                cost_class=class_weight,
                # cost_object=object_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )
        elif matcher_type == "FixedMatcher":

            if sem_seg_head.num_classes != cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES:
                raise ValueError(
                    "When using FixedMatcher, number of object queries must be equal to number of classes")

            matcher = FixedMatcher()

        else:
            raise ValueError(f"Given Matcher ({matcher_type}) is not defined")

        weight_dict = {
            "sam_c_loss": 1.0,
            "sam_iou_loss": 1.0, 
            "sam_bce_loss": 1.0,
            "sam_dice_loss": 1.0,
            "loss_obj_mask": 1.0,
            "loss_obj_dice":1.0,
            "loss_obj_ce":1.0,
            "contrastive_loss": contrastive_weight,
            "ood_mse_loss": ood_weight,
            "loss_ce": class_weight,
            # "loss_object":object_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "smoothness_loss": smoothness_weight,
            "sparsity_loss": sparsity_weight,
            "outlier_loss": outlier_weight,
            "gambler_loss": gambler_weight,
            "densehybrid_loss": densehybrid_weight
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses =[ "sam_bce", "sam_dice"]#, "sam_c"]#[ "labels", "objects" ,"masks"]#, "obj_mask", "obj_labels"]["contrastive",

        if cfg.MODEL.MASK_FORMER.GAMBLER_LOSS:
            losses = ["gambler"]
        if cfg.MODEL.MASK_FORMER.DENSE_HYBRID_LOSS:
            losses = ["densehybrid"]
        if cfg.MODEL.MASK_FORMER.SMOOTHNESS_LOSS:
            losses.append("smoothness")
        if cfg.MODEL.MASK_FORMER.SPARSITY_LOSS:
            losses.append("sparsity")
        #if cfg.MODEL.MASK_FORMER.OUTLIER_SUPERVISION:
        #    losses.append("outlier")
        
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            smoothness_score=cfg.MODEL.MASK_FORMER.SMOOTHNESS_SCORE,
            outlier_loss_target=cfg.MODEL.MASK_FORMER.OUTLIER_LOSS_TARGET,
            inlier_upper_threshold=cfg.MODEL.MASK_FORMER.INLIER_UPPER_THRESHOLD,
            outlier_lower_threshold=cfg.MODEL.MASK_FORMER.OUTLIER_LOWER_THRESHOLD,
            score_norm=cfg.MODEL.MASK_FORMER.SCORE_NORM,
            outlier_loss_func=cfg.MODEL.MASK_FORMER.OUTLIER_LOSS_FUNC,
            pebal_reward=cfg.MODEL.MASK_FORMER.PEBAL_REWARD,
            ood_reg=cfg.MODEL.MASK_FORMER.PEBAL_OOD_REG,
            densehybrid_beta=cfg.MODEL.MASK_FORMER.DENSE_HYBRID_BETA,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "force_region_partition": cfg.SOLVER.FORCE_REGION_PARTITION,
            "outlier_supervision": cfg.MODEL.MASK_FORMER.OUTLIER_SUPERVISION,
            "open_panoptic": cfg.MODEL.MASK_FORMER.OPEN_PANOPTIC,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, include_void=False, return_separately=False, 
                return_aux=False, return_ood_pred=False, panoptic_ood_threshold=-0.3, 
                panoptic_pixel_min=200, return_panoptic_ood=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        #gt = [x["instances"].to(self.device) for x in batched_inputs]
        #gt = torch.stack(gt, dim=0)
        #gt = torch.where(gt == 255, 0, gt)
        #gt = retry_if_cuda_oom(sem_seg_postprocess)(
        #                                    gt.float(), (gt.shape[-2], gt.shape[-1]), 1024, 2048)
        #print(gt.shape)
        #print(torch.unique(gt))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        reshape_images = F.interpolate(
                                images.tensor,
                                size=(1024, 2048),
                                mode="bilinear",
                                align_corners=False,
                                      )
        #reshape_images = images.tensor
        features = self.backbone(reshape_images)
        outputs = self.sem_seg_head(features)
        # print(outputs['pred_oods'].shape)
        # print((outputs['aux_outputs']))
        # exit()
        if self.force_region_partition:
            B, N, H, W = outputs["pred_masks"].shape
            outputs["pred_masks"] = outputs["pred_masks"].softmax(dim=1)
            # outputs["pred_ood_masks"]= outputs["pred_ood_masks"].softmax(dim=1)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(
                    self.device) for x in batched_inputs]
                if self.outlier_supervision:
                    outlier_masks = [x["outlier_mask"].to(self.device) for x in batched_inputs]
                    sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
                    #ood_map = [x["ood_map"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images, outlier_masks, sem_seg)
                else:
                    targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            mask_cls_results = outputs["pred_logits"] # B x N x K
            mask_pred_results = outputs["pred_masks"] # B x N x H x W
            sem  = retry_if_cuda_oom(self.batched_semantic_inference)(
                                                mask_cls_results, mask_pred_results)
            ood = retry_if_cuda_oom(self.batched_nls_ood_inference)(
                                                mask_cls_results, mask_pred_results)
            ood_train = F.interpolate(
                ood,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
            sem_train = F.interpolate(
                sem,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
            #ood = ood.unsqueeze( 
            #for i in range(ood_train.shape[0]):
            #    ood_train[i] = ood_train[i] - torch.min(ood_train[i])
            #print(torch.min(ood))
            #print(torch.max(ood))
            inlier = sem
            #con_map = []
            f_feature = []
            #con_map.append(ood_train)
            #for i in range(sem_train.shape[1]):
            #    con_map.append(sem_train[:,i,...].unsqueeze(0))
            #for i in range(len(con_map)):

            f_feature.append(self.mask_downscaling(ood_train))


            #f_feature.append(self.mask_downscaling_id(sem_train))

            #print(f_feature[0].shape)
            #print(f_feature[1].shape)
            #print(inlier.shape)
            #print(ood.shape)
            #con_map = torch.cat([sem_train,ood_train], dim=1)
            #prob_sum = con_map.max(dim=1, keepdim=True)  # 计算每个像素位置的概率和, shape: [1, H, W]
                                    # Step 2: 对每个类别的概率进行归一化处理
            #con_map = con_map #/ prob_sum
            #con_map = ood_train.view(1, 256, 64, 64, 1)
            #con_map = self.sam.prompt_encoder.mask_downscaling(con_map)
            ood = retry_if_cuda_oom(sem_seg_postprocess)(
               ood, (ood.shape[-2], ood.shape[-1]), 1024, 2048)

            inlier = retry_if_cuda_oom(sem_seg_postprocess)(
	        inlier, (inlier.shape[-2], inlier.shape[-1]), 1024, 2048)
            #images_t =  F.interpolate(
	    #images.tensor,
	    #size=(1024, 2048),
	    #mode="bilinear",
	    #align_corners=False,
	    #)
            images_t = reshape_images
            #ood_masks = torch.stack(outlier_masks, dim = 0)
            #ood_masks = retry_if_cuda_oom(sem_seg_postprocess)(
            #                       ood_masks.float(), (ood_masks.shape[-2], ood_masks.shape[-1]), 1024, 2048)
            #ood_map = torch.stack(ood_map, dim = 0)
            #ood_map = retry_if_cuda_oom(sem_seg_postprocess)(
            #                    ood_map.float(), (ood_map.shape[-2], ood_map.shape[-1]), 1024, 2048)
            points, train_masks = self.fine_with_sam_train(images_t,  ood, inlier, f_feature)
            train_mask = train_masks[0].squeeze(1)
            #train_mask = torch.sum(train_masks[0], dim=0)
            train_mask = retry_if_cuda_oom(sem_seg_postprocess)(
                        train_mask.float(), (1024,2048), images.tensor.shape[-2] ,images.tensor.shape[-1])
            points = self.app_coords(points, images.tensor.shape[-2], images.tensor.shape[-1] )

            pred = {"train_masks": train_mask, "points": points}
            # bipartite matching-based loss
            #print(points)
            losses = self.criterion(outputs, targets,  pred)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]            
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            # mask_obj_results = outputs["pred_object_logits"]
            mask_cls_results = outputs["pred_logits"] # B x N x K
            mask_pred_results = outputs["pred_masks"] # B x N x H x W
            sem = retry_if_cuda_oom(self.batched_semantic_inference)(
                                                mask_cls_results, mask_pred_results)
            ood = retry_if_cuda_oom(self.batched_nls_ood_inference)(
                                                mask_cls_results, mask_pred_results)
            #for i in range(ood.shape[0]):
            #    ood[i] = ood[i] - torch.min(ood[i], dim=0)[0]
            ood_train = F.interpolate(
                ood,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
            sem_train = F.interpolate(
                sem,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
 
            #for i in range(ood.shape[0]):
            #    ood_train[i] = ood_train[i] - torch.min(ood_train[i])
            #sem = sem
            #con_map = torch.cat([sem_train,ood_train], dim=1)
            #prob_sum = con_map.sum(dim=1, keepdim=True)  # 计算每个像素位置的概率和, shape: [1, H, W]
            # Step 2: 对每个类别的概率进行归一化处理
		
            con_map = []
            f_feature = []
            #con_map.append(ood_train)
            #for i in range(sem_train.shape[1]):
            #    con_map.append(sem_train[:,i,...].unsqueeze(0))
            #for i in range(len(con_map)):
            #    o_feature.append(self.sam.prompt_encoder.mask_downscaling(con_map[i]))
            f_feature.append(self.mask_downscaling(ood_train))
            #f_feature.append(self.mask_downscaling_id(sem_train))
            #con_map = ood_train.view(1, 256, 64, 64, 1)
            #con_map = con_map.unsqueeze(0)
            #con_map = self.reshape_net(con_map)
            # mask_ood_cls_results = outputs["pred_o_logits"]
            # mask_ood_pred_results = outputs["pred_o_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            # mask_ood_pred_results = F.interpolate(
            #     mask_ood_pred_results,
            #     size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            #     mode="bilinear",
            #     align_corners=False,
            # )
            if return_aux:
                aux = outputs["aux_outputs"]

            if return_ood_pred:
                ood_pred = outputs["ood_pred"]
                ood_pred = F.interpolate(ood_pred, size=images.image_sizes[0], mode='bilinear', align_corners=True)
            del outputs

            processed_results = []
            for mask_cls_result,   mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
          
                    # mask_ood_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    #     mask_ood_pred_result, image_size, height, width
                    # )
                    



                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                    mask_obj_result = mask_obj_result.to(mask_pred_result)
                    # mask_ood_cls_result = mask_ood_cls_result.to(mask_ood_pred_result)
                # semantic segmentation inference
                # index = 0
                # for i in mask_ood_pred_result:
                #     pred_image_path = "/home/gyang/Codes/RbA/view/{}.jpg".format( mask_ood_cls_result[index].sigmoid())
                #     view = torch.where(i>0.3, 1, 0)
                #     view = view.cpu().numpy() * 255
                #     cv2.imwrite(pred_image_path , view)
                #     index = index + 1
                if self.semantic_on:
                    if include_void:
                        r = retry_if_cuda_oom(self.semantic_inference_with_void)(
                            mask_cls_result, mask_pred_result)
                        # o = retry_if_cuda_oom(self.semantic_inference_with_void)(
                        #     mask_obj_result, mask_pred_result)
                        ood = retry_if_cuda_oom(self.nls_ood_inference)(
                            mask_cls_result, mask_pred_result
                        )
                        # ood = retry_if_cuda_oom(self.object_inference)(
                        #     mask_ood_cls_result, mask_ood_pred_result)
                    else:
                        r = retry_if_cuda_oom(self.semantic_inference)(
                            mask_cls_result, mask_pred_result)
                        # o = retry_if_cuda_oom(self.semantic_inference)(
                        #     mask_obj_result, mask_pred_result)
                        ood = retry_if_cuda_oom(self.nls_ood_inference)(
                            mask_cls_result, mask_pred_result
                        )
                        # ood = retry_if_cuda_oom(self.object_inference)(
                        #     mask_ood_cls_result, mask_ood_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(
                            r, image_size, height, width)
                        # o = retry_if_cuda_oom(sem_seg_postprocess)(
                        #     o, image_size, height, width)
                    ood = retry_if_cuda_oom(sem_seg_postprocess)(
                            ood, image_size, 1024, 2048)
                    inlier = retry_if_cuda_oom(sem_seg_postprocess)(
                            r, image_size, 1024, 2048)

                    processed_results[-1]["sem_seg"] = r
                    # processed_results[-1]["sem_seg_o"] = o
                    # processed_results[-1]["sem_seg_ood"] = ood
                    images_t =  F.interpolate(
                        images.tensor,
                        size=(1024, 2048),
                        mode="bilinear",
                        align_corners=False,
                    )

                    #ood = gt            
                    fine_ood, points, mask, train_masks = self.fine_with_sam(images_t, ood, inlier, f_feature)
                    points = self.app_coords(points, height, width)
                    fine_ood = retry_if_cuda_oom(sem_seg_postprocess)(
                            fine_ood, (1024,2048), height ,width)
                    mask = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask.float(), (1024,2048), height ,width)
                    # print(fine_ood.shape)
                    processed_results[-1]["sem_seg_ood"] = fine_ood
                    processed_results[-1]["points"] = points
                    processed_results[-1]["sam_masks"] = train_masks

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result, self.open_panoptic, 
                        panoptic_ood_threshold, panoptic_pixel_min, return_panoptic_ood)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            if return_aux:
                return processed_results, aux
            if return_ood_pred:
                return processed_results, ood_pred

            if return_separately:
                return processed_results, mask_cls_result, mask_pred_result

            return processed_results
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        for name, param in nets.named_parameters():
            param.requires_grad = requires_grad
            #if name == "_encoder":
             #   print(name)
             #   param.requires_grad = True

        #for param in nets.prompt_encoder.parameters():
        #    param.requires_grad = True
        #for name, param in nets.prompt_encoder.mask_downscaling.named_parameters():
        #    param.requires_grad = True
        #    if name == "2.weight" or name == "3.weight":
        #        param.requires_grad = False
        #for name, param in nets.prompt_encoder.not_a_point_embed.named_parameters():
        #    param.requires_grad = True


    def prepare_targets(self, targets, images, outlier_masks=None, sem_seg=None):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for i, targets_per_image in enumerate(targets):
            # pad gt
            gt_masks = targets_per_image.gt_masks


            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1],
                         : gt_masks.shape[2]] = gt_masks
            
            entry = {
                "object_labels": torch.zeros_like(targets_per_image.gt_classes),  # class-agnostic
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks
            }
            # if padded_masks.shape[0] != 0:
            #     ood_mask = padded_masks[0]
            #     for i in range(len(padded_masks)):
            #         ood_mask = ood_mask | padded_masks[i]
            #     ood_mask = ~ood_mask
            # else:
            #     ood_mask = torch.ones(size=(gt_masks.shape[1], gt_masks.shape[2])).to(padded_masks)


            # img = transforms.ToPILImage()(ood_mask.float())
            # img.save("test.jpg")
            # exit(


            # entry["ood_mask"] = outlier_masks[i]
            if outlier_masks is not None:
                entry["outlier_masks"] = outlier_masks[i]
                #print(outlier_masks[i].shape)
                #print(torch.unique(outlier_masks[i]))
                #exit()
                #print(torch.unique(outlier_masks[i]))
                #print(outlier_masks[i].shape)
                # gt_objects = torch.cat((gt_masks, gt_masks),0)
                # gt_objects = torch.cat((gt_masks, outlier_masks[i].bool().unsqueeze(0)),0)
                # padded_objects = torch.zeros((gt_objects.shape[0], h_pad, w_pad), dtype=gt_objects.dtype, device=gt_objects.device)
                # padded_objects[:, : gt_objects.shape[1],
                #          : gt_objects.shape[2]] = gt_objects
                # entry["objects"] = padded_objects
                # entry["objects_s"] = torch.ones(padded_objects.shape[0], dtype=torch.int64).to('cuda')
            if sem_seg is not None:
                entry["sem_seg"] = sem_seg[i]
                #print(torch.unique(sem_seg[i]))
            new_targets.append(entry)
                
        return new_targets



    def batched_mask_nms(self, masks, logits, iou_threshold):
        sorted_indices = torch.sort(logits, descending=True)[1]
        removed_mask_indices = list()
        num_classes = 2  # predicted / not predicted
        # going from the most confident mask compute iou with other masks
        for highest_score_index in sorted_indices:
            if highest_score_index in removed_mask_indices:
                # if mask already removed - don't check it
                continue
            target_mask = masks[highest_score_index].view(-1)
            # search only masks that were not removed
            remaining_masks = list(set(range(len(masks))) - set(removed_mask_indices))
            for mask_index in remaining_masks:
                if highest_score_index == mask_index:
                    # do not compare the mask with itself
                    continue
                x = masks[mask_index].view(-1) + num_classes * target_mask
                bincount_2d = torch.bincount(x, minlength=num_classes**2)

                true_positive = bincount_2d[3]
                false_positive = bincount_2d[1]
                false_negative = bincount_2d[2]
                # Just in case we get a division by 0, ignore/hide the error
                # no intersection
                if true_positive == 0:
                    continue
                # intersection of a source mask with the target mask
                iou = true_positive / (true_positive + false_positive + false_negative)
                if iou > iou_threshold:
                    removed_mask_indices.append(mask_index)
        resulting_masks = list(set(range(len(masks))) - set(removed_mask_indices))
        return resulting_masks


    def app_coords(self, coords, h, w):
        old_h, old_w = 1024, 2048
        new_h, new_w = h, w
        coords = deepcopy(coords).float()
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return torch.round(coords)
    def scales_tensor(self, tensor_A, tensor_B):
        # 获取 tensor_A 和 tensor_B 的最小值和最大值
        min_A, max_A = tensor_A.min(), tensor_A.max()
        #min_B, max_B = tensor_B.min(), tensor_B.max()
        thres = min_A + 0.7*(max_A-min_A)
        # 线性变换公式
        #tensor_A_scaled = (tensor_A - min_A) / (max_A - min_A)  # 将 tensor_A 归一化到 [0, 1]
        #tensor_A_scaled = tensor_A_scaled * (max_B - min_B) + min_B  # 映射到 tensor_B 的范围
        tensor_A = (tensor_A > thres).float()
        return tensor_A * tensor_B

    def compute_distance_map_gpu(self, road_mask):
        distance_map = torch.full_like(road_mask, float('inf'))
        distance_map[road_mask == 1] = 0
        kernel = torch.tensor([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=torch.float32, device=road_mask.device).unsqueeze(0).unsqueeze(0)
        distance_map_up = F.conv2d(distance_map, kernel, padding=1)
    
        distance_map_down = F.conv2d(distance_map, kernel.flip([2, 3]), padding=1)

        distance_map = torch.minimum(distance_map_up, distance_map_down)

        return distance_map

    def sam_output(self, x, query_coordinates,  uncertainty, o_feature, road_prediction):
        self.predictor.set_torch_image(x, original_image_size=(1024, 2048))
        feature = self.predictor.features
        #input_image = self.predictor.model.preprocess(x)
        #feature = self.predictor.model.image_encoder(input_image)
        #exit()
        def apply_coords(coords):
            old_h, old_w = 1024, 2048
            new_h, new_w = 512, 1024
            coords = deepcopy(coords).float()
            coords[..., 0] = coords[..., 0] * (new_w / old_w)
            coords[..., 1] = coords[..., 1] * (new_h / old_h)
            return coords

        masks = []
        scores = []
        nlses = []
        train_masks = []
        for i in range(x.shape[0]):
            #self.predictor.set_torch_image(x[i], original_image_size=(1024, 2048))
                            #feature = self.predictor.features
            #input_image = self.predictor.model.preprocess(x[i])
            #feature = self.predictor.model.image_encoder(input_image)
            nls = uncertainty[i]
            nlses.append(nls.cpu())
            points = apply_coords(query_coordinates[i]).cuda()
            points = (points[..., :2], points[..., -1])
            road = road_prediction[i]
            #road_mask = self.dilate_mask(road)
            road_mask =  F.interpolate(
            road.unsqueeze(0).unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )   


            #road_mask = (road_mask > 0.5).float()
            #inverted_mask = 1 - road_mask
             
            #distance_map = distance_transform_edt(inverted_mask.cpu().numpy())
            #distance_map = torch.tensor(distance_map, device="cuda")
            
            #distance_map = distance_transform_edt(inverted_mask)
            mask_prompt = F.interpolate(
            nls.unsqueeze(0).unsqueeze(0),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
            #mask_prompt = torch.where(mask_prompt > torch.min(mask_prompt)+ 0.8*(torch.max(mask_prompt) - torch.min(mask_prompt)), torch.tensor(1.0, device=nls.device), torch.tensor(False, device=nls.device) )
            (
                sparse_embeddings_orin,
                dense_embeddings_orin
            )   = self.predictor.model.prompt_encoder(
                boxes=None,
                points=None,
                masks=mask_prompt,
            )
            #dense_embeddings = dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            #                        x.shape[0], -1, self.predictor.model.prompt_encoder.image_embedding_size[0], self.predictor.model.prompt_encoder.image_embedding_size[1]
            #                                    )
            extra_prompt = self.scales_tensor(mask_prompt,road_mask)
            
            #road_mask = self.dilate_mask(road_mask)
            #mask_prompt = mask_prompt - torch.min(mask_prompt)
            ood_embeddings = self.ood_attention(mask_prompt)
            extra_embeddings = self.road_attention(extra_prompt)


            #print(torch.max(extra_embeddings))
            #print(torch.min(extra_embeddings))
            
            #extra_embeddings = ood_embeddings * extra_embeddings 
            dense_embeddings,_ = self.prompt_fusion(feat_list=[ood_embeddings, extra_embeddings])
            #print(torch.max(dense_embeddings))
            #print(torch.min(dense_embeddings))
           # print(torch.max(dense_embeddings))
           # print(torch.min(dense_embeddings))
           # dense_embeddings = nn.functional.gelu(dense_embeddings)
            #sparse_embeddings = sparse_embeddings.flatten(2).transpose(1, 2)
            #sparse_embeddings  = sparse_embeddings.view(1, 1, -1)
            #print(sparse)
            #weights = torch.cat([self.road_embedding[i].weight for i in range(self.num * self.num)], dim=0).to("cuda")
            #sparse_embeddings = sparse_embeddings + weights.unsqueeze(0)
            #sparse_embeddings = sparse_embeddings.repeat(1, 2, 1)
            #embed_list = []
            #embed_list.append(dense_embeddings1)
            #embed_list.append(dense_embeddings2)
            #print(sparse_embeddings.shape)
            #print(dense_embeddings.shape)
            #dense_embeddings = torch.cat(embed_list, dim=0)
            #dense_embeddings,_ = self.prompt_fusion(feat_list = embed_list)
            pe = self.predictor.model.prompt_encoder.get_dense_pe()
            sam_features = feature#self.predictor.features
            o_feature.insert(0, sam_features)
            #o_feature.insert(0, extra_embeddings)
            #print(self.road_embedding[0].weight.shape)
            #features = sam_features
            #print(feature.shape)
            features, _= self.fusion_module(feat_list=o_feature)
            # 获取掩码
            #mask_ge_0 = sparse_embeddings >= 0
            #mask_lt_0 = sparse_embeddings < 0

            # 处理形状
            #road_embedding_0 = self.road_embedding[0].weight.reshape(1, -1, 1, 1).expand(
            #            x.shape[0], -1, 
            #                self.predictor.model.prompt_encoder.image_embedding_size[0], 
            #                    self.predictor.model.prompt_encoder.image_embedding_size[1]
            #                    ).to(features.device)
            #road_embedding_1 = self.road_embedding[1].weight.reshape(1, -1, 1, 1).expand(
            #            x.shape[0], -1, 
            #                self.predictor.model.prompt_encoder.image_embedding_size[0], 
            #                    self.predictor.model.prompt_encoder.image_embedding_size[1]
            #                    ).to(features.device)

            # 按条件更新 features
            #features[mask_ge_0] += road_embedding_0[mask_ge_0]
            #features[mask_lt_0] += road_embedding_1[mask_lt_0]

            (
                low_res_masks,
                iou_predictions,
            ) = self.predictor.model.mask_decoder(
                image_embeddings=features[i : i + 1],#self.predictor.features[i : i + 1],
                image_pe=pe,
                sparse_prompt_embeddings=sparse_embeddings_orin,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            mask = self.predictor.model.postprocess_masks(
                low_res_masks,
                self.predictor.input_size,
                self.predictor.original_size,
            )
            train_masks.append(mask)
            best_masks = iou_predictions.max(1)[1]
            mask = mask[torch.arange(mask.shape[0]), best_masks].unsqueeze(1)
            iou_predictions = iou_predictions[
                torch.arange(mask.shape[0]), best_masks
            ].unsqueeze(1)
            mask = mask > self.predictor.model.mask_threshold
            masks.append(mask.flatten(0, 1))
            score = []
            for i in range(mask.shape[0]):
                nls_score = nls[mask.squeeze(1)[i]].mean()
                if nls_score.isnan():
                    nls_score = -torch.ones_like(nls_score)
                score.append(nls_score)
            score = torch.stack(score)
            scores.append(score)
        return {"pred_logits": scores, "pred_masks": masks, "uncert": nlses, "train_masks": train_masks}

    def dilate_mask(self, mask, kernel_size=15, iterations=2):
            # 创建卷积核
        mask = mask.unsqueeze(0)
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
        for _ in range(iterations):
                                # 使用卷积进行膨胀
            mask = F.conv2d(mask, kernel, padding=kernel_size // 2).clamp(0, 1)
            mask = (mask > 0).float()  # 保持浮点类型
        return mask

    def fine_with_sam_train(self, x, uncert, inlier_output, o_feature):
        (
            query_coordinates,
            nls_uncertainty,
            road_prediction,
        ) = self.test_proposal_sampler(x, uncert, inlier_output)
        points = []
        road_prediction = (inlier_output.max(0)[1] == 0).unsqueeze(0).float()
        #road_prediction += (inlier_output.max(0)[1] == 1).unsqueeze(0).float()
        road_prediction = self.dilate_mask(road_prediction).squeeze(0)
        #cert,_ = inlier_output.max(dim=0)
        #cert = cert.unsqueeze(0)
        #(
            #id_query_coordinates,
            #_,
            #_,
        #) = self.test_proposal_sampler_1(x, cert, inlier_output)
        #id_query_coordinates[0][:,:,2] = 0.0
        #query_coordinates[0] = torch.cat((query_coordinates[0], id_query_coordinates[0]),0)
        x = F.interpolate(
            x,
            size=(512, 1024),
            mode="bilinear",
            align_corners=False,
        )
        ood_output = self.sam_output(
            x,
            query_coordinates,
            nls_uncertainty,
            o_feature,
            road_prediction
        )
        for i in range(len(ood_output["train_masks"])):
            points.append(query_coordinates[i][:,:,:2])

        return torch.stack(points).cuda(), ood_output["train_masks"]




    def fine_with_sam(self, x, uncert, inlier_output, o_feature):
        (
            query_coordinates,
            nls_uncertainty,
            road_prediction,
        ) = self.test_proposal_sampler(x, uncert, inlier_output)
        road_prediction = (inlier_output.max(0)[1] == 0).unsqueeze(0).float()
        #road_prediction += (inlier_output.max(0)[1] == 1).unsqueeze(0).float()
        road_prediction = self.dilate_mask(road_prediction).squeeze(0)
        #print(nls_uncertainty.shape)
        #print(road_prediction.shape)
        #exit()
        #cert,_ = inlier_output.max(dim=0)
        #cert = cert.unsqueeze(0)
        #(
            #id_query_coordinates,
            #_,
            #_,
        #) = self.test_proposal_sampler_1(x, cert, inlier_output)
        #id_query_coordinates[0][:,:,2] = 0.0
        #query_coordinates[0] = torch.cat((query_coordinates[0], id_query_coordinates[0]),0)
        x = F.interpolate(
            x,
            size=(512, 1024),
            mode="bilinear",
            align_corners=False,
        )
        ood_output = self.sam_output(
            x,
            query_coordinates,
            nls_uncertainty,
            o_feature,
            road_prediction
        )
        uncertainty = list()
        points = list()
        instance_prediction = []
        for i in range(len(ood_output["pred_logits"])):
            logits = ood_output["pred_logits"][i]
            masks = ood_output["pred_masks"][i]

            keep_by_nms = self.batched_mask_nms(
                masks, logits, iou_threshold=self.nms_thresh
            )
            keep_by_nms = torch.Tensor(keep_by_nms).long()

            if self.ignore_mask is not None:
                keep_by_ignore = list()
                for j in keep_by_nms.tolist():
                    intersection = torch.sum(masks[j] * self.ignore_mask)
                    if intersection < 100:
                        keep_by_ignore.append(j)
            else:
                keep_by_ignore = keep_by_nms
            points.append(query_coordinates[i][:, :, :2])
            #query_coordinates[i] = query_coordinates[i][keep_by_ignore]

            #points.append(query_coordinates[i][:, :, :2])


            uncertainty.append(
                torch.einsum(
                    "q,qhw->hw",
                    logits[keep_by_ignore].cpu(),
                    masks[keep_by_ignore].float().cpu(),
                )
            )

            sam_mask = torch.sum(masks[keep_by_ignore],  dim=0)
            sam_mask = torch.where(sam_mask >=1 , 1, sam_mask).unsqueeze(0)
            # instance_prediction.append(
            #     {
            #         "scores": logits[keep_by_ignore].cpu().numpy()[..., None],
            #         "masks": masks[keep_by_ignore].cpu().numpy().astype(np.uint8),
            #     }
            # )
        points = torch.stack(points)
        uncertainty = torch.stack(uncertainty)
        if self.minimal_uncert_value_instead_zero:
            uncertainty[uncertainty == 0] = self.minimal_uncert_value_instead_zero
        nls_uncert = torch.stack(ood_output["uncert"])
        # print(torch.min(uncertainty))
        # print(torch.min(nls_uncert))
        uncertainty =  uncertainty+nls_uncert#+


        def map_instance_id2color(arr):
            # Get the unique values in the array
            unique_values = np.unique(arr)
            # Create a dictionary mapping the unique values to indices
            value_to_index = {value: index for index, value in enumerate(unique_values)}
            # Use np.vectorize to map the values to indices
            mapped_arr = np.vectorize(lambda x: value_to_index[x])(arr)
            return mapped_arr.astype(np.uint8)

        # if not self.visualize:
        #     return
        return uncertainty, points, sam_mask, ood_output["train_masks"]



    def object_inference(self, obj_cls, obj_pred):
        obj_cls = F.softmax(obj_cls, dim=-1)[..., :-1]
        index = torch.where(obj_cls>0.7)
        obj_cls = obj_cls[index[0]]
        obj_pred = obj_pred.sigmoid()
        obj_pred = obj_pred[index[0]]
        # K x H x W
        return torch.einsum("qc,qhw->chw", obj_cls, obj_pred)

    def nls_ood_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        uncertainty = -torch.sum(semseg, dim=0)
        #uncertainty =  -torch.max(semseg, dim=0, keepdim=True)[0]
        uncertainty = uncertainty.unsqueeze(0)
        return uncertainty

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        # K x H x W
        return semseg

    def batched_semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        # K x H x W
        return semseg

    def batched_nls_ood_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        #uncertainty = -torch.max(semseg, dim=1, keepdim=True)[0]
        uncertainty = -torch.sum(semseg, dim=1, keepdim=True)
        return uncertainty

    def semantic_inference_with_void(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, open_panoptic=False, ood_threshold=-0.1, pixel_min=300, return_ood_pred=False):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (
            scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros(
            (h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(
                                pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(
                                pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
            
            if open_panoptic:
                
                mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
                semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
                ood_mask = -(semseg.tanh()).sum(0)

                ood_mask_binary = ood_mask > ood_threshold

                # compute connected components of ood_mask
                ood_mask_binary = ood_mask_binary.cpu().numpy().astype(np.uint8)
                ood_mask_binary = cv2.morphologyEx(ood_mask_binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                ood_mask_binary = cv2.morphologyEx(ood_mask_binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                num_labels, labels_im = cv2.connectedComponents(ood_mask_binary, connectivity=4)
                labels_im = torch.from_numpy(labels_im).to(self.device)

                for i in range(1, num_labels):
                    mask = (labels_im == i) & (panoptic_seg == 0)
                    if mask.sum() < pixel_min:
                        continue
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": True,
                            "category_id": 255,
                        }
                    )

                if return_ood_pred:
                    return panoptic_seg, segments_info, ood_mask

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(
            0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(
            0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(
            1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
