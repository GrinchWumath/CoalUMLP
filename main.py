import argparse
import os
from functools import partial
from kornia.losses import HausdorffERLoss3D
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from networks.CoalUMLP import CoalUMLP
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader
from networks.CoalUMLP import MyNorm
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric,HausdorffDistanceMetric,compute_hausdorff_distance
from monai.transforms import (AsDiscrete,AsDiscreted,Activationsd,Activations, EnsureChannelFirstd,
                              SqueezeDimd,EnsureTyped,AsDiscrete, Compose,LoadImaged,ConvertToMultiChannelBasedOnBratsClassesd,
                              Orientationd,RandCropByPosNegLabeld,SaveImaged,ScaleIntensityRanged,Spacingd,
                              Invertd,MapTransform,NormalizeIntensityd, RandScaleIntensityd,RandShiftIntensityd,RandSpatialCropd,)

from monai.utils.enums import MetricReduction
from monai.utils import set_determinism
from monai.handlers.utils import from_engine




parser = argparse.ArgumentParser(description="CoalUMLP segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="/root/tf-logs", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="CoalUMLP_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=500, type=int, help="max number of training epochs")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--model_name", default="CoalUMLP", type=str, help="model name")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--img_size",default=128,type=int,help="image size")
parser.add_argument("--embed_dim",default=64,type=int,help="embed_dim")
parser.add_argument("--as_bias",default=True,type=bool,help="if add bias")
parser.add_argument("--mlp_ratio",default=4.,type=float,help="mlp_ratio")
parser.add_argument("--in_chan", default=4, type=int, help="number of input channels")
parser.add_argument("--num_classes", default=3, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-57.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=164.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=224, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=224, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
#parser.add_argument("use_checkpoint",default=True,type=bool,help="if use checkpoint")
parser.add_argument('--depth', type=int, nargs='+', default=[3, 3, 3])
parser.add_argument('--shift_sizes', type=int, nargs='+', default=[3, 5, 7, 9])
parser.add_argument("--drop",default=0.,type=float,help="drop")
parser.add_argument('--drop_path', default=0., type=float, help='Drop path rate')
parser.add_argument('--mask_prob', type=float, default=0.15)
parser.add_argument('--loss_func', type=str, default='DiceCELoss')
parser.add_argument('--lambda1',type=int,default=1)
parser.add_argument('--lambda2',type=int,default=2)


def parse_tuple_string(s):
    return tuple(s.split(','))

parser.add_argument("--dims", default=('D', 'H', 'W'), type=parse_tuple_string, help="dims")

parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


class ModifiedHausdorffERLoss3D(HausdorffERLoss3D):
    def __init__(self, alpha=2.0, k=5, reduction='mean'):
        super().__init__(alpha=alpha, k=k, reduction=reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, target):
        input = self.sigmoid(input)
        return super().forward(input, target)


class CombinedLoss(nn.Module):
    def __init__(self,lambda1=1, lambda2=1):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(include_background=False, to_onehot_y=False, softmax=True)
        self.hd_loss = ModifiedHausdorffERLoss3D()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, input, target):
        loss1 = self.dice_loss(input, target)
        losses = []
        num_channels = input.shape[1]
        for i in range(num_channels):
            input_channel = input[:, i:i + 1, ...]
            target_channel = target[:, i:i + 1, ...]
            loss_channel = self.hd_loss(input_channel, target_channel)
            losses.append(loss_channel)
        loss2 = torch.mean(torch.stack(losses))

        # Combine losses
        loss = self.lambda1*loss1 + self.lambda2*loss2
        return loss

'''class CombinedLoss(nn.Module):
    def __init__(self,lambda1=1, lambda2=1):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(include_background=False, to_onehot_y=False, softmax=True)
        self.hd_loss = ModifiedHausdorffERLoss3D()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, input, target):
        # Convert target to binary for Dice loss
        loss1 = self.dice_loss(input, target)
        target_binary = torch.argmax(target, dim=1, keepdim=True)

        # Calculate Hausdorff loss for each channel
        num_channels = input.shape[1]
        hd_losses = []
        for i in range(num_channels):
            input_channel = input[:, i, ...].unsqueeze(1)
            target_channel = (target_binary == i).float()
            hd_loss_channel = self.hd_loss(input_channel, target_channel)
            hd_losses.append(hd_loss_channel)

        # Average Hausdorff losses
        loss2 = torch.mean(torch.stack(hd_losses))

        # Combine losses
        loss = self.lambda1*loss1 + self.lambda2*loss2
        return loss
        '''

def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir
    if (args.model_name is None) or args.model_name == "CoalUMLP":
        model = CoalUMLP(
            in_chans=4,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            num_classes=args.num_classes,
            depth=args.depth,
            mlp_ratio=args.mlp_ratio,
            use_checkpoint=args.checkpoint,
            shift_sizes=args.shift_sizes,
            mask_prob=args.mask_prob,
            embed_dim=args.embed_dim,
        )

        if args.resume_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            model.load_state_dict(model_dict)
            print("Use pretrained weights")

        if args.resume_jit:
            if not args.noamp:
                print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    else:
        raise ValueError("Unsupported model " + str(args.model_name))
    if args.loss_func is not None:
        if args.loss_func == "DiceCELoss":
            dice_loss = DiceCELoss(
                to_onehot_y=False, sigmoid=True, squared_pred=args.squared_dice, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
            )
        elif args.loss_func == "DiceLoss":
            dice_loss = DiceLoss(
                to_onehot_y=False, sigmoid=True, squared_pred=args.squared_dice, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
            )
        elif args.loss_func == "CrossEntropyLoss":
            dice_loss = CrossEntropyLoss(to_onehot_y=False, sigmoid=True)
        elif args.loss_func == "ModifiedLoss":
            dice_loss = CombinedLoss(lambda1=args.lambda1, lambda2=args.lambda2)
        else:
            raise ValueError("Unsupported loss function " + str(args.loss_func))

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, logit_thresh=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    hd=HausdorffDistanceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH,percentile=95,get_not_nans=True)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    best_hd=10000
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        if "best_hd" in checkpoint:
            best_hd = checkpoint["best_hd"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})(besthd {})".format(args.checkpoint, start_epoch, best_acc, best_hd))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    semantic_classes = ["Dice_Val_TC", "Dice_Val_WT", "Dice_Val_ET"]
    accuracy,hd = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        hd_func=hd,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
        semantic_classes=semantic_classes,
    )
    return accuracy,hd


if __name__ == "__main__":
    main()