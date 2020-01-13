import yaml
import torch
import albumentations as alb
import albumentations.pytorch as albp
from addict import Dict

from loss_funcs import *
from datasets import CloudDataset, DistributedChangeRateSampler, RandomInputBlack, RandomInputBlackEllipse
from models import select_seg_model

"""
Functions for converting config to each object
"""


def get_transform(conf_augmentation, img_height, img_width):
    def get_object(trans):
        if trans.name in {'Compose', 'OneOf'}:
            augs_tmp = [get_object(aug) for aug in trans.member]
            return getattr(alb, trans.name)(augs_tmp, **trans.params)

        if hasattr(alb, trans.name):
            return getattr(alb, trans.name)(**trans.params)
        else:
            return eval(trans.name)(**trans.params)

    if conf_augmentation is None:
        augs = list()
    else:
        augs = [get_object(aug) for aug in conf_augmentation]
    augs.extend([
        alb.Resize(height=img_height, width=img_width),
        albp.ToTensorV2()
    ])
    return alb.Compose(augs)


def get_dataset(conf, transforms, phase='train'):
    assert phase in {'train', 'valid', 'test'}
    conf_ds = conf.Data.dataset
    conf_ds.debug = conf.General.debug

    # Make Dataset
    if phase in {'train', 'valid'}:
        df_path = conf_ds.train_df
        img_dir = conf_ds.train_img_dir
    else:
        df_path = conf_ds.test_df
        img_dir = conf_ds.test_img_dir
    ds = CloudDataset(
        alb_transforms=transforms,
        df_path=df_path,
        img_dir=img_dir,
        img_height=conf_ds.img_height,
        img_width=conf_ds.img_width,
        kfold=conf_ds.kfold,
        phase=phase,
        background_class=conf_ds.background_class,
        debug=conf_ds.debug
    )
    return ds


def get_dataloader(conf, phase='train'):
    assert phase in {'train', 'valid', 'test'}

    conf_aug = conf.Augmentation[phase]
    img_height = conf.Data.dataset.img_height
    img_width = conf.Data.dataset.img_width
    transforms = get_transform(conf_aug, img_height=img_height, img_width=img_width)

    ds = get_dataset(conf, transforms=transforms, phase=phase)

    # Make Loader
    conf_loader = conf.Data.dataloader
    kwargs_loader = {
        'num_workers': conf_loader.num_workers,
        'pin_memory': True,
        'drop_last': phase == 'train'
    }

    # Batch size
    bs = conf_loader.batch_size
    if conf.General.multi_gpu_mode == 'ddp':
        bs //= len(conf.General.gpus)
    kwargs_loader['batch_size'] = bs

    # Sampler(shuffle and sampler)
    conf_sampler = conf_loader[phase]
    sampler = None
    shuffle = False
    if phase in {'train': 'valid'}:
        if phase == 'train' and conf_sampler.change_rate_sampler:
            sampler = DistributedChangeRateSampler(
                ds, shuffle=True, max_pos_rate=conf_sampler.max_pos_rate,
                epochs=conf.General.epoch
            )
        elif conf_sampler.use_sampler:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=(phase == 'train'))
        else:
            shuffle = (phase == 'train')
    elif conf_sampler.use_sampler:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, shuffle=False)
    kwargs_loader['sampler'] = sampler
    kwargs_loader['shuffle'] = shuffle
    return torch.utils.data.DataLoader(ds, **kwargs_loader)


def get_model(conf):
    num_class = len(conf.General.labels)
    model = select_seg_model(
        model_arch=conf.Model.model_arch,
        encoder_type=conf.Model.encoder,
        num_class_seg=conf.Model.out_channel,
        num_class_cls=num_class,
        pretrained=conf.Model.pretrained,
        base_ch=conf.Model.base_ch,
        use_aspp=conf.Model.use_aspp
    )
    return model


def get_loss(conf):
    conf_base = conf.Loss.base_loss
    ret_loss = eval(conf_base.name)(**conf_base.params)
    if len(conf.Loss.wrapper_loss) > 0:
        conf_wrappers = conf.Loss.wrapper_loss
        # conf_wrapper = conf.Loss.wrapper_loss
        for conf_wrapper in conf_wrappers:
            ret_loss = eval(conf_wrapper.name)(ret_loss, **conf_wrapper.params)
    print(ret_loss)
    return ret_loss


def get_optimizer(conf):
    conf_optim = conf.Optimizer
    optimizer_cls = getattr(
        torch.optim, conf_optim.optimizer
    )

    if hasattr(conf_optim, 'lr_scheduler'):
        scheduler_cls = getattr(
            torch.optim.lr_scheduler,
            conf_optim.lr_scheduler.name
        )
    else:
        scheduler_cls = None
    return optimizer_cls, scheduler_cls


def read_yaml(fpath='./configs/sample.yaml'):
    with open(fpath, mode='r') as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)


if __name__ == '__main__':
    d = read_yaml()
    print(d.Augmentation)

