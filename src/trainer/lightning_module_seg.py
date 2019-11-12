import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict, defaultdict

from factory import get_model, get_dataloader, get_loss, get_optimizer
from utils import dice_nomask, dice_pos_neg, dice_pytorch


class LightningModuleSeg(pl.LightningModule):
    def __init__(self, cfg):
        super(LightningModuleSeg, self).__init__()
        assumed_models = {'unet', 'clsunet', 'msunet', 'msclsunet'}
        assert cfg.Model.model_arch in assumed_models
        self.cfg = cfg

        self.model_arch = cfg.Model.model_arch
        self.softmax = (cfg.Model.output == 'softmax')
        self.num_class = len(cfg.General.labels)
        self.num_class_seg = cfg.Model.out_channel
        self.skip_first_class = self.num_class < cfg.Model.out_channel

        self.net = get_model(cfg)
        self.loss = get_loss(cfg)
        self.metrics_keys = ['dice0', 'dice', 'dice_p', 'dice_n']
        self.tta = set(cfg.Augmentation.tta)  # ex. {'hflip', 'vflip'}
        self.pred_cls_thres = 0.6

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, label = batch
        y_hat = self.forward(img)
        loss = self.loss(y_hat, label)
        return {'loss': loss}

    def final_layer(self, logits):
        if self.softmax:
            # softmax channel-wise
            preds = torch.softmax(logits, dim=1)

            # Convert max class value to one, other 0
            preds = F.one_hot(preds.argmax(dim=1),
                              self.num_class_seg).transpose(2, 3).transpose(1, 2).float()
        else:
            preds = torch.sigmoid(logits)
        return preds

    def drop_by_cls_prob(self, preds, preds_cls):
        indxs_no_mask = preds_cls < self.pred_cls_thres
        if self.num_class_seg != self.num_class:
            preds_tmp = preds[:, 1:]  # without no mask class
            preds_tmp[indxs_no_mask] = 0.
            preds[:, 1:] = preds_tmp
        else:
            preds[indxs_no_mask] = 0.
        return preds

    def separate_logits(self, logits):
        logits_cls = None

        if self.model_arch == 'clsunet':
            logits, logits_cls = logits
        elif self.model_arch == 'msunet':
            logits, _ = logits
        elif self.model_arch == 'msclsunet':
            logits, logits_cls, _ = logits

        return logits, logits_cls

    def pred_imgs(self, x):
        logits_origin = self.forward(x)
        preds_cls = None

        logits, logits_cls = self.separate_logits(logits_origin)

        preds = self.final_layer(logits)
        if logits_cls is not None:
            preds_cls = torch.sigmoid(logits_cls)
        return logits_origin, preds, preds_cls

    def apply_tta(self, x, preds, preds_cls=None):
        if ('hflip' not in self.tta) and ('vflip' not in self.tta):
            return preds, preds_cls

        tta_count = 1.

        if 'hflip' in self.tta:
            tta_count += 1
            _, preds_hflip, preds_cls_hflip = self.pred_imgs(x.flip(3))
            preds += preds_hflip.flip(3)
            if preds_cls is not None:
                preds_cls += preds_cls_hflip

        if 'vflip' in self.tta:
            tta_count += 1
            _, preds_vflip, preds_cls_vflip = self.pred_imgs(x.flip(2))
            preds += preds_vflip.flip(2)
            if preds_cls is not None:
                preds_cls += preds_cls_vflip

        preds /= tta_count
        if preds_cls is not None:
            preds_cls /= tta_count
        return preds, preds_cls

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits_all, preds, preds_cls = self.pred_imgs(x)
        loss_val = self.loss(logits_all, y)

        # Test Time Augmentation
        preds, preds_cls = self.apply_tta(x, preds, preds_cls)

        dice_simple = dice_pytorch((preds > 0.5).float(), y)

        if self.model_arch in {'clsunet', 'msclsunet'}:
            preds = self.drop_by_cls_prob(preds, preds_cls)

        dice, dice_pos, dice_neg = dice_pos_neg(
            preds, y, num_class=self.num_class,
            top_score_thresholds=[0.7],
            min_contour_areas=[10000],
            bottom_score_thresholds=[0.3],
            post_process=False,
            skip_first_channel=self.skip_first_class
        )

        output = OrderedDict({
            'val_loss': loss_val,
            'dice0': dice_simple,
            'dice': dice,
            'dice_p': dice_pos,
            'dice_n': dice_neg
        })
        return output

    def validation_end(self, outputs):
        total_loss = 0.
        metrics_list = defaultdict(list)

        for output in outputs:
            val_loss = output['val_loss']
            total_loss += val_loss
            for metrics_key in self.metrics_keys:
                metrics_list[metrics_key].append(output[metrics_key])

        # Calc mean
        if len(outputs) > 0:
            total_loss /= len(outputs)
        tqdm_dict = {
            'v_' + k: torch.cat(metrics_list[k], dim=0).mean()
            for k in self.metrics_keys
        }
        tqdm_dict['val_loss'] = total_loss
        ret_dict = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': total_loss}
        return ret_dict

    def configure_optimizers(self):
        conf_optim = self.cfg.Optimizer

        optimizer_cls, scheduler_cls = get_optimizer(self.cfg)

        optimizer = optimizer_cls(self.parameters(), lr=conf_optim.init_lr, **conf_optim.params)
        scheduler = scheduler_cls(optimizer, **conf_optim.lr_scheduler.params)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return get_dataloader(self.cfg, 'train')

    @pl.data_loader
    def val_dataloader(self):
        return get_dataloader(self.cfg, 'valid')

    @pl.data_loader
    def test_dataloader(self):
        return get_dataloader(self.cfg, 'test')
