import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict

from factory import get_model, get_dataloader, get_loss, get_optimizer
from utils import dice_nomask


class LightningModuleSeg(pl.LightningModule):
    def __init__(self, cfg):
        super(LightningModuleSeg, self).__init__()
        self.cfg = cfg

        self.model_arch = cfg.Model.model_arch
        self.return_multi = (self.model_arch in {'clsunet', 'msunet'})
        self.softmax = (cfg.Model.output == 'softmax')
        self.num_class = len(cfg.General.labels)
        self.num_class_seg = cfg.Model.out_channel
        self.skip_first_class = self.num_class < cfg.Model.out_channel

        self.net = get_model(cfg)
        self.loss = get_loss(cfg)

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

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self.forward(x)
        loss_val = self.loss(logits, y)

        logits_sub = None
        if self.return_multi:
            logits, logits_sub = logits

        preds = self.final_layer(logits)

        # TODO add tta

        # Use classification results
        # TODO in to metrics func
        if self.model_arch == 'clsunet':
            preds_cls = torch.sigmoid(logits_sub)  # BATCH x num_class_cls
            indxs_no_mask = preds_cls < 0.4
            if self.num_class_seg != self.num_class:
                preds_tmp = preds[:, 1:]  # without no mask class
                preds_tmp[indxs_no_mask] = 0.
                preds[:, 1:] = preds_tmp
            else:
                preds[indxs_no_mask] = 0.

        # calc dice separate pos and neg
        # Use same thresholds for each class
        # TODO calc dice using logits_cls
        # TODO get by cfg
        dice, dice_pos, dice_neg = dice_nomask(
            preds, y, num_class=self.num_class, threshold=0.5,
            # 4 is default class
            skip_first_class=self.skip_first_class,
            min_contour_area=1024*16
        )

        output = OrderedDict({
            'val_loss': loss_val,
            'dice': dice,
            'dice_pos': dice_pos,
            'dice_neg': dice_neg
        })
        return output

    def validation_end(self, outputs):
        total_loss = 0.
        dices = list()
        dices_pos = list()
        dices_neg = list()

        for output in outputs:
            val_loss = output['val_loss']

            total_loss += val_loss
            dices.append(output['dice'])
            dices_pos.append(output['dice_pos'])
            dices_neg.append(output['dice_neg'])

        if len(outputs) > 0:
            total_loss /= len(outputs)
        dices = torch.cat(dices, dim=0).mean()
        dices_pos = torch.cat(dices_pos, dim=0).mean()
        dices_neg = torch.cat(dices_neg, dim=0).mean()

        tqdm_dict = {
            'val_loss': total_loss,
            'val_dice': dices,
            'val_dice_pos': dices_pos,
            'val_dice_neg': dices_neg
        }
        ret_dict = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': total_loss}
        return ret_dict

    def configure_optimizers(self):
        conf_optim = self.cfg.Optimizer

        optimizer_cls, scheduler_cls = get_optimizer(self.cfg)

        optimizer = optimizer_cls(self.parameters(), lr=conf_optim.init_lr)
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
