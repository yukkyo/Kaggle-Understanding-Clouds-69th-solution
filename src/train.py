import shutil
import argparse
import torch
import pandas as pd
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only

from factory import read_yaml
from utils import seed_everything, src_backup, setup_logger
from trainer import LightningModuleSeg


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--debug', action='store_true', help='debug')
    arg('--config', default=None, type=str, help='config path', required=True)
    arg('--kfold', type=int, default=None, required=True)
    return parser


class MyTrainer(Trainer):
    def __init__(self, **args):
        super(MyTrainer, self).__init__(**args)

    def run_tng_epoch(self):
        """
        Override original method to change batch_sampler for each epoch
        `self.tng_dataloader.sampler.set_epoch(epoch_nb)` is called before this func
        """
        if self.use_ddp:
            self.train_dataloader.batch_sampler = torch.utils.data.BatchSampler(
                self.train_dataloader.sampler,
                self.train_dataloader.batch_size,
                self.train_dataloader.drop_last
            )
            self.nb_training_batches = len(self.train_dataloader)
            self.total_batches = self.nb_training_batches + self.nb_val_batches

            # Reset progress_bar when requested
            if self.show_progress_bar:
                self.progress_bar.reset(self.total_batches)

        # Call Original Code
        super(MyTrainer, self).run_training_epoch()


class MyLogger(LightningLoggerBase):
    def __init__(self, logger_df_path: Path):
        super(MyLogger, self).__init__()
        self.all_metrics = defaultdict(list)
        self.df_path = logger_df_path

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step_num):
        if self.rank > 0 or len(metrics) < 2:
            return

        if step_num is None:
            step_num = len(self.all_metrics['step_num'])
        self.all_metrics['step_num'].append(step_num)

        if 'created_at' not in metrics:
            metrics['created_at'] = str(datetime.utcnow())

        for k, v in metrics.items():
            self.all_metrics[k].append(v)

        metrics_df = pd.DataFrame(self.all_metrics)
        metrics_df = metrics_df[sorted(metrics_df.columns)]
        metrics_df.to_csv(self.df_path, index=False)


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, kfold, **args):
        super(MyModelCheckpoint, self).__init__(**args)
        self.monitor = 'val_loss'
        self.kfold = kfold
        self.latest_path = f'{self.filepath}/kfold_{kfold}_latest.ckpt'

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose > 0:
            print(f'\nSaving latest model to {self.latest_path}')
        self.save_model(self.latest_path, overwrite=False)


def train_a_kfold(cfg, output_path):
    model = LightningModuleSeg(cfg)
    debug = cfg.General.debug

    kfold = cfg.Data.dataset.kfold
    logger_name = f'kfold_{str(kfold).zfill(2)}.csv'
    mylogger = MyLogger(logger_df_path=output_path / logger_name)

    checkpoint_callback = MyModelCheckpoint(
        kfold=kfold,
        filepath=str(output_path),
        verbose=True,  # print save result, not must
    )

    trainer = MyTrainer(
        logger=mylogger,
        max_nb_epochs=3 if debug else cfg.General.epoch,
        checkpoint_callback=checkpoint_callback,
        train_percent_check=0.01 if debug else 1.0,
        val_percent_check=0.01 if debug else 1.0,
        gpus=cfg.General.gpus,
        use_amp=cfg.General.fp16,
        amp_level=cfg.General.amp_level,
        distributed_backend=cfg.General.multi_gpu_mode,
        log_save_interval=5 if debug else 200,
        row_log_interval=1e8,  # no log
        accumulate_grad_batches=cfg.General.grad_acc
    )
    trainer.fit(model)


def main():
    args = make_parse().parse_args()

    cfg = read_yaml(fpath=args.config)
    cfg.Data.dataset.kfold = args.kfold
    cfg.General.debug = args.debug
    for key, value in cfg.items():
        print(f"    {key.ljust(30)}: {value}")

    output_path = Path(cfg.General.workdir)
    if args.debug:
        name_tmp = output_path.name
        output_path = Path('../output/tmp') / name_tmp
        cnt = 0
        while output_path.exists():
            output_path = output_path.parent / (name_tmp + f'_{cnt}')
            cnt += 1

    output_path.mkdir(parents=True, exist_ok=True)
    output_path.chmod(0o777)

    # Backup
    shutil.copy2(args.config, str(output_path / Path(args.config).name))
    src_backup_path = output_path / "src_backup"
    src_backup_path.mkdir(exist_ok=True)
    src_backup(input_dir=Path("./"), output_dir=src_backup_path)

    seed_everything(seed=cfg.General.seed)
    train_a_kfold(cfg, output_path)


if __name__ == '__main__':
    main()
