import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from test_tube import Experiment


class CoolSystem(pl.LightningModule):

    def __init__(self):
        super(CoolSystem, self).__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()))


def main():
    model = CoolSystem()

    # PyTorch summarywriter with a few bells and whistles
    exp = Experiment(save_dir='../output/tmp')
    print(f"exp.save_dir: {exp.save_dir}")
    exp.save()
    print(f"saved !!!")

    # train on cpu using only 10% of the data (for demo purposes)
    # pass in experiment for automatic tensorboard logging.
    trainer = Trainer(
        experiment=exp, max_nb_epochs=1, train_percent_check=0.1
    )

    # train on 4 gpus (lightning chooses GPUs for you)
    # trainer = Trainer(experiment=exp, max_nb_epochs=1, gpus=4)

    # train on 4 gpus (you choose GPUs)
    # trainer = Trainer(experiment=exp, max_nb_epochs=1, gpus=[0, 1, 3, 7])

    # train on 32 gpus across 4 nodes (make sure to submit appropriate SLURM job)
    # trainer = Trainer(experiment=exp, max_nb_epochs=1, gpus=8, nb_gpu_nodes=4)

    # train (1 epoch only here for demo)
    trainer.fit(model)


if __name__ == '__main__':
    main()
