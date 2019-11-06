import math
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class DistributedChangeRateSampler(DistributedSampler):
    """
    DistributedSampler changing negative example rate for each epoch
    PyTorch-Lightning call set_epoch() for each epoch

    References:
        SIIM segmentation 4th solution:
            https://github.com/amirassov/kaggle-pneumothorax
    """

    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, max_pos_rate=1.0, epochs=50):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.max_pos_rate = max_pos_rate
        self.epochs = epochs

        self.current_epoch = 0
        self.current_pos_rate = max_pos_rate

        # 1 if image has label else 0, This is empty list when test
        self.labels = torch.Tensor(dataset.labels)
        self.pos_indices = torch.where(self.labels == 1)[0]
        self.neg_indices = torch.where(self.labels == 0)[0]

        self.pos_num = self.pos_indices.size(0)  # No change for each epoch
        self.neg_num = 0
        self.update_neg_num()

        self.min_pos_rate = self.labels.float().mean().item()  # Original pos rate

    def update_pos_rate(self):
        if self.current_epoch >= self.epochs:
            self.current_pos_rate = self.min_pos_rate
        else:
            self.current_pos_rate = self.max_pos_rate
            self.current_pos_rate -= self.current_epoch * (self.max_pos_rate - self.min_pos_rate) / self.epochs

    def update_neg_num(self):
        if self.current_epoch >= self.epochs:
            self.neg_num = self.neg_indices.size(0)
        else:
            self.neg_num = int(self.pos_num * (1. - self.current_pos_rate) / self.current_pos_rate)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()

        # Sample negative indices
        g.manual_seed(self.current_epoch)
        neg_indices_tmp = torch.randperm(self.neg_indices.size(0), generator=g).tolist()
        neg_indices_sampled = self.neg_indices[neg_indices_tmp[:self.neg_num]]

        # Concat
        indices = torch.cat([self.pos_indices, neg_indices_sampled])

        # Shuffle
        if self.shuffle:
            g.manual_seed(self.current_epoch)
            indices_tmp = torch.randperm(indices.size(0), generator=g).tolist()
            indices = indices[indices_tmp]

        indices = indices.tolist()

        # Divide for each distributed process
        num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
        total_size = num_samples * self.num_replicas

        # Add extra samples to make it evenly divisible
        indices += indices[:(total_size - len(indices))]
        assert len(indices) == total_size

        # Subsample
        indices = indices[self.rank:total_size:self.num_replicas]
        assert len(indices) == num_samples

        return iter(indices)

    def __len__(self):
        return int(math.ceil((self.pos_num + self.neg_num) * 1.0 / self.num_replicas))

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.update_pos_rate()
        self.update_neg_num()
