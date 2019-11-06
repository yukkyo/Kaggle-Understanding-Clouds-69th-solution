import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetEncoder(nn.Module):
    def __init__(self, model_name):
        super(EfficientNetEncoder, self).__init__()
        assert model_name in {'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5'}

        self.encoder = EfficientNet.from_pretrained(model_name)

        model_name_to_extracts = {
            'efficientnet-b3': {4, 7, 14, 25},
            'efficientnet-b4': {5, 9, 21, 31},
            'efficientnet-b5': {7, 12, 26, 36}
        }
        self.extracts = model_name_to_extracts[model_name]
        self.len_encoder = max(self.extracts) + 1

        model_name_to_planes = {
            'efficientnet-b3': [32, 48, 136, 384],
            'efficientnet-b4': [32, 56, 160, 448],
            'efficientnet-b5': [40, 64, 176, 512]
        }
        self.planes = model_name_to_planes[model_name]

    def forward(self, x):
        x = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(x)))
        outputs = list()

        # Encoder blocks
        for idx in range(self.len_encoder):
            drop_connect_rate = self.encoder._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.encoder._blocks)
            x = self.encoder._blocks[idx](x, drop_connect_rate=drop_connect_rate)

            if idx in self.extracts:
                outputs.append(x)
        return outputs
