import torch

import segmentation_models_pytorch as smp
from vision.models.base import BaseModel
from torch import Tensor


class SegModelPytorchClf(BaseModel):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, n_features_out,
                 out_classes, verbose=False):
        super(SegModelPytorchClf, self).__init__()
        self.verbose = verbose

        if arch == "FPN":
            self.model = smp.create_model(
                arch,
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_classes
            )

            # for param in self.model.parameters():
            #     param.requires_grad = False

            self.model.adapt = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
            )

            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout1d(0.10),

                torch.nn.Linear(n_features_out, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.Sigmoid(),

                torch.nn.Linear(512, out_classes),
            )

            del self.model.decoder
            del self.model.segmentation_head
            del self.model.classification_head

            params = smp.encoders.get_preprocessing_params(encoder_name)
            self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
            self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        else:
            raise ValueError(f'Not implemented arch = {arch}')

    def forward(self, x: Tensor):
        x = (x - self.mean) / self.std
        features = self.model.encoder(x)
        features = self.model.adapt(features[-1]).squeeze(-1).squeeze(-1)
        logits = self.model.fc(features)
        return logits


class SegModelPytorch(BaseModel):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, verbose=False):
        super(SegModelPytorch, self).__init__()
        self.verbose = verbose

        if arch == "FPN":
            self.model = smp.create_model(
                arch,
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_classes
            )

            params = smp.encoders.get_preprocessing_params(encoder_name)
            self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
            self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        else:
            raise ValueError(f'Not implemented arch = {arch}')

    def forward(self, x: Tensor):
        x = (x - self.mean) / self.std
        x = self.model(x)
        return x
