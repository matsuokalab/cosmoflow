import os

import numpy as np
import pytorch_lightning as pl
import tfrecord
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import build_model

path_data = (
    "/groups1/gac50489/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_small/train"
)
# "univ_ics_2019-03_a10000668_000.tfrecord"
# TODO: this is a bit ugly, but I expect to find some torch-y out of the box method later
tensor_x = []
tensor_y = []

for name_file in os.listdir(path_data)[:12]:
    path_file = os.path.join(path_data, name_file)
    reader = tfrecord.reader.tfrecord_loader(data_path=path_file, index_path=None)
    data = next(reader)  # we expect only one record in a file
    x = data["x"].astype(np.float32).reshape(8, 128, 128, 128) / 255 - 0.5
    y = data["y"].astype(np.float32)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    tensor_x.append(x)
    tensor_y.append(y)
    # print("in shape:", x.shape)
    # print("in max:", x.max())
    # print("y", y)

tensor_x = torch.stack(tensor_x)
tensor_y = torch.stack(tensor_y)
dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
dataloader = DataLoader(dataset)  # create your dataloader

# write data iterator or reuse off-the shelf something
# either pytorch dataloader or I want to try lightning actually


class Cosmoflow(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = build_model((128, 128, 128, 8), 4, 0)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)


train, val = random_split(dataset, [8, 4])

model = Cosmoflow()
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=10,
    verbose=False,
    mode="min",
)
trainer = pl.Trainer(max_epochs=50, early_stop_callback=early_stop_callback)
trainer.fit(model, DataLoader(train), DataLoader(val))

# TODO: load more data
# TODO: proper test/val split
# TODO: stop trigger on val loss stop decrease
# TODO: explicitly set batch size
# TODO: add wandb logging
