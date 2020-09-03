import numpy as np
import pytorch_lightning as pl
import tfrecord
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import build_model

path = "/groups1/gac50489/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_small/train/univ_ics_2019-03_a10000668_000.tfrecord"

reader = tfrecord.reader.tfrecord_loader(data_path=path, index_path=None)

for data in reader:
    print(data)
    x = data["x"].astype(np.float32).reshape(8, 128, 128, 128) / 255 - 0.5
    x = np.array([x] * 4)
    y = data["y"].astype(np.float32)
    y = np.array([y] * 4)
    print("in shape:", x.shape)
    print("in max:", x.max())
    print("y", y)
    ####

tensor_x = torch.from_numpy(x)
tensor_y = torch.from_numpy(y)
my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
my_dataloader = DataLoader(my_dataset)  # create your dataloader

# write data iterator or reuse off-the shelf something
# either pytorch dataloader or I want to try lightning actually

# @todo(vatai): PROPER NAME
class LitClassifier(pl.LightningModule):
    def __init__(self):
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
        return torch.optim.Adam(self.net.parameters(), lr=0.02)


train, val = random_split(dataset, [2, 2])

model = LitClassifier()
trainer = pl.Trainer()
trainer.fit(model, DataLoader(train), DataLoader(val))

# extend to work with lightning


# add training loop -
# can also try lightning here
# I actually really loved Chainer's trainer for its extensions architecture
# and it is ported to torch, but not sure that will be good long-term investment :-\

# train ^_^
# add wandb logging
