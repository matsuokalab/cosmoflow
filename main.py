import numpy as np
import pytorch_lightning as pl
import tfrecord
import torch

from model import build_model

path = "/groups1/gac50489/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_small/train/univ_ics_2019-03_a10000668_000.tfrecord"

reader = tfrecord.reader.tfrecord_loader(data_path=path, index_path=None)

for data in reader:
    print(data)
    x = data["x"].reshape(8, 128, 128, 128) / 255 - 0.5
    y = data["y"]
    print("in shape:", x.shape)
    print("in max:", x.max())

x = x[np.newaxis, :].astype(np.float32)
x = torch.from_numpy(x)

# write data iterator or reuse off-the shelf something
# either pytorch dataloader or I want to try lightning actually


class LitClassifier(pl.LightningModule):
    def __init__(self):
        self.net = build_model((128, 128, 128, 8), 4, 0)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, on_epoch=True)
        return result


net = build_model((128, 128, 128, 8), 4, 0)
net.training_step = training_step


# extend to work with lightning

result = net(x)
print(result)


# add training loop -
# can also try lightning here
# I actually really loved Chainer's trainer for its extensions architecture
# and it is ported to torch, but not sure that will be good long-term investment :-\

# train ^_^
# add wandb logging
