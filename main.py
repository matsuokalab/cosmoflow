import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from model import build_model
from data import CFDataModule
import horovod.torch as hvd


class Cosmoflow(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = build_model((128, 128, 128, 8), 4, 0)
        self.example_input_array = torch.zeros((1, 8, 128, 128, 128))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        print(f"\n#####training step seees {x.shape} on worker {hvd.rank()} of {hvd.size()}\n")
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.TrainResult(loss)
        result.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        result = pl.EvalResult(early_stop_on=loss, checkpoint_on=loss)
        result.log("val_loss", loss, sync_dist=True)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)


def main():
    wandb_logger = WandbLogger(project="cosmoflow")
    early_stop_callback = EarlyStopping(
        min_delta=0.0001,
        patience=5,
        verbose=True,
        mode="min",
    )
    print("create tainer")
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1,
        distributed_backend="horovod",
        replace_sampler_ddp=False,
        early_stop_callback=early_stop_callback,
        logger=wandb_logger,
    )
    print("tainer created")

    path_data = "/groups1/gac50489/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_small"
    batch_size = 2
    data_module = CFDataModule(path_data, batch_size)
    model = Cosmoflow()
    print("fit")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()

# TODO: load more data
