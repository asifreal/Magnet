from argparse import ArgumentParser

import numpy as np
import os
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics

from utils import Conv1dSame, ActivationLSTMCell, CustomLSTM
import seisbench.data as sbd
import seisbench.generate as sbg
from seisbench.util import worker_seeding

class MagLabeller(sbg.PickLabeller):
    def __init__(self, **kwargs):
        self.label_method = "Mag"
        kwargs["dim"] = kwargs.get("dim", -2)
        kwargs["label_columns"] = ["source_magnitude"]
        super().__init__(label_type="multi_label", noise_column=False, **kwargs)

    def label(self, X, metadata):
        y = metadata['source_magnitude']
        return np.float32(y)

    def __str__(self):
        return f"MagLabeller (label_type={self.label_type}, dim={self.dim})"


def customLoss(yPred, yTrue):
    y_hat = yPred[:, 0].view(-1, 1)
    s = yPred[:, 1].view(-1, 1)
    return torch.mean(0.5 * torch.exp(-1 * s) * torch.square(yTrue - y_hat) + 0.5 * s)


class LitMagNet(pl.LightningModule):
    def __init__(self, input_dim=(3,6000), hidden_dim=128, drop_rate=0.2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.dropout = torch.nn.Dropout(drop_rate)
        self.conv1 = torch.nn.Conv1d(input_dim[0], 64, 3, padding=1)
        self.maxpool1 = torch.nn.MaxPool1d(5)
        self.conv2 = torch.nn.Conv1d(64, 32, 3, padding=1)
        self.maxpool2 = torch.nn.MaxPool1d(5)
        self.bilstm = torch.nn.LSTM(240, 32, batch_first=True, bidirectional=True)
        self.mlp = torch.nn.Linear(64, 1) # 64, 2
        self.loss =  F.mse_loss #customLoss


    def forward(self, x):
        x = self.dropout(self.conv1(x))
        x = self.maxpool1(x)
        x = self.dropout(self.conv2(x))
        x = self.maxpool2(x)
        x, _ = self.bilstm(x)
        x = self.mlp(x[:, -1, :])
        return x

    def shared_step(self, batch):
        x = batch["X"]
        y_true = batch["y"].view(-1, 1)
        y_pred = self.forward(x)
        return y_pred, y_true

    def training_step(self, batch, batch_idx):
        y_pred, y_true = self.shared_step(batch)
        loss = self.loss(y_pred, y_true)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y_true = self.shared_step(batch)
        loss = self.loss(y_pred, y_true)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        y_pred, y_true = self.shared_step(batch)
        loss = self.loss(y_pred, y_true)
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        y_pred, y_true = self.shared_step(batch)
        return y_pred, y_true
        #return y_pred[:,0].view(-1, 1), y_true

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return  {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, 'min', patience=4, min_lr=0.5e-6),
                "monitor": "val_loss",
                "frequency": 2
            },
        }

    def configure_callbacks(self):
            early_stop = EarlyStopping("val_loss", mode="min", patience=6)
            checkpoint = ModelCheckpoint(
                save_top_k=3,
                monitor="val_loss",
                mode="min",
                filename="magnet-{epoch:02d}-{val_loss:.2f}",
            )
            return [early_stop, checkpoint]
            
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def prepare_data(reshuffle, batch_size, mode="preload_all"):
    stead = sbd.STEAD(sampling_rate=100, component_order="ZNE", dimension_order="NCW", cache="trace")
    
    mask = (stead.metadata["trace_category"]  == 'earthquake_local') & \
       (stead.metadata["source_distance_km"] <= 110) & \
       (stead.metadata["source_magnitude_type"] == 'ml') & \
       (stead.metadata["trace_p_arrival_sample"] >= 200) & \
       (stead.metadata["trace_p_arrival_sample"] <= 1500) & \
       (stead.metadata["trace_s_arrival_sample"] >= 200) & \
       (stead.metadata["trace_s_arrival_sample"] <= 2500) & \
       (stead.metadata["path_p_travel_sec"].notnull()) & \
       (stead.metadata["path_p_travel_sec"] > 0) & \
       (stead.metadata["source_distance_km"].notnull()) & \
       (stead.metadata["source_distance_km"] > 0) & \
       (stead.metadata["source_depth_km"].notnull()) & \
       (stead.metadata["source_magnitude"].notnull()) & \
       (stead.metadata["path_back_azimuth_deg"].notnull()) & \
       (stead.metadata["path_back_azimuth_deg"] > 0) 
    stead.filter(mask)
    stead.metadata['coda_end_sample'] = stead.metadata.apply(lambda x: int(x['trace_coda_end_sample'][2:-3]), axis=1)
    stead.metadata['snr_db'] = stead.metadata.apply(lambda x: sum([float(a) for a in x['trace_snr_db'][1:-1].split(' ') if a != ''])/3, axis=1)
    mask = (stead.metadata["coda_end_sample"] <= 3000) & (stead.metadata["snr_db"] > 0)
    stead.filter(mask)
    print(f"available data size = len(stead)")

    if reshuffle:
        split = np.array(["train"] * len(stead))
        p1, p2 = int(0.6 * len(stead)), int( 0.7 * len(stead))
        split[p1:p2] = "dev"
        split[p2:] = "test"
        np.random.shuffle(split)
        stead._metadata["split"] = split

    train_data, dev_data, test_data = stead.train_dev_test()
    print(len(train_data), len(dev_data), len(test_data))

    if mode == "preload_all":
        train_data.preload_waveforms(pbar=True)
        dev_data.preload_waveforms(pbar=True)
        test_data.preload_waveforms(pbar=True)
    else:
        train_data.preload_waveforms(pbar=True)
        dev_data.preload_waveforms(pbar=True)

    train_generator = sbg.GenericGenerator(train_data)
    dev_generator = sbg.GenericGenerator(dev_data)
    test_generator = sbg.GenericGenerator(test_data)
    labeler = [MagLabeller()]
    train_generator.add_augmentations(labeler)
    dev_generator.add_augmentations(labeler)
    test_generator.add_augmentations(labeler)

    train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_seeding, drop_last=True)
    val_loader = DataLoader(dev_generator, batch_size=batch_size, num_workers=2, worker_init_fn=worker_seeding,)
    test_loader = DataLoader(test_generator, batch_size=batch_size, worker_init_fn=worker_seeding,)
    return train_loader, val_loader, test_loader


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--reshuffle', default=False, type=bool)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitMagNet.add_model_specific_args(parser)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # ------------
    # data
    # ------------
    train_loader, val_loader, test_loader = prepare_data(args.reshuffle, args.batch_size)

    # ------------
    # model
    # ------------
    model = LitMagNet(**config.get("model_args", {}))

    # ------------
    # training
    # ------------
    # CSV logger - also used for saving configuration as yaml
    experiment_name = os.path.basename(args.config)[:-5]
    csv_logger = CSVLogger("results", experiment_name)
    csv_logger.log_hyperparams(config)
    loggers = [csv_logger]

    default_root_dir = os.path.join(
        "results"
    )
    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        logger=loggers,
        **config.get("trainer_args", {}),
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(os.path.join(default_root_dir, "model.ckpt"))

    # ------------
    # testing
    # ------------
    result = trainer.predict(dataloaders=test_loader)
    y_pred, y_true = [], []
    for r in result:
        y_pred.append(r[0].detach())
        y_true.append(r[1].detach())
    np_result = torch.cat( (torch.cat(y_pred, 0), torch.cat(y_true, 0)), 1).cpu().numpy()
    with open(os.path.join(default_root_dir, "result.npy"), 'wb') as f:
        np.save(f, np_result)
    y_pred = np_result[:, 0]
    y_label = np_result[:, 1]
    print("MSE: ", np.average(np.abs(y_pred - y_label)))
    print("Sigma: ", np.sqrt(np.average(np.square(y_pred - y_label))))

if __name__ == '__main__':
    cli_main()
