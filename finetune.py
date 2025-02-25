import os
import argparse
from pprint import pformat

import yaml
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

from finetune.WLASL.WLASLDataModule import WLASLDataModule
from finetune.SignBERTModel import SignBertModel

from IPython import embed;

class Config:
    """Stores configuration parameters"""
    def __init__(self, **config):
        self.__dict__.update(config)
    
    def __repr__(self):
        return pformat(vars(self), indent=2)


def main(args):
    with open(args.config, "r") as fid:
        cfg = yaml.load(fid, yaml.SafeLoader)
    # Update configuration with command line arguments (args overrides cfg)
    cfg.update(args.__dict__)
    # Create Config object
    config = Config(**cfg)
    print(config)
    # Instantiate data module with batch size and additional datamodule arguments
    datamodule = WLASLDataModule(
        batch_size=config.batch_size,
        **config.datamodule_args,
        meta_data_file_name = args.metadata        
    )
    
    # Instantiate the model with checkpoint, learning rate, and head arguments
    print ("Loading the model..")
    model_checkpoint = "checkpoints/Hands17/ckpts/epoch=epoch=4806-step=step=28842-val_PCK_20=0.9101.ckpt"
    model = SignBertModel(ckpt=model_checkpoint, lr=config.lr, head_args=config.head_args)
    
    print ("Setting up logging and checkpoint directories..")
    # Setup logging and checkpoint directories
    logs_dpath = os.path.join(os.getcwd(), "finetune_logs")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logs_dpath, name=config.name)
    ckpt_dirpath = os.path.join(tb_logger.log_dir, "ckpts")
    # Configure model checkpoint callback
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dirpath, 
        save_top_k=5, 
        monitor="val_acc", 
        mode="max", 
        filename="epoch={epoch:02d}-step={step}-{val_acc:.4f}", 
        save_last=True
    )

    print ("Setting up Trainer object..")
    # Setup and configure the Trainer
    trainer = Trainer(
        accelerator="gpu",
        strategy="auto",
        devices=[config.device],
        max_epochs=config.epochs,
        logger=tb_logger,
        callbacks=[ckpt_cb],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        precision=config.precision
    )

    print ("Starting training..")
    trainer.fit(model, datamodule) # Start training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--name", default="test", type=str)
    parser.add_argument("--precision", default="32-true", type=str)
    parser.add_argument("--metadata", default="nslt_2000.json", type=str)
    args = parser.parse_args()

    main(args)