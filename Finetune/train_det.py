import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from methods.det_model import FinetuneDetector
import torch
import yaml
import os
import datetime
import argparse
from pytorch_lightning import LightningModule, seed_everything
from datasets.det_dataset import RSNADetectionDataset
from datasets.data_module import DataModule
from datasets.transforms import DetectionDataTransforms
from methods.backbones.detector_backbone import ResNetDetector
from dateutil import tz
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning, 
                        message=".*torch.distributed._sharded_tensor.*")

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning Training')
    parser.add_argument("--dataset", type=str, default="rsna", help="Dataset to use: rsna")
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--config', type=str, default='../configs/rsna.yaml', help='Path to config file')
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for dataloader")
    parser.add_argument("--data_pct", type=float, default=0.01, help="Percentage of data to use")
    parser.add_argument("--max_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument('--ckpt_dir', type=str, default='../data/ckpts', help='Directory to save model checkpoints')
    parser.add_argument('--logger_dir', type=str, default='../data/log_output', help='Directory to save logs')
    return parser.parse_args()

if __name__ == '__main__':
    print()
    print('-----' * 10) 
    seed = 42
    seed_everything(seed)
    args = parse_args()
    config = load_config(args.config)


    if args.dataset == "rsna":
  
        datamodule = DataModule(dataset=RSNADetectionDataset, 
                                config=config, collate_fn=None,
                                transforms=DetectionDataTransforms, data_pct=args.data_pct,
                                batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        print("Dataset not supported")
        exit()

    # Initialize the encoder part and start debugging

    img_encoder = ResNetDetector("resnet_50")

    # Get img_encoder model state_dict keys for inspection
    img_encoder_keys = list(img_encoder.model.state_dict().keys())

    # Print out the first 3 keys and total number of keys
    print("Image encoder keys before loading checkpoint:")
    print("First 3 keys:", ", ".join(img_encoder_keys[:3]))  # Display only the first 3 keys
    print("Total number of keys:", len(img_encoder_keys))  # Print the total number of keys

    # import pdb; pdb.set_trace()
    
    # Here you may need to modify to different paper's checkpoint
    # find the encoder part's state_dict keys in the checkpoint
    # Check if pretrained weights should be loaded
    if config['det']['pretrained']:
        # Load the checkpoint file
        ckpt = torch.load(config['det']['checkpoint'])
        
        # Dictionary to store modified checkpoint weights
        ckpt_dict = {}

        # Process each key-value pair in the checkpoint
        for k, v in ckpt["state_dict"].items():
            if k.startswith("img_encoder_q.model"):
                # Remove "img_encoder_q.model" prefix
                new_k = ".".join(k.split(".")[2:])
                ckpt_dict[new_k] = v

        # Print the keys of the ckpt_dict
        print("Keys in ckpt_dict:")
        print("Number of keys in ckpt_dict:", len(ckpt_dict))  # Print only the number of keys in ckpt_dict

        # Find and print common keys between ckpt_dict and img_encoder's state_dict
        img_encoder_keys_set = set(img_encoder.model.state_dict().keys())
        common_keys = img_encoder_keys_set.intersection(ckpt_dict.keys())
        print("Common keys between ckpt_dict and img_encoder:", len(common_keys))
        
        # Load the adjusted checkpoint dictionary into img_encoder
        img_encoder.model.load_state_dict(ckpt_dict, strict=False)

    for param in img_encoder.parameters():
        param.requires_grad = False

    # import pdb; pdb.set_trace()
    ## initilize detector model
    model = FinetuneDetector(img_encoder=img_encoder, config=config)


    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(args.ckpt_dir, f"{'FinetuneDet'}/{args.dataset}/{extension}")
    logger_dir = os.path.join(args.logger_dir, f"{'FinetuneDet'}/{args.dataset}/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)
    print('ckpt_dir: ', ckpt_dir)
    print('args.ckpt_dir:', args.ckpt_dir)
    
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        callbacks=callbacks,
        logger=pl.loggers.WandbLogger( project='FinetuneDet', name=f"{args.dataset}_{args.data_pct}_{extension}"),
        strategy='ddp', #ddp, ddp_spawn
        )

    model.training_steps = model.num_training_steps(trainer, datamodule)

    
    # train
    # import pdb; pdb.set_trace()
    trainer.fit(model, datamodule)
    # test
    trainer.test(model, datamodule, ckpt_path="best")

