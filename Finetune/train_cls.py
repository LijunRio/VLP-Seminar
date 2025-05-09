import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from methods.cls_model import FinetuneClassifier
import torch
import yaml
import os
import datetime
import argparse
from pytorch_lightning import LightningModule, seed_everything
from datasets.cls_dataset import RSNAImageClsDataset, ChexPertImageClsDataset
from datasets.data_module import DataModule
from datasets.transforms import DataTransforms
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
    parser.add_argument("--dataset", type=str, default="chexpert", help="Dataset to use: chexpert, rsna")
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--config', type=str, default='../configs/chexpert.yaml', help='Path to config file:chexkpert.yaml, rsna.yaml')
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for dataloader")
    parser.add_argument("--data_pct", type=float, default=1, help="Percentage of data to use")
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
  
        datamodule = DataModule(dataset=RSNAImageClsDataset, 
                                config=config, collate_fn=None,
                                transforms=DataTransforms, data_pct=args.data_pct,
                                batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == "chexpert":
        datamodule = DataModule(dataset=ChexPertImageClsDataset, 
                                config=config, collate_fn=None,
                                transforms=DataTransforms, data_pct=args.data_pct,
                                batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        print("Dataset not supported")
        exit()
    
    if config['cls']['pretrained']:
        checkpoint_path = config['cls']['checkpoint']
        checkpoint = torch.load(checkpoint_path)
        model = FinetuneClassifier(config)
        model_state_dict = model.state_dict() # need to check your model's state_dict, and make sure the checkpoint's state_dict name is the same as the model here.
        common_keys = set(checkpoint['state_dict'].keys()).intersection(set(model_state_dict.keys()))
        print(f"Number of common keys between checkpoint and model: {len(common_keys)}")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model = FinetuneClassifier(config)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(args.ckpt_dir, f"{'FinetuneCLS'}/{args.dataset}/{extension}")
    logger_dir = os.path.join(args.logger_dir, f"{'FinetuneCLS'}/{args.dataset}/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)
    print('ckpt_dir: ', ckpt_dir)
    print('logger_dir:', logger_dir)
    
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
        logger=pl.loggers.WandbLogger( project='FinetuneCLS', name=f"{args.dataset}_{args.data_pct}_{extension}",dir=logger_dir),
        strategy='ddp', #ddp, ddp_spawn
        )

    model.training_steps = model.num_training_steps(trainer, datamodule)

    
    # train
    # import pdb; pdb.set_trace()
    trainer.fit(model, datamodule)
    # test
    trainer.test(model, datamodule, ckpt_path="best")

