import torch
import argparse
import warnings
import torch.nn as nn

from src.models import HyPINO
from src.models.utils import has_batchnorms
from src.data.datamodule import PDEDataModule

MODEL_REGISTRY = {
    "hypino": HyPINO,
}

def main():
    parser = argparse.ArgumentParser(description='Training function parameters')

    parser.add_argument('--checkpoint', type=str, default=None, help='Path to pre-trained model')
    parser.add_argument('--resume', action='store_true', help='Resume training including optimizer state')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--accumulate_grad_batches', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--model', type=str, choices=MODEL_REGISTRY.keys(), default='hypino', help='Model architecture to use')
    parser.add_argument('--supervised_only', action='store_true', help='Use only supervised data in the datamodule')
    parser.add_argument('--devices', type=str, default='-1' if torch.cuda.is_available() else 'cpu', help='Device to use for training: -1 for all GPUs, an integer for a specific GPU, or "cpu"')

    args = parser.parse_args()

    # Resolve device setting
    if args.devices == '-1':
        devices = -1  # all GPUs
    elif args.devices.isdigit():
        devices = int(args.devices)
    else:
        devices = args.devices

    datamodule = PDEDataModule(
        data_size=args.batch_size * args.accumulate_grad_batches * 1_000,
        batch_size=args.batch_size,
        supervised_only=args.supervised_only,
    )

    ModelClass = MODEL_REGISTRY[args.model.lower()]

    if args.checkpoint is not None:
        print('Loading model from checkpoint:', args.checkpoint)
        checkpoint = args.checkpoint
        model = ModelClass.load_from_checkpoint(
            args.checkpoint,
            strict=False,
            ignore_size_mismatch=True
        )
    else:
        model = ModelClass()
        checkpoint = None

    if has_batchnorms(model):
        print('Model has batchnorms. Converting to SyncBatchNorm...')
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.fit(
        datamodule,
        epochs=args.epochs,
        lr=args.learning_rate,
        name=args.model,
        accumulate_grad_batches=args.accumulate_grad_batches,
        checkpoint=checkpoint if args.resume else None,
        devices=devices,
    )

if __name__ == "__main__":
    main()
