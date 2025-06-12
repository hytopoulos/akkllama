def main():
    import argparse, pickle
    from pprint import pprint
    import os
    import torch
    import numpy as np

    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--HF_HOME', type=str, default=None, help='Huggingface cache directory')
    parser.add_argument('--WANDB_KEY', type=str, default=None, help='Wandb key')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--disable_checkpoint', action='store_true', help='Disable checkpointing')

    # DATASET
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Max sequence length')

    # MODEL
    parser.add_argument('--model', type=str, default="microsoft/codebert-base", help='Model name')
    parser.add_argument('--encoder', action='store_true', help='Specify model is an encoder')
    # parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional attention')
    parser.add_argument('--val_interval', type=int, default=1, help='Epochs between validation')
    parser.add_argument('--log_steps', type=int, default=200, help='Steps between logging')
    parser.add_argument('--val_steps', type=int, default=4000, help='Steps between validation')
    parser.add_argument('--ft_layers', type=int, default=0, help='Number of layers to fine-tune')

    # PARAMETERS
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Warmup steps')
    parser.add_argument('--lora_r', type=int, default=8, help='Lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Lora alpha')
    parser.add_argument('--scheduler', type=str, default='linear', help='Scheduler to use')

    # EVAL
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model (inference)')
    parser.add_argument('--load', type=str, default="", help='Path to load/resume from')

    args = parser.parse_args()

    if args.HF_HOME:
        os.environ["HF_HOME"] = args.HF_HOME
        print(f"HF_HOME set to: {os.environ['HF_HOME']}")
    if args.WANDB_KEY:
        os.environ["WANDB_API_KEY"] = args.WANDB_KEY
        print(f"WANDB_API_KEY set to: {os.environ['WANDB_API_KEY']}")
    if args.load:
        # set default values to the saved arguments
        last_args = pickle.load(open(f"{args.load}/args.pkl", "rb"))
        parser.set_defaults(**vars(last_args))
        # need to reparse to get the new defaults
        args = parser.parse_args()
        print("Loaded args:")
        pprint(vars(args), indent=4, sort_dicts=False)
    if args.evaluate:
        evaluate(args)
    else:
        train(args)

def train(args):
    from agent import Agent
    import wandb

    wandb_config = {
        "model": args.model,
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_len": args.max_seq_len,
        "device": args.device,
        "val_interval": args.val_interval,
        "log_steps": args.log_steps,
        "val_steps": args.val_steps,
        "scheduler": args.scheduler,
        "load": args.load,
        "encoder": args.encoder,
        "dropout": args.dropout,
        "ft_layers": args.ft_layers,
    }

    a = Agent(args)
    run = wandb.init(project='akkadian_final', config=wandb_config)

    a.train()
    run.finish()

def evaluate(args):
    from agent import Agent
    from dataset import EvaCun

    a = Agent(args)
    # test_dataset = EvaCun('akk_test.csv', a.tokenizer, a.args)
    val_acc, val_mrr = a.test(a.val_loader)
    # test_acc = a.evaluate(test_dataset.df, args.salience)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation MRR: {val_mrr:.4f}")
    # print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
