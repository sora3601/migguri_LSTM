import argparse
from migguri.train import train
from migguri.train_config import Config_MAMA, Config_BABY_MIDDLE

parser = argparse.ArgumentParser()
parser.add_argument('--gsi_bodylength', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--past_history', type=int)
parser.add_argument('--future_target', type=int)
parser.add_argument('--step', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--save', type=str)
ARGS = parser.parse_args()

assert ARGS.gsi_bodylength.lower() in ['gsi', 'body_length']

class Config(Config_MAMA if ARGS.gsi_bodylength.lower() == 'gsi' else Config_BABY_MIDDLE):

  root = ARGS.dataset
  save_root = ARGS.save
  past_history = ARGS.past_history
  future_target = ARGS.future_target
  STEP = ARGS.step
  STEP_TO_FUTURE = STEP
  EPOCHS = ARGS.epoch
  BATCH_SIZE = ARGS.batch_size
  
CONFIG = Config()
CONFIG.display()
train(CONFIG)