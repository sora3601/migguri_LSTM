import argparse
from migguri.test import test
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model_root', type=str)
parser.add_argument('--model_file', type=str)
parser.add_argument('--test_save', type=str)
parser.add_argument('--test_dataset', type=str)
ARGS = parser.parse_args()

class LoadModelConfig:
    cmd = '>python ' + ' '.join(sys.argv)
    args = ARGS
    model_root = ARGS.model_root
    model_file = ARGS.model_file
    model_path = model_root + '/' + model_file
    test_save = ARGS.test_save
    
    test_dataset_setting = ARGS.test_dataset
    
test(LoadModelConfig, graph=True, rangeY=(0, 20))