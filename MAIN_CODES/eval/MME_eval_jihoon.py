import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import json

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--results_dir', default='/home/onomaai/deeptext_multicaption/jihoon/testing/MAIN_CODES/generated_captions/mme/llava-1.5/aaa_generated_captions.json', type=str)
parser.add_argument('-g', '--gt_dir', default='/home/onomaai/deeptext_multicaption/jihoon/eval_tool/LaVIN/color.txt', type=str)
parser.add_argument('-s', '--save', action='store_true', default=False)


