from argparse import ArgumentParser
import json
import os
import sys
sys.path.append('./src')
from lib.plotting import training_metrics_plot


parser = ArgumentParser()
parser.add_argument("-l", "--log_path", type=str, default='/hear_and_see/data/models/model.json')
parser.add_argument("-w", "--write_path", type=str, default='/hear_and_see/data/results/evaluation')

args = parser.parse_args()

# Load log
with open(args.log_path, 'r') as f:
    model_log = json.load(f)

# Create write directory if it doesn't exist
if not os.path.exists(args.write_path):
    os.makedirs(args.write_path)

# Plot training metrics
filename = 'model_accuracy'
filepath = os.path.join(args.write_path,filename)
training_metrics_plot(
    filepath=filepath,
    training_values=model_log['training_accuracy'],
    validation_values=model_log['validation_accuracy'],
    metric='accuracy'
)

filename = 'model_loss'
filepath = os.path.join(args.write_path,filename)
training_metrics_plot(
    filepath=filepath,
    training_values=model_log['training_loss'],
    validation_values=model_log['validation_loss'],
    metric='loss'
)