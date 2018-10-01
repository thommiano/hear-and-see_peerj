from argparse import ArgumentParser
import json
import sys
sys.path.append('./src')
from lib.plotting import confusion_matrix as plot_confusion_matrix
from sklearn import metrics


parser = ArgumentParser()
parser.add_argument("-r", "--filepath", type=str, default='/hear_and_see/results/testing/baseline_spectrograms/padded_spectrograms_test_metrics.json')
args = parser.parse_args()

# Load log
with open(args.filepath, 'r') as f:
    model_log = json.load(f)

y_true = model_log['y_true']
y_pred = model_log['y_pred']
class_dict = model_log['class_dict']

cm = metrics.confusion_matrix(y_true,y_pred)

write_path = args.filepath.split('.json')[0]
class_names = [name.replace('_',' ').title() for name in class_dict.values()]
class_names = sorted(class_names)

# font_family='Times New Roman',
plot_confusion_matrix(
    filepath=write_path,
    cm = cm, 
    classes=class_names, 
    normalize=True,
    title='', 
    cmap='Blues',
    filetype='png'
)