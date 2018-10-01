import os
from argparse import ArgumentParser
from matplotlib import use as plot_backend_use
plot_backend_use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('./src')
from lib.eda import dir_avg

parser = ArgumentParser()
parser.add_argument("-r", "--read_path", type=str, default="/hear_and_see/data/padded_spectrograms/trainset/")
parser.add_argument("-w", "--write_path", type=str, default='/hear_and_see/results/dataset_averages/padded_spectrograms')

args = parser.parse_args()


# Create write directory if it doesn't exist
test_dir = '/hear_and_see/results/dataset_averages'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Create write directory if it doesn't exist
test_dir = args.write_path
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

target_classes = [
    'gun_shot','street_music','siren','dog_bark','engine_idling',
    'car_horn','drilling','air_conditioner','jack_hammer','children_playing'
]

for target_class in target_classes:

    src_path = os.path.join(args.read_path,target_class)
    avg_img = dir_avg(src_path)
    avg_n = len([name for name in os.listdir(src_path)])

    plt.figure(figsize=(8, 6), dpi=400)
    plt.imshow(avg_img)
    title_class = target_class.replace('_',' ').title()
    plt.title(title_class+" average spectrogram"+" (n="+str(avg_n)+")")
    plt.ylabel("Frequency (Hz)")
    plt.yticks([])
    plt.xlabel("Time")
    plt.xticks([])
    plt.tight_layout()

    write_name = ''.join([target_class,'.png'])
    filename = os.path.join(args.write_path,write_name)
    plt.savefig(filename)
    plt.close()