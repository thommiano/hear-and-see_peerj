from librosa.core import get_duration
import os
import sys
sys.path.append('./src')
from lib.eda import generate_histograms
sys.path.append("/audio") 
import torchaudio

read_path = "/hear_and_see/data/torchified/trainset"
save_path = "/hear_and_see/results/urbansound8k_histograms"

# Create write directory if it doesn't exist
test_dir = save_path
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

class_colors = {
    "gun_shot":"#b00036",
    "street_music":"#ff8765",
    "siren":"#c35d03",
    "dog_bark":"#c2c331",
    "engine_idling":"#007008",
    "car_horn":"#00a069",
    "drilling":"#00b4f5",
    "air_conditioner":"#0164cd",
    "jack_hammer":"#bfa2ff",
    "children_playing":"#e44caa"
}

# TODO: Re-write below to be more efficient
duration_log={}
for root, dirs, _ in os.walk(read_path):  
    for class_dir in dirs:
        print(class_dir)
        files = os.listdir(os.path.join(read_path, class_dir))
        # initialize new class in duration_log
        duration_log[class_dir] = []
        for file in files:
            sound_in = os.path.join(read_path,class_dir,file)
            sound_tensor, sr = torchaudio.load(sound_in)
            duration = get_duration(sound_tensor.numpy()[:,0], sr=sr)
            duration_log[class_dir].append(duration)

generate_histograms(save_path,duration_log,class_colors)