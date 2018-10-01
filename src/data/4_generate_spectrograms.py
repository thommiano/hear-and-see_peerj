# Generate spectrograms from tensors

from argparse import ArgumentParser
import os
import sys
sys.path.append('/hear_and_see/src')
from lib.data_processing import save_spectrogram
from librosa import load as libload
from librosa.feature import melspectrogram
sys.path.append("/audio") # Current docker images installs this here
import torchaudio

parser = ArgumentParser()
parser.add_argument("-r", "--read_dir",type=str, default='padded')
parser.add_argument("-w", "--write_dir",type=str, default='padded')
args = parser.parse_args()

src_dir = args.read_dir
src_root = os.path.join('/hear_and_see/data',args.read_dir)
write_root = os.path.join('/hear_and_see/data',args.write_dir)

# You need to create the directories in processed if they don't exist from here.
if not os.path.exists(write_root):
    os.makedirs(write_root)

# Create a list of complete filepaths for every sound file
target_files = []
for dirpath, subdirs, files in os.walk(src_root):
    target_files.extend(os.path.join(dirpath, x) for x in files)

# Convert each .wav to a spectrogram
for filepath in target_files:

    # There's currently a bug with torchaudio.save
    # See: https://github.com/pytorch/audio/issues/14
    #sound_tensor, sr = load_sound_tensor(filepath)
    #spectrogram = create_tensor_spectrogram(sound_tensor,sr)
    sound_array, sr = libload(filepath)
    spectrogram = melspectrogram(y=sound_array, sr=sr)

    if src_root != write_root:

        # Grab full filepath from sound tensor
        datadir_filename = filepath.split(src_dir)[1]
        dir_class_file = datadir_filename.split('/')
        dir_class_file = [name for name in dir_class_file if len(name) > 0]  # filtering out potential ""
        
        # Write dataset directory from filepath if it doesn't exist
        datadir = dir_class_file[0] #e.g., 'training'
        write_dir = os.path.join(write_root,datadir)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        
        # Write class directory from filepath if it doesn't exist
        classdir = dir_class_file[1] #e.g., 'air_conditioner'
        write_dir = os.path.join(write_dir,classdir)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        # Write file
        filename = dir_class_file[2]
        write_path = os.path.join(write_dir,filename)

    else:

        write_path = filepath

    write_path = write_path.split('.wav')[0]
    write_path = ''.join([write_path,'.jpg'])
    save_spectrogram(spectrogram,write_path)

print('Spectrograms complete.')