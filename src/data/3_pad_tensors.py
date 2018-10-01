# Applying padding to sample less that 4 seconds

# With the librosa implementation this takes about 30 minutes.
# Despite the torchaudio bug, it may be worth using it here since
# you use librosa in the next step.

import os
import sys
sys.path.append('/hear_and_see/src')
from lib.data_processing import pad_sound_array
from librosa import load as libload
from librosa.output import write_wav
sys.path.append("/audio") # Current docker images installs this here
import torchaudio


src_dir = 'torchified'
src_root = os.path.join('/hear_and_see/data',src_dir)
write_root = '/hear_and_see/data/padded'

# You need to create the directories in processed if they don't exist from here.
if not os.path.exists(write_root):
    os.makedirs(write_root)

# Create a list of complete filepaths for every sound file
target_files = []
for dirpath, subdirs, files in os.walk(src_root):
    target_files.extend(os.path.join(dirpath, x) for x in files)

# Pad every sound file
for filepath in target_files:

    # There's currently a bug with torchaudio.save
    # See: https://github.com/pytorch/audio/issues/14
    #sound_tensor, sr = load_sound_tensor(filepath)
    #padded_sound_tensor = pad_sound_tensor(sound_tensor,sr)
    #torchaudio.save(filepath,src=padded_sound_tensor,sample_rate=sr)
    sound_array, sr = libload(filepath)
    padded_sound_array = pad_sound_array(sound_array,sr)

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
    write_wav(write_path,padded_sound_array, sr)

print('Padding complete.')