# Roll samples by .25, .5, and .75
from argparse import ArgumentParser
import os
import sys
sys.path.append('/hear_and_see/src')
from lib.data_processing import roll_spectrogram
import matplotlib.pylab as plt

parser = ArgumentParser()
parser.add_argument("-d", "--data_src", type=str, default='padded_spectrograms')
parser.add_argument("-s", "--src_root", type=str, default='/hear_and_see/data')
parser.add_argument("-w", "--write_root", type=str, default='/hear_and_see/data/rolled_spectrograms/')
parser.add_argument("-f", "--filter",type=str, default="testset")
args = parser.parse_args()

data_src = args.data_src
src_root = os.path.join(args.src_root,data_src)
write_root = args.write_root
filter = args.filter

# You need to create the directories in processed if they don't exist from here.
if not os.path.exists(write_root):
    os.makedirs(write_root)

# Create a list of complete filepaths for every sound file
target_files = []
for dirpath, subdirs, files in os.walk(src_root):
    target_files.extend(os.path.join(dirpath, x) for x in files)

# Hard-coded list of roll proportions
p_list = [0.25,.50,.75,-.25,-.5,-.75]

# Pad every sound file
for filepath in target_files:

    src_spectrogram = plt.imread(filepath)

    # Grab full filepath from sound tensor
    datadir_filename = filepath.split(data_src)[1]

    dir_class_file = datadir_filename.split('/')
    dir_class_file = [name for name in dir_class_file if len(name) > 0]
    datadir = dir_class_file[0] #e.g., 'training'

    # You can remove this if if you want to roll for all your data
    if datadir == filter:

        # Write dataset directory from filepath if it doesn't exist
        write_dir = os.path.join(write_root,datadir)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        
        # Write class directory from filepath if it doesn't exist
        classdir = dir_class_file[1] #e.g., 'air_conditioner'
        write_dir = os.path.join(write_dir,classdir)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        # Write original spectrogram
        filename = dir_class_file[2]
        write_path = os.path.join(write_dir,filename)
        plt.imsave(write_path,src_spectrogram)

        # Make a spectrogram for each proportion
        for p in p_list:

            spectrogram = roll_spectrogram(src_spectrogram,roll_proportion=p)

            # Write spectrogram
            write_path = write_path.split('.jpg')[0]
            write_path = ''.join([write_path,'_',str(p),'_','.jpg'])
            plt.imsave(write_path,spectrogram)
            #print('Wrote',write_path)

print('Rolling spectrograms complete.')