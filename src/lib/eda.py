import librosa
from matplotlib import use as plot_backend_use
plot_backend_use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image



def dir_avg(src_path):
    '''Generate the average image given a directory of images.'''
    # Access all PNG files in directory
    allfiles=os.listdir(src_path)
    imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]
    # Assuming all images are the same size, get dimensions of first image
    img_open = os.path.join(src_path,imlist[0])
    print(img_open)
    w,h=Image.open(img_open).size
    N=len(imlist)
    # Create a numpy array of floats to store the average (assume RGB images)
    arr=np.zeros((h,w,3),np.float)
    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        img_open = os.path.join(src_path,im)
        imarr=np.array(Image.open(img_open),dtype=np.float)
        arr=arr+imarr/N
    # Round values in array and cast as 8-bit integer
    arr=np.array(np.round(arr),dtype=np.uint8)
    return arr


def generate_histograms(save_path,input_dict,color_key):
    
    for key in input_dict.keys():
        
        plt.figure(figsize=(8, 6), dpi=400) # to create a new fig for each loop
        plt.hist(input_dict[key],color=color_key[key])
        
        n_obs = len(input_dict[key])
        title_class = key.replace('_',' ').title()
        plt.title(' '.join([title_class,"excerpt lengths","(n=" + str(n_obs) + ")"]))
        plt.ylabel("Number of observations")
        plt.xlabel("Length of excerpt in seconds")
        plt.tight_layout()
        filename = ''.join([key,'.png'])
        filepath = os.path.join(save_path,filename)
        plt.savefig(filepath)
        plt.close()