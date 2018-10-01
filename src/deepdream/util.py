import PIL.Image
from io import StringIO, BytesIO
from IPython.display import clear_output, Image, display
import numpy as np
import time
from datetime import datetime

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    #f = StringIO()
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))
    
def showtensor(a):
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    inp = a[0,:, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std.reshape([1, 1, 3]) * inp + mean.reshape([1, 1, 3])
    inp *= 255
    showarray(inp)
    clear_output(wait=True)

    
''' Part of sound-to-image'''

def writearray(a, filepath,fmt='.jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    im = PIL.Image.fromarray(a)
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S%f")[:-3]
    filepath = filepath + timestamp
    im.save(filepath+fmt)
    
def writetensor(a, filepath):
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    inp = a[0,:, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std.reshape([1, 1, 3]) * inp + mean.reshape([1, 1, 3])
    inp *= 255
    writearray(inp,filepath)

def streamarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    d_buffer = BytesIO()
    PIL.Image.fromarray(a).save(d_buffer, fmt)
    #display(Image(data=f.getvalue()))
    #d_buffer.seek(0)
    #img = Image(data=d_buffer.getvalue())
    #return img
    display(Image(data=d_buffer.getvalue()))
    d_buffer.close()

def streamtensor(a,clear_out=True):
    if clear_out: 
        clear_output(wait=True)
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    inp = a[0,:, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std.reshape([1, 1, 3]) * inp + mean.reshape([1, 1, 3])
    inp *= 255
    #arr = streamarray(inp)
    #display(arr)
    streamarray(inp)
    time.sleep(.0001)