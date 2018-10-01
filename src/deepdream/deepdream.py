import numpy as np
import torch
from deepdream.util import showtensor, streamtensor, writetensor
import scipy.ndimage as nd
from torch.autograd import Variable
import scipy
import random


# -------------------------------------
# ~*~*~*~*~ Objective guides~*~*~*~*~*~
# -------------------------------------

# def obective_neuron_L2(dst, guide_features,neuron):
#     return dst[:,:,:,neuron]

def objective_L2(dst, guide_features, neuron_unit):
    return dst.data[:,:,:,:]


def objective_deepdream(dst, guide_features, neuron_unit):
    return dst.data[:,:,:,:]**2


def objective_channel(dst, guide_features, neuron_unit):
    return dst.data[:,neuron_unit,:,:]

# def objective_neuron(dst, guide_features, neuron_unit):
#     return dst.data[:,neuron_unit,h,w]

def objective_guide(dst, guide_features, neuron_unit):
    
    
    # Pytorch implementation
    # ----------------------
    
    x = dst.data[0].clone()
    y = guide_features.data[0]
    
    ch, h, w = x.shape
    x = x.view(ch,-1).transpose_(0,1) # (c，w*h)
    y = y.view(ch,-1)
    
    # matrix-wise dot product, for 1d use torch.dot()
    A = torch.mm(x,y) 
    _,max_vals = torch.max(A,1) # argmax
    
    # Select max value indices, reshape and a dim
    result = y[:,max_vals].view(ch,h,w).unsqueeze_(0)
    return result
    
    # Original numpy implementation
    # -----------------------------
    
#     x = dst.data[0].cpu().numpy().copy()
#     y = guide_features.data[0].cpu().numpy()
#     ch, w, h = x.shape
#     x = x.reshape(ch,-1).T    # (c，w*h)
#     y = y.reshape(ch,-1)      # (c, w*h)
#     A = x.dot(y) # compute the matrix of dot-products with guide features
#     max_vals = A.argmax(1)
#     result = y[:,max_vals] # select ones that match best
#     result = torch.Tensor(np.array([result.reshape(ch, w, h)], dtype=np.float)).cuda()
#     return result


def objective_guide_channel(dst, guide_features, neuron_unit):
    
#     dst = dst.data[:,neuron_unit,:,:]
#     guide_features = guide_features.data[:,neuron_unit,:,:]

    x = dst.data[0,neuron_unit,:,:].clone()
    y = guide_features.data[0,neuron_unit,:,:]
    
    # Need to update this to account for 2d tensor
    h, w = x.shape
    x = x.view(h,-1).transpose_(0,1) # (c，w*h)
    y = y.view(h,-1)
        
    print(x.shape)
    print(y.shape)
    
    # matrix-wise dot product, for 1d use torch.dot()
    A = torch.mm(x,y) 
    _,max_vals = torch.max(A,1) # argmax
    
    # Select max value indices, reshape and a dim
    result = y[:,max_vals].view(h,w).unsqueeze_(0)
    return result

# --------------------------------
# ~*~*~*~*~ Image zoom ~*~*~*~*~*~
# --------------------------------

# def tensor_roll(tensor, shift, axis):
#     if shift == 0:
#         return tensor

#     if axis < 0:
#         axis += tensor.dim()

#     dim_size = tensor.size(axis)
#     after_start = dim_size - shift
#     if shift < 0:
#         after_start = -shift
#         shift = dim_size - abs(shift)

#     before = tensor.narrow(axis, 0, dim_size - shift)
#     after = tensor.narrow(axis, after_start, shift)
#     return torch.cat([after, before], axis)


def resize(img,h_scale,w_scale):
    img = scipy.ndimage.zoom(img[0], (1,h_scale,w_scale))
    img = np.expand_dims(img, axis=0)
    return img


def crop_zoom(img,zoom_interval,zoom_orientation,nproll,oscillator=None):
    h,w = img.shape[-2:]
    
    if nproll is not None:
        # int(zoom_interval/2) clips down middle when using strings
        roll_n = random.randint(0,5)
        #roll_n = perlin(zoom_interval)
        if zoom_orientation is 'right':
            img = np.roll(img,-roll_n)
        else:
            img = np.roll(img,roll_n)

    if zoom_orientation is not None:
        if zoom_orientation=='center':
            img = img[:,:,zoom_interval:-zoom_interval,zoom_interval:-zoom_interval]
        elif zoom_orientation=='strings':
            img = img[:,:,zoom_interval:-zoom_interval,0:zoom_interval]
        elif zoom_orientation=='oscillate':
            img = img[:,:,zoom_interval:-zoom_interval-oscillator,zoom_interval:-zoom_interval+oscillator]
        elif zoom_orientation=='sin':
            try:
                i += 1
            except:
                i = 0
                sin_range = [i for i in np.arange(0,zoom_interval,.1)]

            zi = int(abs(np.sin(i)*100))*10
            if zi <= 0:
                zi = 1
            img = img[:,:,zi:-zi,zi:-zi]
        else:
            img = img[:,:,zoom_interval:-zoom_interval,zoom_interval:-zoom_interval]

        h_scale = h/img.shape[2]
        w_scale = w/img.shape[3]
        img = resize(img,h_scale,w_scale)
    
    return img

# ----------------------------------
# ~*~*~*~*~ Optimization ~*~*~*~*~*~
# ----------------------------------

def zoom_step(X, model,end_layer,neuron_unit,write=False,use_gpu=True,stream_out=False, **kwargs):

    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])

    learning_rate = kwargs.pop('lr', 5.0)
    max_jitter = kwargs.pop('max_jitter', 100)
    num_iterations = kwargs.pop('num_iterations', 100)
    show_every = kwargs.pop('show_every', 25)
    #end_layer = kwargs.pop('end_layer', 3)
    object = kwargs.pop('objective', objective_L2)
    guide_features = kwargs.pop('guide_features', None)
    
    for t in range(num_iterations):
        
        # Add jitter
        ox, oy = np.random.randint(-max_jitter, max_jitter + 1, 2)
        X = np.roll(np.roll(X, ox, -1), oy, -2)
        
        model.zero_grad()
        X_tensor = torch.Tensor(X)
        
        if use_gpu: X_Variable = Variable(X_tensor.cuda(), requires_grad=True)
        else: X_Variable = Variable(X_tensor.cpu(), requires_grad=True)

        # -----------------------------------------
        # ~*~*~*~*~*~ Optimization step ~*~*~*~*~*~
        # -----------------------------------------
        
        act_value = model.forward(X_Variable,end_layer,neuron_unit)
        diff_out = object(act_value, guide_features, neuron_unit)
        
        if object is objective_channel:
            act_value = act_value[:,neuron_unit,:,:]
        #if object is objective_neuron:
        #    act_value = act_value[:,neuron_unit,h,w]
            
        act_value.backward(diff_out)

        #learning_rate_ = learning_rate / np.abs(X_Variable.grad.data.cpu().numpy()).mean()
        learning_rate_ = learning_rate / torch.mean(torch.abs(X_Variable.grad.data))
    
        X_Variable.data.add_(X_Variable.grad.data * learning_rate_)
        
        # Convert this to tensor instead of using np (do on GPU)
        X = X_Variable.data.cpu().numpy()
        X = np.roll(np.roll(X, -ox, -1), -oy, -2)
        #X[0, :, :, :] = np.clip(X[0, :, :, :], -mean / std, (1. - mean) / std)
        X[0] = np.clip(X[0], -mean / std, (1. - mean) / std)

            
        if stream_out:
            if t == 0 or (t + 1) % show_every == 0:
                if not write:
                    streamtensor(X)
                else:
                    writetensor(X,filepath=filepath)
            
    return X


# ---------------------------------------
# ~*~*~*~*~ Generator wrapper ~*~*~*~*~*~
# ---------------------------------------

def dream(model, base_img, end_layer,neuron_unit,zoom_depth=20,zoom_interval=10,zoom_orientation=None,
               nproll=None,fade_alpha=.3, octave_n=4, octave_scale=1.4, end='', write=False,filepath=None,use_gpu=True,
               stream_out=False,zoom_echo=False,awaken=False,awaken_alpha=0.1,clear_out=True,**step_params):
    
    if clear_out is None:
        clear_out = True
    receive_socket = False

    root_img = base_img.copy()
    
    if not write:
        streamtensor(root_img,clear_out=clear_out)
    else:
        writetensor(root_img,filepath=filepath)
        
    #streamtensor(root_img,clear_out=clear_out)
        
    #streamtensor(resize(root_img,3,3))

    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    octaves = [base_img]
    for i in range(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        base_img = octave_base + detail
    
    loop_count=0
    j=0
    oscillator = 1
    oscillator_direction = 'positive'
    
    current_img_idx = 0
    #while True:
    for frames in range(0,zoom_depth):
        
        try:
#             if receive_socket:
                
            # ---- Zoom Dream --------------#        
#             if zoom_echo:
#                 if write:
#                     writetensor(base_img,filepath=filepath)
#                 else:
#                     streamtensor(base_img)
                #streamtensor(base_img)

            base_img = zoom_step(base_img, model, end_layer=end_layer,neuron_unit=neuron_unit,**step_params,use_gpu=use_gpu,stream_out=stream_out)

            #streamtensor(base_img,clear_out=clear_out)
            if not write:
                streamtensor(base_img,clear_out=clear_out)
            else:
                writetensor(base_img,filepath=filepath)
            
            base_img = crop_zoom(base_img,zoom_interval,zoom_orientation,nproll,oscillator) * ((1.0 - fade_alpha) + base_img * fade_alpha)
                
            #WHOAAA:
            #base_img = crop_zoom(base_img,zoom_interval,zoom_orientation) * (1.0 - fade_alpha) + base_img * fade_alpha - 150

            #if awaken:
                #if not z % 3:
                    #base_img = crop_zoom(base_img,zoom_interval,zoom_orientation) * (1.0 - fade_alpha) + base_img * fade_alpha
            #if awaken:
                #if not z % 5:
                    #base_img = crop_zoom(base_img,zoom_interval,zoom_orientation) * (1.0 - awaken_alpha) + awaken_alpha * root_img
                    #prior_img = base_img.copy()
                #if i % 2:
                    # interesting effect: root_img * prior_img, instead of awaken_alpha * prior_img
                    #base_img = crop_zoom(base_img,zoom_interval,zoom_orientation) * (1.0 - awaken_alpha) + awaken_alpha * prior_img
                    #base_img = crop_zoom(base_img,zoom_interval,zoom_orientation) * (1.0 - awaken_alpha) + awaken_alpha * root_img

            #if not i % 2:
            #        prior_img = base_img.copy()
            #base_img = crop_zoom(base_img,zoom_interval,zoom_orientation) * (1.0 - awaken_alpha) + awaken_alpha * prior_img

            # regular zoom
            #base_img = crop_zoom(base_img,zoom_interval,zoom_orientation)
            # ---- Sound control --------------#   
#             if receive_socket:
#                 if predicted_sound == 'Dog Bark':
#                     #base_img = crop_zoom(base_img,zoom_interval,zoom_orientation) * (1.0 - predicted_prob) + (predicted_prob) * root_img
#                     base_img = root_img.copy()
        
            loop_count += 1
            
#             if loop_count % 5 == 0: 
#                     #loop_count=0
#                 end_layer+=1
#                 if end_layer == 4:
#                     end_layer = 0
                #lr = 0.51
#             else:
#                 end_layer = 3
#                 lr = 0.21
            
            # Class update test
            # -----------------
            # This is too much of just an image "behind" the deep dream.
            # The deep dream itself needs to be a response to the sounds...
#             if not loop_count % 10: 
#                 base_img = (base_img) + (img_bucket[current_img_idx]*.5)
#                 #use np.array roll
#                 current_img_idx += 1
#                 if current_img_idx == len(img_bucket)-1:
#                     current_img_idx = 0
            
            # Layer oscillator
            # ----------------
            if zoom_orientation == 'oscillate':
            
                if (oscillator + zoom_interval) > (base_img.shape[2]/2 - 20):
                    oscillator_direction = 'negative'
                if (oscillator + zoom_interval) < 6:
                    oscillator_direction = 'positive'


                if loop_count % 2 == 0: 
                    #loop_count=0
                    #end_layer=1
                    if oscillator_direction is 'positive':
                        oscillator += 4
                    else:
                        oscillator -= 4

                    #j+=1
                    #if not j % 2:
                        #zoom_orientation = 'center'

                else:
                    #end_layer = 3
                    if oscillator_direction is 'positive':
                        oscillator -= 2
                    else:
                        oscillator += 2

                    #if not j % 3:
                        #zoom_orientation = 'center'
                        #base_img = crop_zoom(base_img,zoom_interval,zoom_orientation) * (1.0 - fade_alpha) + base_img * fade_alpha - 10


        except KeyboardInterrupt:
            if receive_socket:
                s.close()
            break
            