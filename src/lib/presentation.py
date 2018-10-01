import librosa, librosa.display

def stream_spectrogram(sound_in, sr):
    
    y = sound_in
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    buffer_ = BytesIO()
    
    plt.ioff()
    librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
    plt.axis('off') # Removes black border
    plt.tight_layout()
    plt.savefig(buffer_,bbox_inches='tight',pad_inches=-0.05,transparency=True, format='jpg')
    image = Image.open(buffer_)
    plt.close()
    buffer_.seek(0)
    #ar = np.asarray(image)
    return image
    buffer_.close()
    
def classify_sound(image,resnet):
    
    resnet.eval()
    predictions = {}
    # Try your own image here.
    inputVar =  Variable(transform(image).unsqueeze(0))
    predictions = resnet(inputVar.cpu())

    probs, indices = (-F.softmax(predictions.cpu())).data.sort()
    probs = (-probs).numpy()[0][:10]; indices = indices.numpy()[0][:10]
    preds = {class_dict[idx]:prob for (prob, idx) in zip(probs,indices)}
    preds = sorted(preds.items(), key=operator.itemgetter(1), reverse=True)
    
    #preds = [class_dict[idx] + ': ' + str(prob) for (prob, idx) in zip(probs, indices)]
    #print(preds)
    #plt.title('\n'.join(preds))
    #plt.imshow(image);
    return preds