import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from librosa.core import istft

def write():
    print("loading txt file")
    get0 = np.loadtxt("./Data/rawData/data.txt")
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    print("saving png files")
    for i in range(int(len(get0)/50000)):
        chromagram_stft = librosa.feature.chroma_stft(y=get0[(i) * 50000:((i + 1) * 50000)], sr=44100)
        librosa.display.specshow(chromagram_stft)
        path =  "./images/test1/" + str(i) + '.png'
        pylab.savefig(path, bbox_inches=None, pad_inches=0)
    pylab.close()
if __name__ == "__main__":
    write()
