from scipy.io.wavfile import read
import numpy as np

from decimal import Decimal

def wave_to_txt():
    # wav파일을 읽고 텍스트형식으로 저장
    print("loading wav file")
    input_data = read("test.wav")
    audio = input_data[1]

    # 저장할 파일 이름 저장
    filename = "./Data/rawData/data.txt"

    # 파일 저장
    print("saving txt file")
    with open(filename, "wb") as fid:
        np.savetxt(fid, audio, delimiter=",")
