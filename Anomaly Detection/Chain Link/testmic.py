# USB 마이크로 녹음 및 txt로 변환
# -*- coding: utf-8 -*-
import pyaudio
import wave
#import os
import numpy as np

def c_mic():
    # 변수 선언
    form_1 = pyaudio.paInt16
    chans=1
    samp_rate = 48000
    chunk = 4096
    record_secs = 2     # default record time 10
    dev_index = 2
    wav_output_filename = 'test.wav'
    audio = pyaudio.PyAudio()

    # 텍스트 파일 열기 - 현재는 사용 X
    #filename = '2.txt'
    #f = open(filename, 'wb')
    #오디오 설정
    stream=audio.open(format = form_1,rate=samp_rate,channels=chans, input_device_index = dev_index, input=True, frames_per_buffer=chunk)
    print("recording")
    frames=[]

    # 
    for ii in range(0,int((samp_rate/chunk)*record_secs)):
        data=stream.read(chunk,exception_on_overflow = False)
    # data1 = np.frombuffer(data, 'float32')
    # np.savetxt(f, data1, delimiter=',')
        frames.append(data)

    print("finished recording")

    stream.stop_stream()
    stream.close()
    audio.terminate()
    # 텍스트 파일을 저장하고 닫음 -현재 사용안함
    #f.close  
    #creates wave file with audio read in
    #Code is from the wave file audio tutorial as referenced below

    wavefile=wave.open(wav_output_filename,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

    #plays the audio file
    # os.system("aplay test1.wav")
    
 
    if __name__ == "__main__":
        print("dd")

#while 1:
    #c_mic()