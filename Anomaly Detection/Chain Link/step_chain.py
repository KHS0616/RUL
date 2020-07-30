import pandas as pd
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
import librosa
plt.ioff() 
class step_chain:
    # 클래스 초기화
    def __init__(self):
        # 디렉토리와 파일 변수 선언
        self.directoryname = "Data"
        self.filename = "C_33_3802-688-9_200421_1415N_D.raw.txt"
        self.file_full_path = "./Data/rawData/data.txt"#self.directoryname+"/"+ self.filename
        self.out_file_name = self.directoryname+"/"+self.filename[0:30] + "_A.txt"

        # 샘플링 주파수, 계산할 체인 링크 개수, pitch 길이 변수 선언
        self.Fs = 50000
        self.link_count = 6
        self.link_pitch = 406.41

        # 초당 스텝 1개가 움직인 거리
        self.chain_velocity = 421.2

        # 첫 피크 인덱스 번호 설정
        self.peak_num = 12
    
    # 데이터 불러오기
    def load_data(self):
        print("Loading Data...")
        temp_df = pd.read_table(self.file_full_path)
        self.happy_go_n = np.array(temp_df.values).flatten()
        self.data = self.happy_go_n.tolist()
    
    # 그래프 0점 이동
    def move_graph(self):
        print("Moving Graph...")
        self.aver = np.mean(self.data)
        self.iot7_BP = self.data - self.aver

    # 샘플링 주파수에 따른 X축을 시간축으로 변경
    def X_trans_time(self):
        t = 1/self.Fs
        row = len(self.iot7_BP)
        t1 = t*row
        self.T = np.arange(0, t1, t)
    
    # 3kh ~ 4kh 대역폭으로 대역통과필터 설정 후 필터링
    def bandpass(self, start = 3000, end = 4000, fs = 50000):
        print("Filtering Data...")
        sos = signal.butter(5, [start, end], "bp", fs=fs, output="sos")
        filtered = signal.sosfilt(sos, self.iot7_BP)
        self.filtered = filtered

    # 자기상관함수
    def auto_correlate(self):
        print("Autocorrelating Data...")
        correlate_result = librosa.autocorrelate(self.filtered)
        self.correlate_result = correlate_result

    # 피크 값 찾기
    def find_peaks(self):
        print("Finding peak...")
        result = signal.find_peaks(self.correlate_result, distance=((self.link_pitch/self.chain_velocity)-0.01)*self.Fs)#0.7*self.Fs
        # 원본 X축, 시간축, 피크 값을 저장
        self.row_X = result[0]
        self.locs, self.k = [], []
        for i in range(len(result[0])):
            self.locs.append(self.T[result[0][i]])
            self.k.append(self.iot7_BP[result[0][i]])
    
    # 피크 간격 구하기
    def peak_interval(self):
        print("Calculating peak interval...")
        r_link_count = self.peak_num+self.link_count
        locs_re = []
        for i in np.arange(self.peak_num-1, r_link_count, 1) :
            locs_re.insert(i - (self.peak_num-1), self.locs[i])
        locs_re = np.array(locs_re)
        self.peakInterval = np.diff(locs_re)
    
    # 몰라도 되는 과정
    def math(self):
        # 모터 속도 = RPM(1분동안 회전수) * 모터효율
        mv = 1745*0.902
        # 감속비
        r = 34.177
        # 작은스프라켓(구동축) RPM(1분동안 회전수)
        small_sp_rpm = mv/r
        # 작은스프라켓(구동축) Hz
        small_sp_hz = small_sp_rpm/60
        # 큰스프라켓(피구동축) Hz = (작은스프라켓 Hz * 작은스프라켓 잇수/큰스프라켓 잇수) * 구동체인 효율
        big_sp_hz = (small_sp_hz*18/71)*0.9922
        # 체인 pitch 설계치
        pitchll= 135.47
        # 원주(mm) = 스텝체인스프라켓 피치원 * PI
        cc = 694.4 * math.pi
        # 원주당 피치갯수
        cc_pitch_num = cc/pitchll
        # 1초에 지나가는 피치 수
        thru_pitch_per_second = cc_pitch_num*big_sp_hz
        # 1초당 이동길이(mm)
        length_per_sceond = thru_pitch_per_second*pitchll
        # 데이터 1개당 이동길이(mm)
        self.length_per_hz = length_per_sceond*(1/self.Fs)

    # 신율을 구하기 위한 간격 설정
    def sin_length(self):
        Link_Length11 = []
        for i in range(0, np.prod(self.peakInterval.shape)):
            Link_Length11.insert(i,(self.peakInterval[i]/0.00002)*self.length_per_hz)
        Link_Length11 = np.array(Link_Length11)
        self.Link_Length111=Link_Length11.reshape(len(Link_Length11), 1)

    # 신율 계산
    def cal_sin(self):
        print("Calculating sin...")
        # 신율 한도 계산
        elongation= 0
        for i in range(0, np.prod(self.Link_Length111.shape)):
            elongation=elongation+self.Link_Length111[i]

        link_pitch=self.link_pitch*self.link_count
        # 신율 = ((늘어난 길이(mm))/설계치 길이)*100
        self.elongation_result=((elongation-link_pitch)/link_pitch)
    
    # 파일 저장
    def save_file(self):
        print("Saving Data...")
        Length_Link_Row = np.arange(1, self.link_count+1)
        Length_Link_Row = np.array(Length_Link_Row)
        Length_Link_Row=Length_Link_Row.reshape(len(Length_Link_Row), 1)
        # text file write
        fid = open(self.out_file_name, 'wt')
        fid.write('Chain')
        fid.write('LinkLength')
        catN = np.concatenate((Length_Link_Row, self.Link_Length111), axis=1)
        for row in range(0, self.link_count):
            fid.write(str(catN[row,:]))
        fid.write('TotalLength')
        fid.write(str(sum(self.Link_Length111, 1)))
        fid.write('Elongation')
        fid.write(str(self.elongation_result))
        fid.close()

    # 그래프를 이용한 시각화
    def show_graph(self, mode = "autocorrelate"):
        if mode == "row":
            plt.clf()
            plt.plot(self.iot7_BP)
            plt.savefig('GG.png')
        elif mode == "bandpass":
            plt.clf()
            plt.plot(self.filtered) 
            plt.savefig('GG.png')
        elif mode == "autocorrelate":
            plt.plot(self.correlate_result)
            fig = plt.gcf()
            fig.savefig('GG.png')
        elif mode == "peak":
            plt.clf()
            plt.plot(self.correlate_result)
            plt.plot(self.row_X, self.k, "x") 
            plt.savefig('GG.png')
        else:
            print("mode typing error!")
        #plt.show()

    # 실행 함수(경로 자동 설정)
    def Start(self):
        self.load_data()
        self.move_graph()
        self.X_trans_time()
        self.bandpass()
        self.auto_correlate()
        self.find_peaks()
        self.peak_interval()
        self.math()
        self.sin_length()
        self.cal_sin()
        self.save_file()