# 관련 라이브러리 불러오기
import pandas as pd
import numpy as np
#from scipy.signal import find_peaks
import math
from scipy import signal
import matplotlib.pyplot as plt
import copy
import librosa
# 메인 클래스
# 데이터를 불러오는 것부터 파일 작성까지 기본적으로 제공된 과정
# 각 중요 기능별로 구간을 나누어 함수화
# 각각의 변수들은 객체 생성후 객체를 통해 불러오기 가능
# main 함수는 기본으로 설정된 default 값을 기반으로 파일 작성까지 자동적으로 실행
# 위에서부터 순서대로 함수를 진행해야 오류가 발생하지 않는다.(작동하더라도 default 값으로 동작)
class chain_main_class:
    # 클래스 초기화
    def __init__(self):
        # 디렉토리와 파일 변수 선언
        self.directoryname = "Data"
        self.filename = "C_33_3802-688-9_200421_1415N_D.raw.txt"
        self.file_full_path = self.directoryname+"/"+ self.filename
        self.out_file_name = self.directoryname+"/"+self.filename[0:30] + "_A.txt"

        # 샘플링 주파수, 계산할 체인 링크 개수, pitch 길이 변수 선언, 필터링할 때 쓸 start, end 값
        self.Fs = 50000
        self.link_count = 6
        self.link_pitch = 406.41/3
        self.start = 0 
        self.end = 0
        self.filtered = []

        # 초당 스텝 1개가 움직인 거리 (링크가 한개이므로 3으로 나눠주면 됨)
        self.chain_velocity = 421.2/3

        # 첫 피크 인덱스 번호 설정
        self.peak_num = 12

    # 데이터 불러오기
    def load_data(self):
        with open(self.file_full_path, 'r') as file_id:
            self.happy_go_n = np.loadtxt(file_id, delimiter=',', dtype='float32')
            self.data = self.happy_go_n[:]
            
    # 음성 데이터 불러오기
    def load_wav_data(self):
        y, sr = librosa.load(wav, sr=self.Fs)
        self.data = y
    
    # 그래프 0점 이동
    def move_graph(self):
        self.aver = np.mean(self.data)
        self.iot7_BP = self.data - self.aver

    # 샘플링 주파수에 따른 X축을 시간축으로 변경
    def X_trans_time(self):
        t = 1/self.Fs
        row = len(self.iot7_BP)
        t1 = t*row
        self.T = np.arange(0, t1, t)
    
    # 피크 값 찾기
    def find_peaks(self):
        result = signal.find_peaks(self.iot7_BP, distance=(self.link_pitch/self.chain_velocity)*self.Fs)#0.7*self.Fs
        # 원본 X축, 시간축, 피크 값을 저장
        self.row_X = result[0]
        self.locs, self.k = [], []
        for i in range(len(result[0])):
            self.locs.append(self.T[result[0][i]])
            self.k.append(self.iot7_BP[result[0][i]])
    
    # 피크 간격 구하기
    def peak_interval(self):
        r_link_count = self.peak_num+self.link_count
        locs_re = []
        for i in np.arange(self.peak_num-1, r_link_count, 1) :
            locs_re.insert(i - (self.peak_num-1), self.locs[i])
        locs_re = np.array(locs_re)
        self.peakInterval = np.diff(locs_re)

    #밴드 패스
    @staticmethod
    def bandpass(data, start, end, fs):
        sos = signal.butter(5, [start, end], "bp", fs=fs, output="sos")
        filtered = signal.sosfilt(sos, data)
        return filtered

    #밴드 패스 출력
    def print_bandpass(self):
        sos = signal.butter(5, [self.start, self.end], "bp", fs=self.Fs, output="sos")
        filtered = signal.sosfilt(sos, self.iot7_BP)
        plt.plot(filtered)

    def print_row(self):
        plt.plot(self.iot7_BP)



    # 최소 분산을 구하는 함수
    # interval 값은 기본 6으로 설정, 변경 가능
    # peak_num은 기본 -1로 설정되어 최소 분산을 구한다.
    # peak_num을 설정하면 해당 피크값에서 분산을 구하여 출력한다.
    def cal_minVar(self, locs, interval=6, peak_num=-1):
        # 분산을 저장할 리스트 선언
        variance = []
        # 반복을 통해 시작 위치부터 일정 간격으로 차이 값 계산
        # 임시 리스트를 통해 각 차이 값 계산
        # 차이 값에 대한 분산을 계산하고 분산 리스트에 저장
        for i in range(len(locs) - interval):
            temp = []
            for j in range(i, i+interval-1, 1):
                temp.append(abs(locs[j] - locs[j+1]))
            variance.append(np.var(temp))
        # 피크 값이 설정되지 않으면 최소 분산
        # 피크 값이 설정되면 해당 피크 값에 대한 분산        
        if peak_num == -1:
            min_variance = min(variance)
        else:
            min_variance = variance[peak_num]
        peak_num = variance.index(min_variance)

        # 분산과 피크 값의 인덱스를 반환
        return min_variance, peak_num 

    @staticmethod
    def auto_bandpass(self,start, end, fs, interval = 6):
        self.Fs = fs
        outer_iterval = 10
        variances = []
        variances2 = []
        row_data = self.iot7_BP
        row_data2 = self.data
        for i in range(0, (end-start)//2, outer_iterval):
            filter_data = self.bandpass(row_data, start + i, end - i, self.Fs)
            self.data = filter_data
            self.move_graph()
            self.find_peaks()
            min_variance, peak_num = self.cal_minVar(self.locs, interval)
            variances.append(min_variance)

        for i in range(0,(end-start), outer_iterval):
            filter_data = self.bandpass(row_data, start + i, start + i + outer_iterval, self.Fs)
            self.data = filter_data
            self.move_graph()
            self.find_peaks()
            min_variance, peak_num = self.cal_minVar(self.locs, interval)
            variances2.append(min_variance)

        self.data = row_data2
        self.iot7_BP = row_data

        if min(variances) > min(variances2):
            self.start =  start+((variances.index(min(variances))*interval))
            self.end =  end-((variances.index(min(variances))*interval))
        else :
            self.start =  start+((variances2.index(min(variances2))*interval)) 
            self.end = end-((variances2.index(min(variances2))*interval))

        filtered = self.bandpass(row_data,self.start,self.end,self.Fs)
        self.auto_bandstop(filtered,self.Fs)

    def peak_plot(self):
        peak = signal.find_peaks(self.result,distance=(self.link_pitch/self.chain_velocity)*self.Fs)
        k = []
        #파인드피크 추가하기
        for i in range(len(peak[0])):
            k.append(self.result[peak[0][i]])
            
        # 정지 상태의 피크 구간 계산
        count = 0
        index = []
        temp_list = copy.deepcopy(peak[0])
        temp_k = copy.deepcopy(k)
        for i in range(len(k)):
            if k[i] < 50 :
                count += 1
            else :
                if count >= 3:
                    index.append(list(map(int, range(i-count, i+1 ,1))))
                count = 0

        # 정지 상태의 피크 구간 제거
        for stop_peak in index:
            temp_list[stop_peak[0]:stop_peak[-1]] = [0]
            temp_k[stop_peak[0]:stop_peak[-1]] = [-1]
            
        # 정지 상태가 아닌 피크의 X축과 값을 저장
        noStop_locs = list(filter(lambda x:x !=0 , temp_list))
        noStop_k = list(filter(lambda x:x !=-1 , temp_k))

        # 그래프 출력
        plt.plot(self.result)
        plt.plot(noStop_locs, noStop_k,'x')
        return noStop_locs, noStop_k

    def bandstop(self,data,start,end,fs):
        sos = signal.butter(5, [start, end], "bs", fs=fs, output="sos")
        filtered = signal.sosfilt(sos, data)
        return filtered
    
    #자동으로 밴드스탑하기 
    #밴드스탑 코드를 넣으면 피크 출력해서 그래프로 찍는 것까지 한번에 자동화 된다.

    def auto_bandstop(self,data,fs):
        interval = 10
        f, P = signal.periodogram(data, fs, nfft=2**12)
        max_p = sorted(P, reverse=True)
        # 5번째까지 첨도가 높은 값을 구하고 해당 주파수 대역을 구하기 위해 하는 연산 
        max = 0
        max_index = -1
        for i in range(len(max_p[0:5])-1):
            a = abs(max_p[0:5][i]-max_p[0:5][i+1])
            if a > max :
                max = a
                max_index = i

        if max_p[0:5][max_index] < max_p[0:5][max_index+1]:
            max = max_p[0:5][max_index+1]

        else :
            max = max_p[0:5][max_index]

        x_index = np.where(P.tolist() == max)
        
        # 얻은 X범위를 +- 10으로 밴드스탑필터를 사용해준다.
        self.result = self.bandstop(data,f[x_index[0][0]]-interval,f[x_index[0][0]]+interval,fs)
        self.peak_plot()

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
        # 신율 한도 계산
        elongation= 0
        for i in range(0, np.prod(self.Link_Length111.shape)):
            elongation=elongation+self.Link_Length111[i]

        link_pitch=self.link_pitch*self.link_count
        # 신율 = ((늘어난 길이(mm))/설계치 길이)*100
        self.elongation_result=((elongation-link_pitch)/link_pitch)*100
    
    # 파일 저장
    def save_file(self):
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
        if (self.elongation_result < 0) | (self.elongation_result > 1.5):
            fid.write('0')
        else:
            fid.write(str(self.elongation_result))
        fid.close()
    

if __name__ == "__main__":
    chain_main_class().main()

