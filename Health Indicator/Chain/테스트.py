# 그래프 찍기
import matplotlib.pyplot as plt
# chain_main3 = 구동체인, step_chain = 스텝체인
from RS_chain import chain_main_class
a = chain_main_class()
# 파이썬 이미지 라이브러리
from PIL import Image
# 파이썬 base64라이브러리 인코딩시 사용
import base64
# 파이썬 입출력
import io
import numpy as np
# json을 위한 임포트
import json
from collections import OrderedDict

import paho.mqtt.client as mqtt

plt.ioff()

#작업 실행
a.load_data()
a.move_graph()
a.X_trans_time()
a.find_peaks()
#a.auto_bandpass(a,10000,14000,50000)
k = a.bandpass(a.iot7_BP,10000,14000,50000)
#a.auto_bandstop(a.filtered,50000)
a.auto_bandstop(k,50000)
a.peak_interval()
a.math()
a.sin_length()
a.peak_plot()
a.print_row()
a.print_bandpass()
a.send_RS()