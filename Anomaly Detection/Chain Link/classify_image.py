# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using TF Lite to classify a given image using an Edge TPU.
   To run this code, you must attach an Edge TPU attached to the host and
   install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
   device setup instructions, see g.co/coral/setup.
   Example usage (use `install_requirements.sh` to get these files):
   ```
   python3 classify_image.py \
     --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
     --labels models/inat_bird_labels.txt \
     --input images/parrot.jpg
   ```
"""
import threading


# 파이썬 이미지 라이브러리
from PIL import Image
# 파이썬 입출력
import io
# 파이썬 base64라이브러리 인코딩시 사용
import base64
# json을 위한 임포트
import json
from collections import OrderedDict
# 그래프 찍기
import matplotlib.pyplot as plt
plt.ioff() 


import argparse
import time
from datetime import datetime
from PIL import Image

import numpy as np

import classify
import tflite_runtime.interpreter as tflite
import platform

import testmic as mic
import testwavtxt as wavetxt
import testwriteimage as writeimage

# chain_main3 = 구동체인, step_chain = 스텝체인
from chain_main3 import chain_main_class
from step_chain import step_chain
# 소켓을 사용하기 위해서는 socket을 import해야 한다.
import socket
# json을 위한 임포트
import json
from collections import OrderedDict

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).
  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

#===============================
def connect_server():
  print("연결 대기 중")

  # 로컬은 127.0.0.1의 ip로 접속한다.
  print("ip 접속 중")
  try:
      #HOST = '192.168.0.10'
      HOST = '10.200.73.11'
      print("ip 접속 성공")
  except:
      print("ip 접속 실패")

  # port는 위 서버에서 설정한 9999로 접속을 한다.
  print("포트 접속 중")
  try:
      PORT = 9999
      print("포트 접속 성공")
  except:
      print("포트 접속 실패")

  # 소켓을 만든다.
  try:
      global client_socket
      client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      print("소켓 생성 성공")
  except:
      print("소켓 생성 실패")

  # connect함수로 접속을 한다.
  try:   
      client_socket.connect((HOST, PORT))# 소켓을 사용하기 위해서는 socket을 import해야 한다.
      print("소켓 연결 성공")
  except:
      print("소켓 연결 실패")
  else:
      print("gg")

# 이상감지 결과를 서버로 전송
def send_data(data):
  clt_data = OrderedDict()
  clt_data['Type'] = 'AD'
  clt_data['Name'] = 'Edge'
  clt_data['Content'] = 'String'
  clt_data['Data'] = data
  clt_data['Time'] = str(datetime.now())
  clt_data['Comment'] = '2'

  # json을 스트링으로 바꾼다.
  jsonString = json.dumps(clt_data)
  # 데이터 전송
  client_socket.sendall(jsonString.encode())

# 실시간 이상감지
def main():
  mic.c_mic()
  wavetxt.wave_to_txt()
  writeimage.write()

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', default="./models/CNN_Model_6_v1_edgetpu.tflite", help='File path of .tflite file.')
  parser.add_argument(
      '-i', '--input', default="./images/test1/0.png", help='Image to be classified.')
  parser.add_argument(
      '-l', '--labels', default="./models/classification.txt", help='File path of labels file.')
  parser.add_argument(
      '-k', '--top_k', type=int, default=1,
      help='Max number of classification results')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')
  parser.add_argument(
      '-c', '--count', type=int, default=1,
      help='Number of times to run inference')
  args = parser.parse_args()

  labels = load_labels(args.labels) if args.labels else {}

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  size = classify.input_size(interpreter)
  image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)
  classify.set_input(interpreter, image)

  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
  for _ in range(args.count):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_output(interpreter, args.top_k, args.threshold)
    print('%.1fms' % (inference_time * 1000))

  print('-------RESULTS--------')
  filename = "./Data/result/anomaly_detection_result.txt"
  with open(filename, "w") as fid:
    for klass in classes:
      print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))
     # np.savetxt(fid, labels.get(klass.id, klass.id))
     # d = ('%s: %s\n'% (str(round(time.time())), labels.get(klass.id, klass.id)))
      d = ('%s' % (labels.get(klass.id, klass.id)))
      fid.write(d)
     # fid.writelines(str(round(time.time()))+"\t"+labels.get(klass.id, klass.id)+"\n")
  send_data(d)

# #신율 계산 작동을 위한 함수
# def RS_cal_sinul():
#     # 구동체인 작동
#     print("구동체인...")
#     global a 
#     a = chain_main_class()
#     a.load_data()
#     a.move_graph()
#     a.X_trans_time()

#     #auto_bandpass에 데이터와 검증 주파수 범위, fs를 입력하면 peak_plot까지 자동으로 수행한다.
#     a.auto_bandpass(a,10000,14000,50000)
#     a.peak_interval()
#     a.math()
#     a.sin_length()
#     print("구동체인 계산 완료...")



# def STEP_cal_sinul():
#   # 스텝체인 작동
#   print("스텝체인...")
#   global chain
#   chain = step_chain()
#   chain.Start()    
#   print("스텝체인 계산 완료...") 

# # RS 명령이 오면 동작 함수
# def RS_Chain():
#   clt_data['Type'] = 'RS'           
#   if data['Content'] == 'Peak':
#       a.peak_plot()
#       with open("rs_peak.png", "rb") as imageFile:
#           rs_peak = base64.b64encode(imageFile.read())
#           clt_data['Data'] = rs_peak.decode("utf-8")
#   elif data['Content'] == 'Raw':
#       a.print_row()
#       with open("rs_raw.png", "rb") as imageFile:
#           rs_raw = base64.b64encode(imageFile.read())
#           clt_data['Data'] = rs_raw.decode("utf-8")
#   elif data['Content'] == 'Bandpass':
#       a.print_bandpass()
#       with open("rs_bandpass.png", "rb") as imageFile:
#           rs_bandpass = base64.b64encode(imageFile.read())
#           clt_data['Data'] = rs_bandpass.decode("utf-8")
#   elif data['Content'] == 'Sin':
#           RS_cal_sinul()
#           a.cal_sin()
#           clt_data['Data'] = a.elongation_result[0]
#   #return clt_data['Data']

# # STEP 명령이 오면 동작 함수
# def Step_Chain():
#     clt_data['Type'] = 'Step'
#     if data["Content"] == "Sin":
#         STEP_cal_sinul()
#         chain.cal_sin()
#         clt_data['Data'] = chain.elongation_result[0]
#     elif data["Content"] == "Raw":
#         chain.show_graph("row")
#         with open("GG.png", "rb") as imageFile:
#             strd = base64.b64encode(imageFile.read())
#             clt_data['Data'] = strd.decode("utf-8")
#     elif data["Content"] == "Bandpass":
#         chain.show_graph("bandpass")
#         with open("GG.png", "rb") as imageFile:
#             strd = base64.b64encode(imageFile.read())
#             clt_data['Data'] = strd.decode("utf-8")
#     elif data["Content"] == "Peak":
#         chain.show_graph("peak")
#         with open("GG.png", "rb") as imageFile:
#             strd = base64.b64encode(imageFile.read())
#             clt_data['Data'] = strd.decode("utf-8")
#     elif data["Content"] == "Autocorrelate":
#         chain.show_graph("autocorrelate") 
#         with open("GG.png", "rb") as imageFile:
#             strd = base64.b64encode(imageFile.read())
#             clt_data['Data'] = strd.decode("utf-8")

# def communication():
#   while 1:
#     print("대기중")
#     global data
#     data = client_socket.recv(1024)
    
#     # 데이터를 수신한다.
#     msg = " "
#     msg = data.decode()

#     # 받은 데이터 확인용
#     print('Received Data : ', msg)

#     # 문자열을 json으로 변환
#     data = json.loads(msg)

#     # 보내기 위한 json 생성
#     global clt_data
#     clt_data = OrderedDict()
#     clt_data['Name'] = data['Name']
#     clt_data['Content'] = data['Content']
#     clt_data['Time'] = data['Time']
#     clt_data['Comment'] = "계양역입니다 1호기"

#     # json의 구동체인 부분을 타입별로 나눠준다.
#     if data["Type"] == "RS":
#       # clt_data['Data'] = RS_Chain(clt_data)
#       RS_Chain()
#       # json을 스트링으로 바꾼다.
#       jsonString = json.dumps(clt_data)
#       # 데이터 전송
#       client_socket.sendall(jsonString.encode())
#     elif data["Type"] == "Step":       
#       Step_Chain()
#       # json을 스트링으로 바꾼다.
#       jsonString = json.dumps(clt_data)
#       # 데이터 전송
#       client_socket.sendall(jsonString.encode())        
    
#     elif data["Type"] == "Stop":
#       break

if __name__ == '__main__':
  # 소켓 연결
  connect_server()
  while 1:
    main()

#   # t2.start()

  # 소켓 연결 해제
  client_socket.close()

