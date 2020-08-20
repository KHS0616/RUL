import json
from collections import OrderedDict
import paho.mqtt.client as mqtt

# 파이썬 base64라이브러리 인코딩시 사용
import base64

# json 데이터 생성
clt_data = OrderedDict()
clt_data['Type'] = "Step"
clt_data['Name'] = "계양역 1호기"
clt_data['Content'] = "Bandpass"
clt_data['Time'] = ""
clt_data['Comment'] = "계양역 1호기 Bandpass 결과 수신"

with open("./Data/Image/StepBandpass.png", "rb") as imageFile:
    strd = base64.b64encode(imageFile.read())
    clt_data['Data'] = strd.decode("utf-8")

# json을 스트링으로 바꾼다.
jsonString = json.dumps(clt_data)

# MQTT 접속
broker_address="192.168.0.10"
client = mqtt.Client("Gyeyang001")
client.connect(broker_address)

# json 데이터 전송
client.publish("Mobius/Gyeyang", jsonString.encode())