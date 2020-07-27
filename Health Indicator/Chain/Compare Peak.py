"""
현장 데이터와 실험실 데이터의 피크 값을 비교하는 코드
"""
from step_chain import step_chain
chain_real = step_chain()
chain_lab = step_chain()

# 데이터 경로 설정
chain_real.file_full_path = "C:/Users/PC/Desktop/RUL/C_33_3802-688-9_200421_1415N_D.raw.txt"
chain_lab.file_full_path = "C:/Users/PC/Desktop/RUL/Lab008.txt"

# 신율 계산
chain_real.Start()
chain_lab.Start()

# 그래프 출력 및 비교
chain_lab.show_graph("peak")
chain_real.show_graph("peak")