import matplotlib.pyplot as plt
from step_chain import step_chain
import numpy as np

# 기존 스텝 체인 클래스 가져오기
chain = step_chain()
chain.Start()

# librosa를 이용한 자기상관함수 결과 출력
plt.plot(chain.correlate_result)
plt.show()

# 넘파이를 이용한 자기상관함수 메소드 선언
def AutoCorrelation(x):
       x = np.asarray(x)
       y = x-x.mean()
       result = np.correlate(y, y, mode='full')
       result = result[len(result)//2:]
       return result 

# 넘파이를 이용한 자기상관함수 결과 출력
aa = AutoCorrelation(chain.filtered)
plt.plot(aa)

# 값 비교를 위한 출력
print(chain.correlate_result)
print(aa)