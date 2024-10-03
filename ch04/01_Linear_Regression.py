import numpy as np
from sklearn.linear_model import LinearRegression

linear_model= LinearRegression()

##기본 예제
# X = [[163],[179],[166],[169],[171]] # 사람 키
# y = [54, 63, 57, 56, 58] # 몸무게
#
# coef = linear_model.coef_
# intercept = linear_model.intercept_
# score=linear_model.score(X, y)
#
# print ("y = {}*X + {:.2f}".format(coef.round(2), intercept))
# print ("데이터와 선형 회귀 직선의 관계점수 :  {:.1%}".format(score))

##실습1 - 남자 = 0, 여자 = 1
X = [[168, 0],[166, 0],[173, 0],[165, 0],[177, 0], [163, 0], [178, 0], [172, 0], [163, 1], [162, 1], [171, 1], [162, 1], [164, 1], [162, 1], [158, 1], [173,1]] # 사람 키
y = [65, 61, 68, 63, 68, 61, 76, 67, 55, 51, 59, 53, 61, 56, 44, 57] # 몸무게
linear_model.fit(X, y)

coef = linear_model.coef_
intercept = linear_model.intercept_
score=linear_model.score(X, y)

print ("y = {}*X + {:.2f}".format(coef.round(2), intercept))
print ("데이터와 선형 회귀 직선의 관계점수 :  {:.1%}".format(score))

# import matplotlib.pyplot as plt
#
# plt.scatter(X, y, color='blue', marker='D')
# y_pred = linear_model.predict(X)
# plt.plot(X, y_pred, 'r:')
# plt.show()

#unseen = [[167]] #키 167일 때 몸무게 몇인지 예측하는 거임
unseen = [[167, 0], [167,1]] #실습1
result = linear_model.predict(unseen)
print ("키 {}cm는 몸무게 {}kg으로 추정됨".format(unseen, result.round(1)))

