from sklearn.linear_model import LogisticRegression

# 제공된 데이터
X = [[168, 0],[166, 0],[173, 0],[165, 0],[177, 0], [163, 0], [178, 0], [172, 0], [163, 1], [162, 1], [171, 1], [162, 1], [164, 1], [162, 1], [158, 1], [173,1]] # 사람 키
y = [65, 61, 68, 63, 68, 61, 76, 67, 55, 51, 59, 53, 61, 56, 44, 57] # 몸무게

y_binary = [1 if weight >= 60 else 0 for weight in y] #60이상이면 1, 아니면 0 = [1,1,1,1,1,1,1,1,0,0,0]

# 로지스틱 회귀
logistic_model = LogisticRegression()
logistic_model.fit(X, y_binary)

# 계수와 절편, 점수
print('계수:', logistic_model.coef_)
print('절편:', logistic_model.intercept_)
print('점수:', logistic_model.score(X,y_binary))

testX = [[167,0], [167,1]]
# 예측 확률
# y_pred = 'predict_proba'
y_pred = logistic_model.predict_proba(testX)
print('예측 확률:', y_pred)

# 예측 결과
# y_pred_logistic = 'predict'
y_pred_logistic = logistic_model.predict_proba(testX)
print('예측 결과:', y_pred_logistic)
