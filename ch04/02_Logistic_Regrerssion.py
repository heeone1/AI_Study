from sklearn.linear_model import LogisticRegression

# 제공된 데이터
X = []
y = []

y_binary = [1 if weight >= 60 else 0 for weight in y]

# 로지스틱 회귀
logistic_model = LogisticRegression()
logistic_model.fit(X, y_binary)

# 계수와 절편, 점수
print('계수:', )
print('절편:', )
print('점수:', )

testX = [ ]
# 예측 확률
y_pred = 'predict_proba'
print('예측 확률:', )

# 예측 결과
y_pred_logistic = 'predict'
print('예측 결과:',)
