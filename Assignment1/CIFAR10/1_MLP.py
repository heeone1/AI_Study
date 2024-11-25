import tensorflow as tf
from tensorflow import keras
import numpy as np

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_images, test_images = X_train / 255, X_test / 255

# 레이블을 one-hot encoding으로 변환
train_labels = keras.utils.to_categorical(y_train, 10)
test_labels = keras.utils.to_categorical(y_test, 10)

# MLP
model = keras.Sequential([
   keras.layers.Flatten(input_shape=(32, 32, 3)),
   keras.layers.Dense(1024, activation='relu'),
   keras.layers.Dense(512, activation='relu'),
   keras.layers.Dense(512, activation='relu'),
   keras.layers.Dense(10, activation='softmax')
])
model.summary()   # 모델의 구조를 요약하여 살펴보자

model.compile(optimizer='adamw',\
             loss='categorical_crossentropy',
             metrics=['accuracy'])

hist = model.fit(train_images, train_labels, epochs=10, verbose=1)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
print('\n테스트 정확도:', test_acc)


import matplotlib.pyplot as plt
# 정확도와 손실 그래프
plt.plot(hist.history['loss'], 'b-', label='loss value')
plt.legend()
plt.plot(hist.history['accuracy'], 'r-', label='accuracy')
plt.legend()
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

# 테스트 데이터에 대해 예측 수행
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# 혼동 행렬 계산
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for MNIST Classification')
plt.show()



