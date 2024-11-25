import tensorflow as tf
from tensorflow import keras
import numpy as np

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 레이블을 one-hot encoding으로 변환
train_labels = keras.utils.to_categorical(y_train, 10)
test_labels = keras.utils.to_categorical(y_test, 10)

train_images = X_train[:, :, :]
test_images = X_test[:, :, :]
train_images, test_images = train_images / 255, test_images / 255

model = keras.models.Sequential( [
    keras.layers.Conv2D(input_shape = (32, 32, 3),
                        kernel_size = (3,3), padding = 'same',
                        filters = 32),
    keras.layers.MaxPooling2D((2, 2), strides=2),
    keras.layers.Conv2D(kernel_size = (3,3), padding ='same',
                        filters = 64),
    keras.layers.MaxPooling2D((2, 2), strides=2),
    keras.layers.Conv2D(kernel_size = (3,3), padding = 'same',
                        filters = 32),
    keras.layers.Flatten()
])

# CNN 특성 추출
train_features = model.predict(train_images)
test_features = model.predict(test_images)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# KNN 분류기 정의
k = 20
knn = KNeighborsClassifier(n_neighbors=k)

# KNN 학습
knn.fit(train_features, y_train.ravel())

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 테스트 데이터로 예측 수행
predicted_labels = knn.predict(test_features)

# 정확도 계산
test_acc = metrics.accuracy_score(y_test, predicted_labels)
print(f"KNN 테스트 정확도 (k={k}): {test_acc:.3f}")

import matplotlib.pyplot as plt

plt.plot(predicted_labels.history['accuracy'], 'b-')
plt.plot(predicted_labels.history['val_accuracy'], 'r--')
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



