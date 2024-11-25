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

from sklearn.svm import LinearSVC
from sklearn import metrics

# CNN으로 특징 추출
train_features = model.predict(train_images, batch_size=128).reshape(len(train_images), -1)
test_features = model.predict(test_images, batch_size=128).reshape(len(test_images), -1)

# SVM 분류기 학습
svm = LinearSVC()
svm.fit(train_features, y_train.ravel())

# SVM 테스트
y_pred = svm.predict(test_features)
test_accuracy = metrics.accuracy_score(y_test.ravel(), y_pred)
print(f"SVM Test Accuracy: {test_accuracy:.3f}")


# import matplotlib.pyplot as plt
#
# plt.plot(hist.history['accuracy'], 'b-')
# plt.plot(hist.history['val_accuracy'], 'r--')
# plt.show()
#
#
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
#
# # 테스트 데이터에 대해 예측 수행
# predictions = model.predict(test_images)
# predicted_labels = np.argmax(predictions, axis=1)
# true_labels = np.argmax(test_labels, axis=1)
#
# # 혼동 행렬 계산
# conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
# # 혼동 행렬 시각화
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix for MNIST Classification')
# plt.show()