# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
#
# class_names = [
#     "Airplane", "Automobile", "Bird", "Cat", "Deer",
#     "Dog", "Frog", "Horse", "Ship", "Truck"
# ]
#
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
#
# train_images, test_images = X_train / 255, X_test / 255
#
# # 레이블을 one-hot encoding으로 변환
# train_labels = keras.utils.to_categorical(y_train, 10)
# test_labels = keras.utils.to_categorical(y_test, 10)
#
# model = keras.models.Sequential( [
#     keras.layers.Conv2D(input_shape = (32, 32, 3),
#                         kernel_size = (3,3), padding = 'same',
#                         filters = 32),
#     keras.layers.MaxPooling2D((2, 2), strides=2),
#     keras.layers.Conv2D(kernel_size = (3,3), padding ='same',
#                         filters = 64),
#     keras.layers.MaxPooling2D((2, 2), strides=2),
#     keras.layers.Conv2D(kernel_size = (3,3), padding = 'same',
#                         filters = 32),
#     keras.layers.Flatten()
# ])
# model.summary()
#
# # CNN 특성 추출
# train_features = model.predict(train_images)
# test_features = model.predict(test_images)
#
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import metrics
#
# # KNN 분류기 정의
# k = 15
# knn = KNeighborsClassifier(n_neighbors=k)
#
# # KNN 학습
# hist = knn.fit(train_features, y_train.ravel())
#
# # 테스트 데이터로 예측 수행
# predicted_labels = knn.predict(test_features)
#
# # 정확도 계산
# test_acc = metrics.accuracy_score(y_test, predicted_labels)
# print(f"KNN 테스트 정확도 (k={k}): {test_acc:.5f}")
#
#
# # # 정확도와 손실 그래프
# import matplotlib.pyplot as plt
# #
# # plt.plot(hist.history['loss'], 'b-', label='loss value')
# # plt.legend()
# # plt.plot(hist.history['accuracy'], 'r-', label='accuracy')
# # plt.legend()
# # plt.show()
#
#
# # 혼동 행렬
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
#
# conf_matrix = confusion_matrix(y_test.ravel(), predicted_labels)
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
#             xticklabels=class_names, yticklabels=class_names)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title(f'Confusion Matrix for KNN (k={k})')
# plt.show()
#
#
#



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
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 모델 학습
hist = model.fit(train_images, train_labels, epochs=10, batch_size=128, verbose=1)

# CNN 특성 추출
train_features = model.predict(train_images)
test_features = model.predict(test_images)


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# KNN 분류기 정의
k = 15
knn = KNeighborsClassifier(n_neighbors=k)

# KNN 학습
hist_knn = knn.fit(train_features, y_train.flatten())

# 테스트 데이터로 예측 수행
# predicted_labels = knn.predict(test_features)
predicted_labels = knn.predict(test_features)

# 정확도 계산
test_acc = metrics.accuracy_score(y_test, predicted_labels)
print(f"KNN 테스트 정확도 (k={k}): {test_acc}")


# 정확도와 손실 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'], 'b-', label='loss value')
plt.legend()
plt.plot(hist.history['accuracy'], 'r-', label='accuracy')
plt.legend()
plt.show()


# 혼동 행렬
from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix for KNN (k={k})')
plt.show()