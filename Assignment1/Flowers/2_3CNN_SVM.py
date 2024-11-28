import numpy as np
import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = (128,128)

def load_train_data(folder_path):
    X = []
    y = []
    class_names = os.listdir(folder_path)
    print(class_names)

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(folder_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = load_img(image_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            X.append(image)
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names


# Load image data
def load_test_data(folder_path):
    X = []
    filenames = []
    for image_name in os.listdir(folder_path):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(folder_path, image_name)
            image = load_img(image_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            X.append(image)
            filenames.append(image_name)
    X = np.array(X)
    return X, filenames

# # Load training and testing data
train_folder = './train'
test_folder = './test'
X_train, y_train, class_names = load_train_data(train_folder)
X_test, test_filenames = load_test_data(test_folder)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
train_images, test_images = X_train / 255.0, X_test / 255.0

# 레이블을 one-hot encoding으로 변환
train_labels = keras.utils.to_categorical(y_train, 5)
test_labels = keras.utils.to_categorical(y_test, 5)

model = keras.Sequential([
    keras.layers.Conv2D(input_shape = (128, 128, 3),
                        kernel_size = (3,3), padding = 'same',
                        filters = 32),
    keras.layers.MaxPooling2D((2, 2), strides=2),
    keras.layers.Conv2D(kernel_size = (3,3), padding ='same',
                        filters = 64),
    keras.layers.MaxPooling2D((2, 2), strides=2),
    keras.layers.Conv2D(kernel_size = (3,3), padding = 'same',
                        filters = 32),
    keras.layers.Flatten(),
    keras.layers.Dense(5, activation='softmax')
])
model.summary()   # 모델의 구조를 요약하여 살펴보자

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# 모델 학습
hist = model.fit(train_images, train_labels, epochs=10, batch_size=128, verbose=1)

# CNN 특성 추출
train_features = model.predict(train_images)
test_features = model.predict(test_images)


from sklearn.svm import LinearSVC
from sklearn import metrics

# SVM 분류기 학습
svm = LinearSVC()
svm.fit(train_features, y_train.flatten())

# 테스트 데이터로 예측 수행
predicted_labels = svm.predict(test_features)

# 정확도 계산
test_acc = metrics.accuracy_score(y_test, predicted_labels)
print(f"SVM 테스트 정확도: {test_acc}")


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
plt.title('Confusion Matrix for SVM')
plt.show()