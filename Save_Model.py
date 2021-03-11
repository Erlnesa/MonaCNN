import configparser
import os

import numpy as np

# 隐藏tensorflow的输出信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()
import pathlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
'''

# 导入配置文件
ini_config = configparser.ConfigParser()
ini_path = 'MonaCNN_config.ini'
ini_config.read(ini_path, encoding='UTF-8')

train_data_path = ini_config.get('train_data', 'path')
train_data_fname = ini_config.get('train_data', 'fname')

# 导入训练数据集
data_dir = tf.keras.utils.get_file(origin=train_data_path, fname=train_data_fname)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print("数据集导入完毕，有效图片数量：" + str(image_count))
# 格式化训练数据
batch_size = int(ini_config.get('model', 'batch_size'))
img_size = int(ini_config.get('model', 'input_size'))
k_Dropout = float(ini_config.get('model', 'k_Dropout'))
train_data_seed = int(ini_config.get('train_data', 'seed'))
train_data_validation_split = float(ini_config.get('train_data', 'validation_split'))
# 拆分出训练数据集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=train_data_validation_split,
    subset="training",
    seed=train_data_seed,
    image_size=(img_size, img_size),
    batch_size=batch_size)
# 拆分出验证数据集
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=train_data_validation_split,
    subset="validation",
    seed=train_data_seed,
    image_size=(img_size, img_size),
    batch_size=batch_size)
class_names = train_ds.class_names
# 输出类名称
print("在训练集中找到了" + str(len(class_names)) + "个分类")
print(class_names)

'''
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
'''
# 自动设置合适的训练缓存
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# 数据标准化
normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# 校验类数量
num_classes = int(ini_config.get('model', 'output_class'))
if num_classes != len(class_names):
    raise Exception("网络的输出类数量与训练集的类数量不匹配！")
    exit()

# 数据扩充
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_size,
                                                              img_size,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
# 最后一层池化后，网络输出为一维向量
# 创建模型
# 该模型由三个卷积块组成，每个卷积块中都有一个最大池层。有一个完全连接的层，上面有128个单元。
# 可以通过relu激活功能激活。尚未针对高精度调整此模型，本教程的目的是展示一种标准方法。
model = Sequential(
    name="MonaCNN",
    layers=[
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(3, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(512, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(1024, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(2048, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes)
    ])
# model = tf.keras.models.load_model(ini_config.get('model', 'save_path'))
# 对于本教程，选择optimizers.Adam优化器和losses.SparseCategoricalCrossentropy(from_logits=False)损失函数。
# 要查看每个训练时期的训练和验证准确性，请传递metrics参数。
model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )
model.build(input_shape=(None, img_size, img_size, 3))
# 查看网络的所有层
model.summary()
# 读取迭代次数
epochs = int(ini_config.get('model', 'epochs'))
# 开始训练
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
print("训练完毕，正在准备绘制训练结果")

# 可视化训练结果
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# 设置pyplot字体，显示中文
plt.rcParams['font.family'] = 'SimHei'
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
# 保存可视化结果
plt.savefig('Model_Performance.jpg')
plt.show()

# 保存模型
model.save(ini_config.get('model', 'save_path'))
# 更新模型版本号
model_version = ini_config.get(section='model', option='version')
index_version = len(model_version.split('.'))
new_model_version = ''
for version_plot in model_version.split('.'):
    if index_version == 1:
        new_model_version = new_model_version + str(int(version_plot) + 1)
    else:
        new_model_version = new_model_version + version_plot + '.'
    index_version = index_version - 1
ini_config.set(section='model', option='version', value=new_model_version)
ini_config.write(open(ini_path, "w"))
print("模型保存完毕，新模型版本为：" + new_model_version)
print("释放内存")
