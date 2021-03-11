import configparser
import glob
from shutil import copyfile
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import numpy as np
import os

# 隐藏tensorflow的输出信息
from PIL.Image import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow_datasets as tfds
import random
import tensorflow as tf

tfds.disable_progress_bar()
import pathlib
import matplotlib.pyplot as plt
from tensorflow import keras

true_mona_pic_conunt = 0
test_mona_pic_conunt = 0
err_for_other_conunt = 0
err_for_mona_conunt = 0

# 导入配置文件
ini_config = configparser.ConfigParser()
ini_path = 'MonaCNN_config.ini'
ini_config.read(ini_path)
# 读最小置信度
minimum_confidence = float(ini_config.get('test_data', 'minimum_confidence'))
# 读是否复制标志位
if ini_config.get('test_data', 'need_copyfile_flag') == "true":
    need_copyfile_flag = True
else:
    need_copyfile_flag = False
# 读文件复制路径
copyfile_path = ini_config.get('test_data', 'copyfile_path')

# 读取模型
model = tf.keras.models.load_model(ini_config.get('model', 'save_path'))
# 查看网络的所有层
model.summary()

# 限制图片显示尺寸
img_size = int(ini_config.get('model', 'input_size'))

# MoeLoader +1s

data_test_path = ini_config.get('test_data', 'path')
data_test_fname = ini_config.get('test_data', 'fname')

data_test_orig = tf.keras.utils.get_file(origin=data_test_path,
                                         fname=data_test_fname)
data_test = pathlib.Path(data_test_orig)
# 解析
all_test_image_paths = list(data_test.glob('*/*'))
all_test_image_paths = [str(path) for path in all_test_image_paths]
random.shuffle(all_test_image_paths)
# 图片总数
image_test_count = len(all_test_image_paths)
print("Image count: " + str(image_test_count))


# 设置pyplot字体，显示中文
# plt.rcParams['font.family'] = 'SimHei'


def Verification_this_img(test_img_array, is_original_img=True):
    global true_mona_pic_conunt
    try:
        test_img_array = keras.preprocessing.image.img_to_array(img)
        # mirror_img = tf.image.flip_left_right(img)
        test_img_array = tf.expand_dims(test_img_array, 0)  # Create a batch
        test_predictions = model.predict(test_img_array)
        test_score = tf.nn.softmax(test_predictions[0])
        print(
            "这张图片最接近分类 {} ，置信度为 {:.2f} %"
                .format(np.argmax(test_score), 100 * np.max(test_score))
        )

        print(test_paths_index)
        # 以图片文件名读取标签来统计正确率，标签不参与神经网络的判断
        target_img_path = test_paths_index.split('\\')[len(test_paths_index.split('\\')) - 1]
        # 看看图片名字里有没有mona
        if target_img_path.count('mona') >= 1 and is_original_img:
            true_mona_pic_conunt = true_mona_pic_conunt + 1
        if (np.argmax(test_score) == 0) and (np.max(test_score) >= minimum_confidence):
            # 当前人物有足够置信度认为是莫纳
            if need_copyfile_flag:
                print("复制文件")
                copyfile(test_paths_index, copyfile_path + str(np.max(test_score)) + '_' + target_img_path)
            return True
        else:
            return False
    except PIL.UnidentifiedImageError:
        print("错误的图片格式或图片已损坏")


for test_paths_index in all_test_image_paths:
    try:
        target_img_path = test_paths_index.split('\\')[len(test_paths_index.split('\\')) - 1]
        # 按照rgb格式读取，忽略其他图层
        img = keras.preprocessing.image.load_img(
            test_paths_index,
            target_size=(img_size, img_size),
            color_mode="rgb"
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        this_is_mona = Verification_this_img(test_img_array=img_array, is_original_img=True)
        if this_is_mona:
            test_mona_pic_conunt = test_mona_pic_conunt + 1
            # 当前人物有足够置信度认为是莫纳
            if target_img_path.count('mona') == 0:
                # 不是mona但被识别为了mona
                err_for_mona_conunt = err_for_mona_conunt + 1
        else:
            mirror_img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            img_array = keras.preprocessing.image.img_to_array(mirror_img)
            this_is_mona = Verification_this_img(test_img_array=img_array, is_original_img=False)
            if this_is_mona:
                test_mona_pic_conunt = test_mona_pic_conunt + 1
                # 当前人物有足够置信度认为是莫纳
                if target_img_path.count('mona') == 0:
                    # 不是mona但被识别为了mona
                    err_for_mona_conunt = err_for_mona_conunt + 1
            elif target_img_path.count('mona') > 0:
                # 是mona但没有被识别出来
                err_for_other_conunt = err_for_other_conunt + 1

        print()
    except PIL.UnidentifiedImageError:
        print("错误的图片格式或图片已损坏")

print('图片总数：' + str(image_test_count))
print('真实mona类总数：' + str(true_mona_pic_conunt))
print('模型分类出的mona类总数：' + str(test_mona_pic_conunt))
print('未被识别出的mona总数：' + str(err_for_other_conunt))
print('被错误识别为mona总数：' + str(err_for_mona_conunt))
print('分类错误的数量：' + str(err_for_mona_conunt + err_for_other_conunt))
print('准确度：' + str(round(((test_mona_pic_conunt - err_for_mona_conunt) / true_mona_pic_conunt) * 100, 2)))
