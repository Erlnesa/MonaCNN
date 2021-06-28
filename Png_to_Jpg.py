from PIL import Image
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import random
import os

# MoeLoader +1s
folder_name = "mona256"
img_resize = 256

import os
from PIL import Image

# 导入数据
data_root_orig = tf.keras.utils.get_file(origin='C:/Users/Erlnesa/.keras/datasets/' + folder_name + '.zip',
                                         fname=folder_name)
data_root = pathlib.Path(data_root_orig)
# 解析
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
# 图片总数
image_count = len(all_image_paths)

for img_index in range(len(all_image_paths)):
    try:
        print(all_image_paths[img_index].replace(".png", ".jpg"))
        img = Image.open(all_image_paths[img_index])
        if img.mode == "P" or img.mode == "RGBA" or img.mode == "LA":
            img = img.convert('RGB')
        if all_image_paths[img_index].endswith(".png"):
            img = img.resize((img_resize, img_resize), Image.ANTIALIAS)
            img.save(all_image_paths[img_index].replace(".png", ".jpg"))
            # 移除
            os.remove(all_image_paths[img_index])
        elif all_image_paths[img_index].endswith(".jpg"):
            img = img.resize((img_resize, img_resize), Image.ANTIALIAS)
            img.save(all_image_paths[img_index])
        # print(all_image_paths[img_index])
        else:
            os.remove(all_image_paths[img_index])
    except FileNotFoundError:
        print("图片不存在或图片已损坏")

print("Done")
