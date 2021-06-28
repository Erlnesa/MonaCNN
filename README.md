# 一、MonaCNN目前的运行表现

从图库网站[danbooru](https://danbooru.donmai.us/)以“genshin_impact”作为关键词批量下载了5342枚图片，其中部分内容如下图。

![分类前](https://github.com/Erlnesa/MonaCNN/blob/main/images/分类前.jpg)

将这5342枚图片以人工识别的方式预先进行分类，以此作为分类前的测试数据集使用，以此检验该系统在一般使用环境下的使用情况和分类准确度。

经过对于测试使用的5342枚测试图片，有如下测试结果：

|          图片总数          |  5342  |
| :------------------------: | :----: |
|    含有“莫娜”的图片数量    |  825   |
|  模型分类出的“莫娜”图片数  |  805   |
|  未被识别出的“莫娜”图片数  |  141   |
| 被错误识别为“莫娜”的图片数 |  121   |
|     总体分类错误的数量     |  262   |
|      “莫娜”单类准确度      | 82.91% |

其中，单张图片预测分类消耗所的时间为：0.06~0.27秒，平均值为0.075秒，众数为0.06秒。运行结果如“图 9 分类后的测试集”所示。

![分类后](https://github.com/Erlnesa/MonaCNN/blob/main/images/分类后.jpg)

# 二、未来的改进和研究方向

## 尝试增大batch size大小

从训练损失曲线来看，在训练过程中验证集曲线震荡明显，很有可能是过小的batch size造成了每次训练确定的梯度下降方向不准确。因此，如果能增加batch size的值可能会为模型带来更高的分类准确度。同时因为一次读入的样本增多了，所以在一定程度上也可能会加快模型的训练速度。

但过大的batch size值也可能会将模型收敛到一些不好的局部最优点。也正是因为这一点，此时要达到同样的分类准确度所需要的训练时间会明显增加。

## 尝试缩小输入图片分辨率

过大的分辨率会对内存和图形处理器内存造成明显负担，而这就直接使得batch size的值被限制在了一个非常小的值以下。从而间接影响到了模型的分类准确度。

但这并表示减小输入分辨率就能使模型的分类准确度变高。更小的分辨率意味着在预处理层对图片resize时，图片会丢失更多画面内容信息，从而对模型的分类准确度产生不利的影响。因此，如何平衡batch size和输入分辨率是一个有意义的后续研究方向。

## 尝试增加迭代次数

从“图6 训练损失曲线”来看，在完成20次迭代训练时，模型的训练损失处于明显的下降趋势中。因此，此时的模型处于欠拟合状态，需要进一步增加迭代次数来增加模型的分类准确度。

但迭代次数也不能无限制的增加。过大的迭代次数可能会使模型陷入过拟合状态，即对于训练数据集来说有相当好的分类准确度，但进行实际预测时准确度会明显下降。因此平衡出合适的迭代次数也是一个有意义的后续研究方向。

## 为卷积层添加适当的Padding

目前MonaCNN中所有的卷积层皆未使用Padding，这导致图像的边缘像素并没有像图片的中心像素一样充分参与到卷积运算中。因此选择一个恰当的Padding方法来解决该问题也是一个有意义的后续研究方向。

# 三、简介

## 1．研究意义

本模型的研究意义旨在提供一个在Tag类别不完整的图库网站中，快速、自动化的找出画面里含有特定动漫角色（本文以游戏《原神》中出现的女性角色“莫娜”为例）的画作。

此处造成Tag类别不完整的原因有很多种，例如：绘画师们的惯用语言不同（人物可能会被打上任何一种语言所描述的Tag）、有些绘画师不会给自己发布的图片打上任何Tag、有些绘画师只会给图片打上游戏名称的Tag而没有打上人物Tag。

对于该类情况，以手动搜索Tag的方式来查找画面中含有“莫娜”的图片便有些困难。由此问题出发产生了本文的研究内容。

## 2．功能概述

将批量保存在本地硬盘的图片送入神经网络进行画面内容检测，根据神经网络的输出结果，从批量保存在本地硬盘的图片中挑出画面里含有动漫角色“莫娜”的部分图片加以保存，同时删除画面里没有“莫娜”的其他图片。

## 3．基于迁移学习的神经网络

对于本系统默认使用的神经网络模型MonaCNN，为了达到较好的识别准确度，模型的训练过程分为预训练和迁移学习两大部分。

首先将训练集调整为游戏“原神”中的男性角色和女性角色两类，经过首次训练得到对于性别特征较为敏感的神经网络模型。将本次训练得到的模型作为预训练模型。

在预训练模型的基础上，将训练集调整为服装、体型、发色与“莫娜”相似和不相似的两类，再迁移学习得到对拥有与“莫娜”相似身体特征的人物敏感的模型。

最后将训练数据集调整为“莫娜”与“其他角色”两类，再经过一轮迁移学习得到能较好识别出角色“莫娜”的神经网络模型。同时将其命名为MonaCNN。

在经历一次预训练和两次迁移学习后，MonaCNN对识别目标“莫娜”拥有95.1%的总体分类准确度。

作为对比的是初代MonaCNN模型，初代的MonaCNN训练中尚未采用迁移学习，因此初代MonaCNN是由“莫娜”与“其他角色”这两类数据集作为正负样本，经过一次训练产生的。其对识别目标“莫娜”仅拥有83.51%的总体分类准确度。

# 四、系统原理及功能

## 1．系统功能

将指定文件夹下的所有图片依次读取，并根据MonaCNN的输出结果判断图片应有的分类。如果MonaCNN对该图片的画面里含有“莫娜”有95%以上置信度，则将图片移动到特定位置保存，否则删除。

也可重新选择训练样本数据集，然后重新训练MonaCNN来完成对不同动漫角色或是其他内容的分类。

## 2．主程序原理

对于主程序“Load_Model.py”，其工作流程如下。

![主程序流程图](https://github.com/Erlnesa/MonaCNN/blob/main/images/主程序流程图.png)

## 3．数据集处理程序原理

对于数据扩充程序“Data_Augmentation.py”，其工作流程如下。

![数据扩充流程图](https://github.com/Erlnesa/MonaCNN/blob/main/images/数据扩充流程图.png)

对于数据集格式调整程序“Png_to_Jpg.py”，其工作流程如下。

![数据集预处理流程图](https://github.com/Erlnesa/MonaCNN/blob/main/images/数据集预处理流程图.png)

## 4．神经网络生成程序原理

对于神经网络生成程序“Save_Model.py”，其工作流程如下。

![模型训练流程图](https://github.com/Erlnesa/MonaCNN/blob/main/images/模型训练流程图.png)

## 5．神经网络工作流程及模型结构

首先按照“图 1 数据扩充流程图”中所述流程对训练样本进行数据扩充，保证正负样本数量和不小于8000张图片，且正负样本数量之差不大于100张图片。

经过数据扩充后的样本会按照2:8的比例划分为训练数据集和验证数据集，之后按照“图 4 MonaCNN训练流程图”中所述流程开始模型训练。

对于训练完毕的MonaCNN，其网络层级结构如“图 5 MonaCNN的网络层级结构”所示。

![model](https://github.com/Erlnesa/MonaCNN/blob/main/images/model.png)

其中预处理层会将图片格式化到 512 × 512 的大小，同时将图片的 *RGB* 值从 [0, 255]映射到[0, 1]区间内。

预处理层后连接的是七组卷积池化层。其神经元数量分别为3、64、128、256、512、1024、2048个；卷积核大小均为 3 × 3；激活函数均为 *ReLU*。经过此过程图片由 512 × 512 × 3 被卷积到 2 × 2 × 2048 。

之后连接两个附带有Dropout的全连接层，其神经元数量分别为2048和1024个，Dropout参数均0.3；激活函数均为 *ReLU* 。

最后连接输出层，输出层拥有两个神经元，它将输出图片分别属于两类的概率数值 *Pa* 与 *Pb* 。将 *Pa* 带入下述公式来确定输入图片应有的分类状况。

![](https://github.com/Erlnesa/MonaCNN/blob/main/images/gs1.png)

其中*P*为用户自定义的最低置信度，默认为95%

当 *f( x ) = 0* 时认为图片属于"画面里含有‘莫娜’"的一类；当 *f( x ) = 1* 时认为图片属于“画面里没有‘莫娜’”的一类。

## 6．神经网络的迁移学习

在初期测试中发现MonaCNN的预测能力并不理想，模型过于专注图片中角色的服装颜色和头发颜色。因此对于MonaCNN本文选择了迁移学习的训练方法。先选择服装和头发颜色与“莫娜”近似的角色作为一类，其他服装和头发颜色与“莫娜”有明显区别的其他的角色作为另外一类。

预训练完毕后再由正常的样本集（画面里含有“莫娜”的图片；画面里没有“莫娜”的图片）训练得到最终不会过于关注角色服装和头发颜色的模型。训练损失曲线如“图 6 训练损失曲线”所示。

![训练损失曲线](https://github.com/Erlnesa/MonaCNN/blob/main/images/Model_Performance.jpg)

# 五、使用说明书

## 1．运行环境

|    操作系统    |              Windows 10 64位操作系统               |
| :------------: | :------------------------------------------------: |
|     处理器     |   Intel（R）Core（TM）i7-7700HQ及更高性能处理器    |
|      主频      |                   2.80GHz及以上                    |
|      内存      |                     16GB及以上                     |
|   图形处理器   | NVIDIA GeForce GTX 1060 MaxQ及更高性能的图形处理器 |
| 图形处理器内存 |                     8GB及以上                      |
|  剩余磁盘空间  |    至少10GB的C盘剩余空间；至少5GB的D盘剩余空间     |

其中处理器和图形处理器需支持Tensorflow-gpu、CUDA、cuDNN。

## 2．安装方法

1）前往Python官方网站（https://www.python.org/downloads/release/python-382/）下载并安装Python3.8到C盘根目录，再将“Python38”文件夹覆盖到C盘根目录下。

2）重新启动计算机。

3）前往NVDIA官方网站下载并安装最新的显卡驱动程序。

4）重新启动计算机。

5）复制“NVIDIA GPU Computing Toolkit”文件夹到“C:\Program Files”文件目录下。并在环境变量中的系统环境变量中添加如下几个环境变量：

| 变量名                 | 变量值                                                       |
| ---------------------- | ------------------------------------------------------------ |
| CUDA_PATH              | C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1     |
| CUDA_PATH_V11_1        | C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1     |
| NVCUDASAMPLES_ROOT     | C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.1         |
| NVCUDASAMPLES11_1_ROOT | C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.1         |
| NVTOOLSEXT_PATH        | C:\Program Files\NVIDIA Corporation\NvToolsExt\              |
| “Path”中“新建”一行     | C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin |
| “Path”中“新建”一行     | C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\libnvvp |
| “Path”中“新建”一行     | C:\Program Files\NVIDIA Corporation\Nsight Compute 2020.2.0\ |
| “Path”中“新建”一行     | C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common       |

6）在`D:\Python_Project\MonaCNN`文件夹下，下载本仓库代码。使所有的.py文件都被包含于内，注意此时MonaCNN文件夹下应该有saved_model文件夹。

7）进入saved_model文件夹，解压缩。

```
务必保证saved_model文件夹在项目根目录下，即和所有.py文件同目录。
同时解压完毕后，mona文件夹需要在saved_model文件夹下，且mona文件夹下包含文件saved_model.pb。
如果不是请尝试调整，可能的原因是：在解压时生成了嵌套的mona文件夹。
```

8）配置数据集文件夹。

你应该看到一个叫datasets的文件夹，它在`C:/Users/你的用户名/.keras/`路径下。如果没有就创建该文件夹。

在datasets文件夹下创建一个`MoeLoader +1s`文件夹，内部放入所有待分类图片。

将`MoeLoader +1s`文件夹压缩为MoeLoader +1s.zip

9）重新启动计算机。

## 3．使用方法

### 图片分类

1）打开`MonaCNN_config.ini`。将其中“你的用户名”字样修改为你正在使用的用户名。

2）运行`Load_Model.py`开始图片分类。

注：分类完毕的图片会被储存到ini文件中`[test_data] copyfile_path`设置的路径中。

### 重新训练模型

1）建立训练样本与预训练样本

样本位置在“C:\Users\你的用户名\.keras\datasets\mona”文件夹下，将正样本和负样本图片分别存放在该文件夹下的“mona”文件夹和“other”文件夹下。

2）返回目录“C:/Users/你的用户名/.keras/datasets”。将“mona”文件夹压缩为“mona.zip”

3）打开`MonaCNN_config.ini`。将其中“你的用户名”字样修改为你正在使用的用户名。

4）运行“Save_Model.py”

