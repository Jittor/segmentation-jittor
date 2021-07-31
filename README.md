# 计图语义分割模型库

本次计图平台所发布的语义分割模型库，已经支持了目前主流的语义分割算法。其中包含了三种经典的 Backbone ，以及六种常见的分割 Head，具体如下表所示。

| Backbone  | Segmentation head |
| --------- | ----------------- |
| ResNet    | PSPNet            |
| Res2Net   | Deeplab V3+       |
| MobileNet | DANet             |
|           | OCNet             |
|           | ANN               |
|           | OCRNet            |

我们对上面所提到的模型在 PASCAL VOC 数据集上做了完整的训练以及 single scale 的测试，得到的测试结果如下。

| Model   | backbone    | batch size | stride | input size | miou  |
| ------- | ----------- | ---------- | ------ | ---------- | ----- |
| Deeplab | resnet-101  | 8          | 16     | 513 x 513  | 78.38 |
| Deeplab | resnet-101  | 16         | 16     | 513 x 513  | 78.50 |
| Deeplab | mobilenet   | 8          | 16     | 513 x 513  | 70.15 |
| Deeplab | res2net-101 | 8          | 16     | 513 x 513  | 78.89 |
| PSPNet  | resnet-101  | 8          | 16     | 513 x 513  | 77.48 |
| DANet   | resnet-101  | 8          | 16     | 513 x 513  | 76.80 |
| OCNet   | resnet-101  | 8          | 16     | 513 x 513  | 76.77 |
| ANN     | resnet-101  | 8          | 16     | 513 x 513  | 77.50 |
| OCRNet  | resnet-101  | 8          | 16     | 513 x 513  | 78.30 |




本模型库的安装以及使用方法:
首先需要下载整个模型库到本地
```
$ git clone https://github.com/Jittor/segmentation-jittor.git
```

然后需要在 [此处](<https://share.weiyun.com/DQiZVGbp>) 下载 backbone 的预训练模型，并放在  ./pretrained_model 目录下面

通过 settings.py 文件来修改相关路径和模型配置，使用命令 

```
$ sh train.sh
```

即可进行完整的训练


欢迎大家使用Jittor的GAN模型库开展自己的研究工作。如果大家发现模型库有什么问题，或者有自己实现的GAN想要发布在这里，请大家在github提交issue或者pr。


### 参考

[1][openseg](<https://github.com/openseg-group/openseg.pytorch>)

[2][Pytorch-Deeplab](<https://github.com/jfzhang95/pytorch-deeplab-xception>)

[3][DANet](<https://github.com/junfu1115/DANet>)

[4][EMANet](<https://github.com/XiaLiPKU/EMANet>)

