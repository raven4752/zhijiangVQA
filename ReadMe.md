#2018之江杯全球人工智能大赛 ：视频识别 问答
## 问题
给一组视频，每个视频有若干个问题，对给个问题回答一个答案
## 数据集
1. 训练集 3000+ 视频 每个视频 5个问题，每个问题3个答案，有重复。
总共约30000个不重复 视频-问题-答案对
视频里包含卡通视频。。（甚至还有海绵宝宝）
2. 测试集 1000 左右视频 ，每个视频 5个问题，要求对每个问题给出一个答案。
候选答案相当有限。 1000个常见答案 占据了所有答案的85%。
3. 数据集很脏，甚至存在格式错误。
4. 视频目测比较短，多数不超过1000帧，码率多在30帧/秒，有的有音频有的没有。
5. 问句长度不超过20，多数不超过15词
6. 目测多数问句仅考虑了一个瞬间的事，需要了解视频中按时序发生的事件的问题比较少。
## 问题难点
1. 数据显然不够训练基于深度网络的端到端的问答系统，参考[这里](https://arxiv.org/abs/1711.09577)给出的结果。
2. 该问题之前缺少研究结果。最相似的问题包括 Visual Qustion Answer, Video Classification 。
3. 视频数量虽然少， 但是帧数长，由于显存限制，难以直接建模为3D输入。
4. 视频和问题提取出的特征相差甚远，难以混合。

## 解决方案
1. 问题建模。原始问题不能直接处理，需要进行转换。
    -[x] 转换为n(1000)类候选答案
        -[x] 转换为视频-问题-单个答案对 ，建模为视频多分类问题
        -[x] 转换为视频-问题-多个答案对， 建模为视频多标签问题
        -[x] 转换为图像问题回答
2. 迁移学习对这个问题来说非常重要，需要找到一个能迁移到该数据集上的模型

    -[x] [VGG-19 on imagenet](https://keras.io/applications/#vgg19) 图像分类模型， 可标注每一帧的特征
    -[x] [resnext-101 on kinetics](https://github.com/raven4752/video-classification-3d-cnn-pytorch) 视频分类模型，可标注每64帧的特征
    -[ ] resnet-101 on imagenet on kinetics 用imagenet预训练的图像分类模型在kinetics数据集上训练的视频分类模型
    -[x] fast-rcnn 物体识别模型
    -[ ] 图像问题回答模型 on VQA v2
3. 由于数据较少，数据增广可能对结果有着重要的影响
    -[ ] 视频级别增广
        -[ ] 全局亮度调整
        -[ ] 全局旋转
        -[ ] 全局剪切
## 参考资料
1. [图像问题回答训练技巧](https://arxiv.org/abs/1708.02711)
2. [图像问题回答模型](https://arxiv.org/abs/1707.07998)
3. [第二次youtube-8m竞赛冠军方案](https://www.kaggle.com/c/youtube8m-2018/discussion/62781)
4. [视频动作分类](https://github.com/kenshohara/video-classification-3d-cnn-pytorch)

## TODO
-[x] 迁移到sacred框架
-[ ] 为mongodb数据库添加加密，用mongodb管理artifacts
-[ ] 添加tensorboard输出
-[x] 模拟云端的性能估计
-[ ] 尝试不同的问题建模
-[ ] 懒惰的资源创建和加载