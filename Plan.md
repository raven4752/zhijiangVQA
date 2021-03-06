# 2018之江杯全球人工智能大赛：视频识别 问答
## 问题
给一组视频，每个视频有若干个问题，对给个问题回答一个答案
## 数据集
1. 训练集 3000+ 视频 每个视频 5个问题，每个问题3个答案，有重复。
总共约30000个不重复 视频-问题-答案对
视频里包含卡通视频。。（甚至还有海绵宝宝） （甚至还有有手机拍摄的海绵宝宝）
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
## 坏结果分析

分析转为n类候选答案+multi-instance visual question answer的模型的预测结果(acc~=0.50)：

| questions                                   |   freq in wrong |   freq in correct |
|:--------------------------------------------|----------------:|------------------:|
| what is in the video                        |             153 |                88 |
| what is the person in the video doing       |             110 |                28 |
| what is the person doing in the video       |             103 |                22 |
| what is the man doing in the video          |              45 |                10 |
| what is the man in the video doing          |              35 |                 6 |
| what is the person doing in video           |              28 |                 7 |
| what is the woman in the video doing        |              18 |                 4 |
| what is in front of the person in the video |              14 |                 0 |
| what can you see in the video               |              13 |                 0 |
| what is the person in video doing           |              13 |                 0 |
| what is the man doing                       |              11 |                 0 |



上表列出了在错误回答中各出现至少4次，且错误回答数目比正确回答多10次以上的问题。
显然可以看出只看一帧的模型很难回答“做什么”的问题 (约1/3的错误回答是正在做什么的问题)

同时，怀疑答案编码存在一些问题，很多二元组也没有回答对
## 解决方案
1. 问题建模。原始问题不能直接处理，需要进行转换。

    - [x] 转换为n(1000)类候选答案
        - [x] 转换为视频-问题-单个答案对 ，建模为视频多分类问题
        - [x] 转换为视频-问题-多个答案对， 建模为视频多标签问题
        - [x] 转换为图像问题回答
2. 迁移学习对这个问题来说非常重要，需要找到一个能迁移到该数据集上的模型

    - [x] [VGG-19 on imagenet](https://keras.io/applications/#vgg19) 图像分类模型， 可标注每一帧的特征
    - [x] [resnext-101 on kinetics](https://github.com/raven4752/video-classification-3d-cnn-pytorch) 视频分类模型，可标注每64帧的特征
    - [ ] resnet-101 on imagenet on kinetics 用imagenet预训练的图像分类模型在kinetics数据集上训练的视频分类模型
    - [x] fast-rcnn 物体识别模型
    - [ ] 问答模型的句子/答案编码器
    - [ ] 图像问题回答模型 on VQA v2
3. 由于数据较少，数据增广可能对结果有着重要的影响
    - [ ] 视频级别增广
        - [ ] 全局亮度调整
        - [ ] 全局旋转
        - [ ] 全局剪切
4. 考虑到仍然有一些问题需要了解视频整体的特征，一些结合帧级别特征生成视频特征的模型可能有作用。
## 参考资料
1. [图像问题回答训练技巧](https://arxiv.org/abs/1708.02711)
2. [图像问题回答模型](https://arxiv.org/abs/1707.07998)
3. [第二次youtube-8m竞赛冠军方案](https://www.kaggle.com/c/youtube8m-2018/discussion/62781)
4. [视频动作分类](https://github.com/kenshohara/video-classification-3d-cnn-pytorch)


## Unknown Bugs
1. ~~训练时用多帧增强数据集，无效果；epoch间更换每个视频训练的帧，训练集损失降低很快，测试集性能大幅下降。 但是
集成时似乎采样帧不同的模型效果效果更好？且测试时平均每帧d的预测结果似乎效果更好？~~(代码有bug)
2. 观测发现数据集的大多数问题都可以通过观察某一帧回答，与实验结果相违背（再次观察发现有一部分问题属于行为问题）
3. ~~cv-ensemble+模型改进只提高了两个千分点？？~~ 可能是因为sub instance level averaging 引起的bug
## TODO
- [x] 迁移到sacred框架
- [ ] 为mongodb数据库添加加密，用mongodb管理artifacts
- [ ] 添加tensorboard输出
- [x] 模拟云端的性能估计
- [ ] 尝试不同的问题建模
    - [ ] fine tuning vgg/resnet
    - [ ] seq2seq answer encoding
- [x] 懒惰的资源创建和加载
    - [x] 按需加载资源
- [ ] 重构项目
    - [x] 支持用yml格式的配置文件创建实验
    - [ ] 将training 从main中分离出来
    - [ ] 添加capture，监视训练过程
    - [ ] 保存预测结果为(video_id,question,answer)的列表，容易合并
    - [ ] 划分验证集
    - [ ] 支持 pytroch/keras 模型
