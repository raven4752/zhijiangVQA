#结果复现说明

##依赖环境
1. ubuntu 16.04 python3.5, 如遇到opencv相关错误，执行


    apt-get install -y libsm6 libxext6 libxrender-dev
   
2. python package 见requirements.txt
3. CUDA 版本为9.0 , cudnn 版本为7.0.5
##预训练模型
我使用了https://github.com/peteanderson80/bottom-up-attention 这个项目的预训练模型生成特征。
如要复现该过程，请按照 https://github.com/peteanderson80/bottom-up-attention 中作者的说明配置环境，再

    pip install h5py
再将该项目放于根目录中，并用我提供的文件覆盖原文件。再进入目录执行 repeat.sh 。可能需要较多显存。

注意该模型的运行环境和本项目依赖环境不同，配置比较麻烦，如果不能配置，可到[这里](https://raven4752.mynetgear.com/nextcloud/index.php/s/UIdezBbbGSfXA4O)下载我处理好的特征和其他中间文件，在根目录中解压。

##结果复现说明
如果使用了我提供的中间结果，执行

    cd code
    python main.py 
    cd ..    
否则需要:
1. 下载预训练的glove词向量
    
    
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    mkdir -p input
    unzip glove.840B.300d.zip
    mv glove.840B.300d.txt input/
 
2. 预处理数据

 拼接A,B数据集：
    
    
    cat  data/DatasetA/train.txt >> data/DatasetB/train.txt
    cp  data/DatasetB/train.txt data/train.txt
    cp data/DatasetB/test.txt data/test.txt
    
   处理词向量，Tokenizer和 答案的编码器
    
    cd code
    python utils.py  fit_encoder --train_path=../data/train.txt
    python utils.py fit_tokenizer --train_path=../data/train.txt --test_path=../data/test.txt
    python utils.py embedding2numpy
    cd ..
处理之前用预训练模型生成的特征

    mkdir -p input/faster_rcnn_10f
    mv bottom-up-attention/tr.h5 input/faster_rcnn_10f/
    mv bottom-up-attention/te.h5 input/faster_rcnn_10f/
    cd code  
    python utils.py dump_meta_data --raw_path=../data/train.txt --feature_path=../input/faster_rcnn_10f/tr.h5
    python utils.py dump_meta_data --raw_path=../data/test.txt --feature_path=../input/faster_rcnn_10f/te.h5
    python utils.py pickle_h5 --raw_path=../data/train.txt --feature_path=../input/faster_rcnn_10f/tr.h5
    python utils.py pickle_h5 --raw_path=../data/test.txt --feature_path=../input/faster_rcnn_10f/te.h5
    cd ..
3. 训练模型


    cd code
    python main.py 
    cd ..

生成的预测结果在submit文件夹下。
##模型结构
见code/model.py
