# JDA
This is a tensorflow implementation of the paper *[Domain Adaptation for EEG Emotion Recognition Based on Latent Representation Similarity](https://ieeexplore.ieee.org/abstract/document/8882370)*

This work is based on *[DANN](https://arxiv.org/abs/1409.7495)* and *[associative domain adaptation](http://openaccess.thecvf.com/content_iccv_2017/html/Haeusser_Associative_Domain_Adaptation_ICCV_2017_paper.html)*. 
See https://github.com/pumpikano/tf-dann/blob/master/utils.py and https://github.com/haeusser/learning_by_association .

#### Environment
- Tensorflow 1.4.1
- Python 2.7

#### Network Structure

![](/pic/JDA_network.png)

#### Data

*SEED* dataset and *DEAP* dataset of emotion EEG dataset. 

You should change the data dir variable in `utils.py` to your data path.