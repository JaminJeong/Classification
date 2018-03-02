mkdir pretrained_model
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
mv resnet_v1_50_2016_08_28.tar.gz pretrained_model
mv vgg_19_2016_08_28.tar.gz pretrained_model
mv vgg_16_2016_08_28.tar.gz pretrained_model
