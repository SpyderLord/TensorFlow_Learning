It's time for us to move onto an absolute new area:Semantic Segmantation!Here we come~
Deconvolution 在FCN中是作为上采样的一种方式，这种Upsample是Learnable的，更好的命名应该是transpose convolution\
First,we have to know,in the object detection mission,we use Fully convolutional networks instead convolutional neural networks.The difference between these two architecture is that there is no fully connected layers in the FCN,for this layer drop the location information,which matters a lot in the localization.All the layers in the FCN are 3-d array of size h*W*d,and that is why it calls FCN.
One important operation in the FCN is Down Sampling and Upsampling,while the later operation make the output the same size as the input image.
What we have to do is we need label all the pixels in the image and finally gat a labeled image.
An FCN naturally operates on an input of any size and produces an output of corresponding spatial dimensions.
There is something that we need to know:it's necessary for us to convert the fully-connected layers to Conv layers.In practice the ability to convert an FC layer to a CONV layer is particularly useful.In other words,we are setting the filter size to be exactly the same size of the input,so the output will simply be 1*1*4096 since only a single depth column "fits" across the input volumn,giving identical results as the initial FC layer.
Note that the fully connected layers have fixed dimensions and throw away the spatial coordinates.If we can view the connected layers as convolutions kernel of the same size of the input,the fully connected layers can be viewed as convolutions.

In the FCN paper,the writer said that they append a 1*1 convolution with channal dimension 21 to predict the classes.----That is understandable.
Transforming fully connected layers into convolution layers enable a classification net output a heatmap.Adding layers and a spatial loss produces an efficient machine for end-to-end dense learning.

I finally figure out the architecture of the FCN.First we have to transform the fully connected layer into fully convolutional layers,then we do the upsampling operation so that we get class number output with the same size of the input.At last we combine all the output to get the heatmap.

看一下为什么传统的CNN需要固定输入图片的大小。
对于CNN，一幅输入图片在经过卷积和pooling层的时候，这些层是不会关心图片的大小的。在进入全连接层的时候，feature map成为一个一维列向量，然后这个一维列向量的每一个元素作为结点都要和下一个层的所有结点进行全连接。神经网络的结构一旦确定下来，它的权值都是固定的，层层向回看，每一个output size都是要固定的，因此输入图片的输入的尺寸是需要固定的。
一个确定的CNN网络结构之所以要固定输入图片的大小，是因为全连接层权值数是固定的，而权值数和feature map的大小有关系。

然后是反卷积层，反卷积层也是卷积层，不关心input的大小，滑窗卷积之后输出output。

那么非常重要的一个问题是如何使反卷积的输出和输入的图片大小一致？在FCN中，所有的层都是卷积层，卷积层不关心输入的大小，输入和输出之间是存在线性关系的，spatial size是和kernel size相关联的。

几个参考的中文链接：http://blog.csdn.net/happyer88/article/details/47205839
http://blog.csdn.net/meanme/article/details/50858240、
http://blog.csdn.net/smf0504/article/details/52745052

Semantic Segmentation有两个固有的性质：
1.semantic：global information 主要解决的是目标是什么的问题
2.location: local information 解决目标在哪的问题。
