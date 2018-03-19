# Some thing to sum up about the neural net:
To train a neural network:
- Gradient check your implementation with a small batch of data and be aware of the pitfalls
- As a sanity check,make sure that your initial loss is reasonable,and that yo can achieve 100% training accuracy on a small portion of the data.
- During training,monitor(跟踪) the loss,the training/validation accuracy and if you're feeling fancier, the magnitude of the updates in relation to parameter value.(如果愿意的话，还可以跟踪更新的参数量相对于总参数的比例，然后如果是对于卷积神经网络，可以将第一层可视化。)
- The two recommended updates to use are SGD+Nesterov Momentum o‘r Adam(both for optimization)
- Decay your learning rate over the period of the training.For /instance,halve the learning rate after a fixed number of epochs,or whenever the validation accuracy tops off.
- Search for good hyperparamters with random search.
- Form model ensembles for extra performance.

## Something about TensorFlow
- 如何使用变量和操作组成更大的集合，怎么运行这个集合。这就是计算图谱Graph和Session的作用了一个TensorFlow的运算，被表示为一个数据流的图。一幅图中包含一些操作（operation）对象，这些对象是计算节点。一个Tensor对象，则是表示在不同的操作间的数据节点。一旦我们开始我们的任务，就会有一个默认的图已经建立好了。添加一个操作到默认的图里面，只要简单的调用一个定义了新操作的函数就可以了。
- 运行一个TensorFlow的操作（operation）的类，一个session包含了操作对象执行的环境。
- 就属性而言，graph是投放到session中的图。
- 重要的函数有：session.run（fetch,feed_dict=None,options=None,run_metadata=None）
### 这个函数的作用是运行操作计算tensor
参数：一个单独的图的元素，或者一个图的元素的列表，或者是一个字典，这个字典就是刚刚所说的一个图的元素（元素列表）
feed_dict:一个字典，为之前占位的元素喂值
返回值：函数返回的值和传入的fetch参数的形状是一样的。只是里面的元素是相应的值而已。

## TensorFlow中的一些基本概念：
### 构建并执行计算图的必要过程：
- graph图计算，使用tf训练神经网络包括两个部分，构建计算图和运行计算图/
首先是构建图，一个计算图包含了一组操作（operation objects,也叫节点，用来进行数据的计算）和一些tensor object（节点直接传递的数据单元）。系统会默认构建一个计算图，这也是我们可以直接进行定义节点的原因。但是在我们写代码的时候，应该将构建计算图的代码写在一个with块中。
mygraph=tf.grapgh()
with myGraph.as_default():
	c=tf.constant(30.0)
- operations（图节点）：
op是计算图中的节点，它能够接收tensor作为输入，并能够产生tensor作为输出。op的创建方式有两种，一种是调用了计算操作，比如说一些计算的函数tf.nn.conv2d()、tf.nn.max_pool()
这些函数名就是节点op。另一种方法是调用Graph.create_op()方法往图里添加op。通常我们都是使用第一种方案。op在构建图的时候是不会执行的，只有在执行图的时候才会执行。
- tensor（向量）
tensor就是op计算的输入/输出结果。同样的，tensor在构建图的时候并不持有值，，而是在运行的时候才会持有数值。tensor作为op的输入/输出在graph中进行传递，从而使得tensorflow能够执行代表着大规模的计算图，这也正是tensorflow得名的原因（向量的流动）。常量和变量都是tensor。
- session（会话）
在前三歩中我们完成了构建一个计算图表，现在我们需要执行这个计算图表。session就是执行计算图表的类。tensor的执行要通过sess.run(tensor)来执行。session对象封装了op执行和tensor传递的环境。session对象开启之后会持有资源，所以在使用完之后需要将它关掉。
### tensorflow中的图变量：
和我们平时所接触的一般的变量在用法上有很大的差异，简单说明：
- 如何初始化图变量。
- 两种定义图变量的方法
- scope如何划分命名空间
- 图变量的复用
- 图变量的种类

## 图变量的初始化方法：
- 在tensorflow中，变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要通过tf.session的run来进行。想要将所有图变量进行集体化初始化的时候我们使用tf.global_variables_initializer

### 两种定义图变量的方法：
- tf.get_variable跟tf.Variable都可以用来定义图变量，但是前者的必需参数（即第一个参数）并不是图变量的初始值，而是图变量的名称。

- tf.Variable的用法要更丰富一点，当指定名称的图变量已经存在时表示获取它，当指定名称的图变量不存在时表示定义它。对应的第一个参数是变量的值。

- 如果是使用get_variable 进行变量定义的话，变量的名字是不能重复的，但是用variable进行定义的时候是可以的重复的，但是当我们打印出来他们的属性的时候会发现他们并不是完全一样的。

## 使用scope进行命名空间的划分：
太多了不想打了：粘贴链接：http://blog.csdn.net/gg_18826075157/article/details/78368924
