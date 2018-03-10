Some thing to sum up about the neural net:
To train a neural network:
1.Gradient check your implementation with a small batch of data and be aware of the pitfalls
2.As a sanity check,make sure that your initial loss is reasonable,and that yo can achieve 100% training accuracy on a small portion of the data.
3.During training,monitor(跟踪) the loss,the training/validation accuracy and if you're feeling fancier, the magnitude of the updates in relation to parameter value.(如果愿意的话，还可以跟踪更新的参数量相对于总参数的比例，然后如果是对于卷积神经网络，可以将第一层可视化。)
4.The two recommended updates to use are SGD+Nesterov Momentum o‘r Adam(both for optimization)
5.Decay your learning rate over the period of the training.For /instance,halve the learning rate after a fixed number of epochs,or whenever the validation accuracy tops off.
6.Search for good hyperparamters with random search.
7.Form model ensembles for extra performance.
