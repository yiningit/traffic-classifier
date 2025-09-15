# Traffic
Task: Use Tensorflow to build a neural network to classify road signs based on an image of those signs.

## Experimentation Process
I began with a basic model with one convolutional, pooling and hidden layer:
- a convolutional layer with 32 filters and 3x3 kernel matrix
- a max pooling layer with 2x2 pool size and stride of 2
- a flatten layer
- a densely-connected hidden layer with 64 nodes
- an output layer with NUM_CATEGORIES nodes

This gave an evaluation accuracy of over 88%, although an accuracy of over 93% was reached in the training set, indicating overfitting.

Aiming for an **evaluation accuracy of over 99%** with **minimal loss and overfitting**, I experimented with different numbers of convolutional, pooling and hidden layers, as well as tuning hyperparameters to see how these changes effected the model's efficacy.

### During my experiementation, I noticed:
- Adding convolutional layers and increasing filter sizes both increased the rate at which training accuracy increases. I also noticed that two convolutional layers results in roughly equal accuracy/loss values as a single layer with their combined filter size, but the latter results in larger output shape, so takes much longer to train. I ended up favouring **multiple convolutional layers with smaller filter sizes**.
- A higher number of hidden layers and nodes increased training time, but did not always increase accuracy.
- A single hidden layer was sufficient to obtain high evaluation accuracy. The optimal number of nodes for this hidden layer seemed to generally be much smaller than (less than 1/4 of) the shape of the data passed into it. Having more nodes than the shape of the flattened data also resulted in much worse overfitting.
- In general, as the shape of the input entering the hidden layer increases, training and evaluation accuracy increases. However, time taken to train the model also increases. 
- Two pooling layers of 2x2 pool size and stride of 2 resulted in **higher accuracy and lower loss** than a single pooling layer with 4x4 pool size and stride of 4.
- **The position of the pooling layer has a large impact on the model outside of its effect on the output shape of the flattened data.** Placing the pooling layer **after all convolutional layers** appeared to be optimal, with evaluation and training accuracy decreasing the more convolutional layers that follow the pooling layer. Placing the pooling in between the 4 convolutional layers (2-2 split) resulted in 93.5% accuracy, while a 3-1 split resulted in 99.2% accuracy. Placing it after all four convolutional layers (4-0 split) resulted in 99.4% accuracy.
- Implementing **dropout with a rate of 50%** on the hidden layer worked best for minimising overfitting without losing out on the final evaluation accuracy.
- Adding dropout to the pooling layer effectively reduced overfitting. I found a rate of 10% was ideal for my final structure as higher rates resulted in lower accuracy, while less or no dropout resulted in overfitting.
- The **RMSprop algorithm** performed slightly better than the Adam algorithm for the optimizer. Both performed much better than the SGD algorithm.
- Using **binary cross-entropy** for the loss function performed better in both accuracy and loss than cateogrical cross-entropy.

## Final structure of model
- 4 convolutional layers with 32 filters, 3x3 kernel matrix and ReLU activation
- 1 pooling layer with 2x2 pool size and stride of 2, with 10% dropout
- a flatten layer
- 1 hidden layer with 256 nodes and ReLU activation, with 50% dropout
- an output layer with NUM_CATEGORIES nodes and softmax activation

#### My final model takes around 7s per epoch to train. Over 6 runs, it had an average evaluation accuracy of **99.25%**, average training accuracy of 99.28% and an average loss of 0.0030.