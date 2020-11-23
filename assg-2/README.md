# Assignment 2

## Part A: Object Recognition

The project uses a sample of the CIFAR-10 dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

The dataset contains RGB color images of size 32x32 and their corresponding labels from 0 to
9. You will be using the **batch_1** of the dataset, which contains 10,000 training samples.
Testing is done on 2,000 test samples. The training data and testing data are provided in files
`data_batch_1` and `test_batch_trim`, respectively.

<ol>
  <li>
    Design a convolutional neural network consisting of:
    <ul>
      <li>An Input layer of 32x32x3 dimensions
      <li>A convolution layer 𝐶! with 50 channels, window size 9x9, VALID padding, and ReLU
  activation
      <li>A max pooling layer 𝑆! with a pooling window of size 2x2, stride = 2, and VALID padding
      <li>A convolution layer 𝐶" with 60 channels, window size 5x5, VALID padding, and ReLU
  activation
      <li>A max pooling layer 𝑆" with a pooling window of size 2x2, stride = 2, and VALID padding
      <li>A fully-connected layer 𝐹# of size 300 with no activation
      <li>A fully-connected layer 𝐹$ of size 10 with Softmax activation
    </ul>
  </li>
  
  <li>
    Train the network using mini-batch gradient descent learning for 1000 epochs. Set the
batch size to 128, and learning rate 𝛼 = 0.001.
  <ol>
    <li>Plot the (1) training cost, (2) test cost, (3) training accuracy, and (4) test accuracy
against learning epochs. One plot for the costs and one plot for the accuracies.</li>
    <li>For the first two test images, plot the feature maps at both convolution layers
(𝐶! and 𝐶") and pooling layers (𝑆! and 𝑆") along with the test images. (In total one
image and four feature maps)</li>
    <b>Note: A feature map with N channels can be viewed as N grayscale images. Do
      make sure that the pixel values are in the correct range when you plot them.</b>
  </ol>
  </li>
  
  <li>Use a grid search ( 𝐶!Î {10, 30, 50, 70, 90}, 𝐶"Î {20, 40, 60, 80, 100} , in total 25
combinations) to find the optimal combination of the numbers of channels at the
convolution layers. Use the test accuracy to determine the optimal combination. Report
all 25 accuracies.</li>
  
  <li>
    Using the optimal combination found in part (2), train the network by:
  <ul>
    <li>adding the momentum term with momentum 𝛾 = 0.1,
    <li>using RMSProp algorithm for learning,
    <li>using Adam optimizer for learning,
    <li>adding dropout (probability=0.5) to the two fully connected layers.
  </ul>
    Plot the costs and accuracies against epochs (as in question 1(a)) for each case. Note that
the sub-questions are independent. For instance, in (d), you do not need to modify the
optimizer.
  </li>
  
  <li>Compare the accuracies of all the models from parts (1) - (3) and discuss their performances.</li>
</ol>

## Part B: Text Classification

The dataset used in this project contains the first paragraphs collected from Wikipage entries
and the corresponding labels about their category. You will implement CNN and RNN layers
at the word and character levels for the classification of texts in the paragraphs. The output
layer of the networks is a softmax layer.

The training and test datasets will be read from `train_medium.csv` and `test_medium.csv`
files. The training dataset contains 5600 entries and test dataset contains 700 entries. The
label of an entry is one of the 15 categories such as people, company, schools, etc.

The input data is in text, which should be converted to character/word IDs to feed to the
networks (Please refer to our given two smaple codes (CNN and RNN) which process text data
in terms of character and word respectively). Restrict the maximum length of the
characters/word inputs to 100 and the maximum number of training epoch to 250. Use the
Adam or SGD optimizers for training with a batch size = 128 and learning rate = 0.01. Assume
there are 256 different characters.

<ol>

  <li>
    Design a Character CNN Classifier that receives character ids and classifies the input. The
CNN has two convolution and pooling layers:
    <ul>
    <li>A convolution layer 𝐶! of 10 filters of window size 20x256, VALID padding, and ReLU
neurons. A max pooling layer 𝑆!with a pooling window of size 4x4, with stride = 2, and
padding = 'SAME'.
    <li>A convolution layer 𝐶" of 10 filters of window size 20x1, VALID padding, and ReLU
neurons. A max pooling layer 𝑆" with a pooling window of size 4x4, with stride = 2 and
padding = 'SAME'.
    </ul>
    Plot the entropy cost on the training data and the accuracy on the testing data against
training epochs.
  </li>
  
  <li>
    Design a Word CNN Classifier that receives word ids and classifies the input. Pass the
inputs through an embedding layer of size 20 before feeding to the CNN. The CNN has
two convolution and pooling layers with the following characteristics:
    <ul>
    <li>A convolution layer 𝐶! of 10 filters of window size 20x20, VALID padding, and ReLU
neurons. A max pooling layer 𝑆! with a pooling window of size 4x4, with stride = 2 and
padding = 'SAME'.
    <li>A convolution layer 𝐶" of 10 filters of window size 20x1, , VALID padding, and ReLU
neurons. A max pooling layer 𝑆" with a pooling window of size 4x4, with stride = 2 and
padding = 'SAME'.
    </ul>
    Plot the entropy cost on training data and the accuracy on testing data against training
epochs.
  </li>
  
  <li>
    Design a Character RNN Classifier that receives character ids and classify the input. The
RNN is GRU layer and has a hidden-layer size of 20. Plot the entropy cost on training data and the accuracy on testing data against training
epochs.
  </li>
  
  <li>
    Design a word RNN classifier that receives word ids and classify the input. The RNN is GRU
layer and has a hidden-layer size of 20. Pass the inputs through an embedding layer of size
20 before feeding to the RNN. Plot the entropy on training data and the accuracy on testing data versus training epochs.
  </li>
  
  <li>
    Compare the test accuracies and the running times of the networks implemented in parts
(1) – (4). Experiment with adding dropout to the layers of networks in parts (1) – (4), and report
the test accuracies. Compare and comment on the accuracies of the networks
with/without dropout.
  </li>
  
  <li>
    For RNN networks implemented in (3) and (4), perform the following experiments with
the aim of improving performances, compare the accuracies and report your findings:
  <ul>
    <li>Replace the GRU layer with (i) a vanilla RNN layer and (ii) a LSTM layer
    <li>Increase the number of RNN layers to 2 layers
    <li>Add gradient clipping to RNN training with clipping threshold = 2.
  </ul>
  </li>

</ol>


