# CNN-Based-Sentiment-Analysis-System

The CNN system in this work is implemented using an available open source Matlab library, MathConvNet,
and is optimized through a set of trial-and-error experiments. The performance of the designed CNN is then
compared with recent similar publications to demonstrate its effectiveness. 

This project using CNN to create a model to find the sentiment of a comment.

## CNN Architecture

To implement the CNN architecture the review sentences are to be pre-processed to have a proper format as
inputs of the CNN. To do this, the tokenized sentences are being used, which includes all the words and symbols
in a sentence as a token separated from each other by spaces. Each sentence is then being converted into a
Sentence Matrix which is a matrix representation of the sentence including all its tokens as matrix rows [2].
During the next step a dictionary of all the unique tokens is created while assigning an index for each token
(a mapping index). The dictionary is being stored as a mean to represent different sentences with numerical
matrices. The numerical representation is used as an input for the CNN. For the last step of data pre-processing,
the inputs which have smaller dimension than the maximum filter size of CNN, are padded so that all the CNN
inputs have a minimum dimension.

## Datasets

The initial dataset which was provided for this project was a Movie Review database including 6000 labeled
data. This dataset was utilized in order to design the CNN system and train it to achieve a reasonable accuracy.
However, in order to increase the accuracy of the sentiment classification, similar databases were utilized to
train the network. The main database added to the initial one, was SST-2 (Stanford Sentiment Tree) which
includes 9616 sentences with their sentiment labels [3].
In addition, a customer review dataset with 3774 sentences was also tested, which resulted in decreasing
the accuracy of the system [4]. The reason of the degraded performance was that the customer review dataset
included casual comments about household devices, which in the nature was different than professionally-written
movie reviews. This type difference resulted in training issues for the network; hence, degrading the performance.
