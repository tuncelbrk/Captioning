A project that interprets a photograph given as a subtitle in the scope of the project has been realized. Also you can examine presentation of project.


System Design

In this project, contains an implementation of image captioning based on neural network (i.e. CNN + RNN). The model first extracts the image feature by CNN and then generates captions by RNN. CNN is VGG16 and RNN is a standard LSTM . Normal Sampling and Beam Search were used to predict the caption of images. The program is given the first photographs. Photographs are coded to reveal distinct objects. The photos are kept in a list after they are typed.


Network Topology

  -Encoder
  The CNN can be thought of as an encoder. The input image is given to CNN to extract
the features. The last hidden state of the CNN is connected to the Decoder.

  - Decoder
The Decoder is a RNN which does language modelling up to the word level. The first
time step receives the encoded output from the encoder and also the vector.

