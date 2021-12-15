# Assignment-5

Modules used:
Keras
Tensorflow
 
 
The file “optdigits-orig.windep” is loaded and unzipped using python. The input values are read from the 22nd line and every 32nd line is inserted into the inputArray and the 33rd line is taken as the target value. The target values are then converted into an array with the shape of (10, 1) which has 10 elements in each array and the index of max value in the array corresponds to the output.
Number of layers in the network = 4 The neural network has 4 layers. The output layer provides a 10 dimensional vector which has the predicted values. The one with the highest value will be taken as the output. An image called “outputGraph.png” is stored in the same directory giving a graphical representation of the output.

Directions to Execute:
The file final.py is executed by giving the proper path to the input file “optdigits-orig.windep”. In the program the input file path is given as I used Google Colab to run the program. 
