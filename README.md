# ML MNIST Challenge
This contest was offered within TU Munich's course Machine Learning 1 (IN2064).<br>
The goal was to implement k-NN, Neural Network, Logistic Regression and Gaussian Process Classifier in 
python from scratch and achieve minimal average test error among these classifiers on well-known MNIST dataset, 
without ensemble learning.

## Results
| Algorithm | <div align="center">Description</div> | Test Error, % |
| :---: | :--- | :---: |
| ***k-NN*** | 3-NN, Euclidean distance, uniform weights.<br/>*Preprocessing*: Feature vectors extracted from ***NN***. | **1.13** |
| ***k-NN<sub>2</sub>*** | 3-NN, Euclidean distance, uniform weights.<br/>*Preprocessing*: Augment (training) data (&#215;9) by using random rotations,<br/>shifts, Gaussian blur and dropout pixels; PCA-35 whitening and multiplying<br/>each feature vector by e<sup>11.6 &#183; ***s***</sup>, where ***s*** &ndash; normalized explained<br/>variance by the respective principal axis. (equivalent to applying PCA<br/>whitening with accordingly weighted Euclidean distance. | **2.06** |
| ***NN*** | MLP 784-1337-D(0.05)-911-D(0.1)-666-333-128-10 (D &ndash; dropout);<br/>hidden activations &ndash; LeakyReLU(0.01), output &ndash; softmax; loss &ndash; categorical<br/>cross-entropy; 1024 batches; 42 epochs; optimizer &ndash; *Adam* (learning rate<br/>5 &#183; 10<sup>&ndash;5</sup>, rest &ndash; defaults from paper).<br/>*Preprocessing*: Augment (training) data (&#215;5) by using random rotations,<br/> shifts, Gaussian blur. | **1.04** |
| ***LogReg*** | 32 batches; 91 epoch; L2-penalty, &#955; = 3.16 &#183; 10<sup>&ndash;4</sup>; optimizer &ndash; *Adam* (learning<br/>rate 10<sup>-3</sup>, rest &ndash; defaults from paper)<br/>*Preprocessing*: Feature vectors extracted from ***NN***. | **1.01** |
| ***GPC*** | 794 random data points were used for training; &#963;<sub>n</sub> = 0; RBF kernel (&#963;<sub>f</sub> = 0.4217,<br/>&#947; = 1/2l<sup>2</sup> = 0.0008511); Newton iterations for Laplace approximation till<br/>&#916;Log-Marginal-Likelihood &leq; 10<sup>&ndash;7</sup>; solve linear systems iteratively using CG with<br/> 10<sup>&ndash;7</sup> tolerance; for prediction generate 2000 samples for each test point.<br/>*Preprocessing*: Feature vectors extracted from ***NN***. | **1.59** |

Also check for some plots (confusion matrices, learning curves etc.).

## How to install
```bash
git clone https://github.com/monsta-hd/ml-mnist
cd ml-mnist/
sudo pip install -r requirements.txt
```
Optionally, one can also run tests:
```bash
make test
```

## How to run
Check `main.py` to reproduce training and testing the final models.<br>
Check `cross_validations.ipynb` to see what I've tried, cross-validations etc.<br>

## System
All computations and time measurements were made on laptop `i7-5500U CPU @ 2.40GHz x 4` `12GB RAM`

## Future work
