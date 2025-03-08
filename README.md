
# Overview

nn.h is a minimalist neural network library implemented in C. Designed to harness the efficiency of low-level programming, it abstracts away complex details like memory management, allowing users to focus solely on building and optimizing neural networks. Additionally, the project includes performance comparisons with PyTorch to demonstrate the advantages of using C for high-performance computations.


## Implementation & Comparison
To demonstrate the optimization capabilities of this library, an adder circuit is implemented using nn.h. The neural network has one hidden layer with 2 * bits + 1 neurons. Increasing the number of bits results in a larger hidden layer (having more neurons), adapting the network's complexity accordingly. This is a real a real time loss minimisation for a 4 bit adder data:
![Demo Screenshot]https://github.com/itoufik/Building-a-NN-library-in-C/blob/master/demos/loss_plot.png).

This is the comparision with PyTorch keeping all the hyperpameters same:

| Data   | Training time in C (sec)   | Training time in PyTorch (sec)  |
|:-------------|:--------------:|-------------:|
| 2 bit adder         | 1.17      | 27.12          |
| 4 bit adder    | 30.12 | 30.21          |
| 6 bit adder    | 1148.37 | 67.24         |

It is evident that, for simpiler networks this library can be 20x faster, but as the network get bigger, PyTorch can be much faster.
## TODO
Add CUDA optimisation