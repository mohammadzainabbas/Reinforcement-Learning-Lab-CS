## Basics of Deep Learning and Pytorch

For this lab, you need to have some ground understanding of pytorch, and basics introduction is available [here](https://pytorch.org/tutorials/beginner/basics/intro.html).

```math
$\newcommand{\underbr}[2]{\underbrace{#1}_{\scriptscriptstyle{#2}}}$
```

### Objective

We want to implement a two layers `Multi-Layer Perceptron` (MLP) with 1 hidden layer in Pytorch, for a binary classification problem.

The output of the network is simply the output of several cascaded functions :
- Linear transformations. We note the weights of a linear transformation with $W$
- Additive biases. We note the parameters of additive biases  with $b$
- Non-linearities.

For this, we will implement in the first part of the lab:
- the forward propagation
- the computation of the loss
- the backward propagation (to obtain the gradients)
- the update of the parameters

In the second part we will simply use pytorch API with a multi-classification problem.

Furthermore, we define the following sizes :

- $n^{[0]}$ : number of input neurons
- $n^{[1]}$ : number of neurons in hidden layer
- $n^{[2]}$ : number of neurons in output layer
- $m$ : number of training datapoints

### Loss function 

We want to solve a binary classification problem. Therefore we will use the binary cross-entropy loss function. The total loss function will be the average of the **loss** over the training data.

$\mathcal{L} = - \left( y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right),$

where 
- $y$ is the ground-truth labels of the data 
- $\hat{y}$ the predicted labels outputed by the network.

### Forward propagation

- $\large \underbr{Z^{[1]}}{(m,n^{[1]})} = \underbr{X}{(m,n^{[0]})} \underbr{W^{[1]}}{(n^{[0]},n^{[1]})}  + \underbr{b^{[1]}}{n^{(1)}} $
- $\large \underbr{A^{[1]}}{(m,n^{[1]})} = f(Z^{[1]})$
- $\large \underbr{Z^{[2]}}{(m,n^{[2]})} = \underbr{A^{[1]}}{(m,n^{[1]})} \underbr{W^{[2]}}{(n^{[1]},n^{[2]})}  + \underbr{b^{[2]}}{n^{(2)}}$
- $\large \underbr{A^{[2]}}{(m,n^{[2]})} = \sigma(Z^{[2]})$

where 
- $f$ is a ```Relu``` function (the code is provided)
- $\sigma$ is a sigmoid function (the code is provided)

### Backward propagation

The backward propagation can be calculated as

- $\large \underbr{dZ^{[2]}}{(m,n^{[2]})} = \underbr{A^{[2]}}{(m,n^{[2]})} - \underbr{Y}{(m,n^{[2]})}$
- $\large \underbr{dW^{[2]}}{(n^{[1]},n^{[2]})} = \frac{1}{m} {\underbr{A^{[1]}}{(m,n^{[1]})}}^{T} \underbr{dZ^{[2]}}{(m,n^{[2]})} $
- $\large \underbr{db^{[2]}}{(n^{[2]})} = \frac{1}{m} \sum_{i=1}^{m} \underbr{dZ^{[2]}}{(m,n^{[2]})}$

- $\large \underbr{dA^{[1]}}{(m,n^{[1]})} = \underbr{dZ^{[2]}}{(m,n^{[2]})} {\underbr{W^{[2]}}{(n^{[1]},n^{[2]})}}^{T} $
- $\large \underbr{dZ^{[1]}}{(m,n^{[1]})} = \underbr{dA^{[1]}}{(m,n^{[1]})} \: \odot \: f' (\underbr{Z^{[1]}}{(m,n^{[1]})})$
- $\large \underbr{dW^{[1]}}{(n^{[0]},n^{[1]})} = \frac{1}{m} {\underbr{X}{(m,n^{[0]})}}^{T} \underbr{dZ^{[1]}}{(m,n^{[1]})} $
- $\large \underbr{db^{[1]}}{(n^{[1]})} = \frac{1}{m} \sum_{i=1}^{m} \underbr{dZ^{[1]}}{(m,n^{[1]})}$

The $\odot$ operator refers to the point-wise multiplication operation.

### Backward propagation

Based on the previous formulas, write the corresponding backpropagation algorithm.

### Parameters update

- Implement a **first version** in which the parameters are updated using a **simple gradient descent**:
    - $W = W - \alpha dW$


- Implement a **second version** in which the parameters are updated using the **momentum method**:
    - $V_{dW}(t) = \beta V_{dW}(t-1) + (1-\beta) dW$
    - $W(t) = W(t-1) - \alpha V_{dW}(t)$

---
## Testing

For testing your code, you can use the code provided in the last cells (loop over epochs and display of the loss decrease).

You should observe a loss which decreases over epochs and see higher training accuracy.