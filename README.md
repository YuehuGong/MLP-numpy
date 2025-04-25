# MLP-numpy
A course project.

## 1. Introduction
In this Project, I tried all methods mentioned in the introduction except CNN.Despite these methods, I also implement a neural ode layer.The following sections will explain the design of neural ode layer, experiments of compositions of different methods and conclusion.

## 2. Neural ode layer
### 2.1. ODE problem
$$
\begin{cases}
\frac{dy}{dx} = f(y(x), x, \theta) \\
y(x_0) = y_0
\end{cases}\\
y(x_0 + \Delta x) = y_0 + \int_{x_0}^{x_0+\Delta x}f(x,y(x))dx
$$

### 2.2. Adjoint Method
- Abstact:
$$
\begin{align}
\text{Problem Setting: } \frac{dh(t)}{dt} &= f(h(t), t;\theta), \quad h(0) = h_0 \\
\text{Loss Function: } L &= L(h(T)) = L(h_0 + \int_{0}^1f(h(t), t;\theta)dt) \\
\text{Adjoint State: } a(t) &= \frac{\partial L}{\partial h(t)} \\
\text{Adjoint Equation: } \frac{da(t)}{dt} &= -a(t)^T \frac{\partial f_\theta}{\partial h} \\
\text{Parameter Gradient: } \frac{dL}{d\theta} &= -\int_{T}^{0} a(t)^T \frac{\partial f_\theta}{\partial \theta} dt
\end{align}
$$


### 2.3. Forward
ResNet block (discrete):
$$
h_{t+1} = h_t + f(h_t, \theta_t)
$$

Neural ODE (continuous):
$$
\frac{dh(t)}{dt} = \text{NN}(h(t), t; \theta) \\
h(T) = h(0) + \int_0^T \text{NN}(h(t), t; \theta) dt
$$

### 2.4. Backward (Adjoint Method)

In the backward pass, the adjoint method is used to efficiently compute gradients, avoiding the need to explicitly solve the ODE at each time step.

- Adjoint State:The adjoint state $a(t)$ is the partial derivative of the loss function with respect to $h(t)$:
$$
a(t) = \frac{\partial L}{\partial h(t)}
$$

- \textbf{Adjoint Equation:} The adjoint state propagates backward in time and satisfies:
  $$
  \frac{da(t)}{dt} = -a(t)^T \frac{\partial f_\theta}{\partial h}
  $$

- \textbf{Parameter Gradient:} The gradient of the loss with respect to parameters is computed using the adjoint state and the network output:
  $$
  \frac{dL}{d\theta} = -\int_{T}^{0} a(t)^T \frac{\partial f_\theta}{\partial \theta} dt
  $$

This method enables efficient computation of the gradient of $\theta$ for parameter updates.

## 3. Experiments
### 3.1. Design of experiments
- Model
  - MLP
  - Deeper but narrower MLP
  - Neural ODE
- Optimizer
  - SGD
  - Moment
- scheduler
  - StepLR
  - MultiStep
  - Exponential
- Regulation
  - Xavier initialization
  - L2 regulation
  - dropout

Among those mehtods, scheduler&optimizer,xavier initialization only influenced speed of training, but design of model, L2 regulation and dropout influenced the behavior of inference.So I the experiments I designed are as follows:

| **Aspect**   | **Model**  | **Optimizer** | **Scheduler** | **Regularization** |
|--------------|------------|---------------|---------------|--------------------|
| **Speed**    | MLP        | SGD           | x             | None               | 
| **Speed**    | MLP        | x             | Step          | None               | 
| **Behavior** | MLP        | SGD           | Step          | x                  | 
| **Behavior** | x          | SGD           | Step          | None               | 

### 3.2. Metrics
two metrics:speed and bahavior(accuracy)
The metric of inference behavior is accuracy, and I define the efficiency of optimizers or schedulers as Loss Decrease Rate:

We assume speed of convergence is $\text{loss}(t) = B + A \cdot t^{-\beta} + \epsilon_t$,where $\beta$ denotes the speed of convergence.Given recordings of training$[(t_i,loss_i),1\leq i\leq T]$,$\hat{\beta} = -\frac{\text{Cov}(\ln t, \ln(\text{loss} - \hat{B}))}{\text{Var}(\ln t)}$

- $B$ = asymptotic loss floor  
- $A$ = scale factor  
- $\beta$ = convergence rate (to be estimated)  
- $\epsilon_t$ = observation noise  
 

### 3.3. Results
#### 3.3.1. Speed
| **Optimizer** | **Scheduler** | **Loss Decrease Rate** |
|---------------|---------------|-------------------------|
| SGD           | StepLR        |          0.88232          |
| SGD           | MultiStep     |        0.78060       |
| SGD           | Exponential   |              (Accuracy is very low)           |
| Moment(0.9)        | StepLR        |          0.87054         |
| Moment(0.95)        | StepLR        |          0.81915         |
---

#### 3.3.2. Behavior

| **Model** | **Regularization** | **Loss Decrease Rate** |
|-----------|--------------------|-------------------------|
| MLP       | None               |       0.95090              |
| MLP       | L2               |            0.95           |
| MLP       | dropout               |        0.95290            |
| MLP       | L2+dropout           |          0.9503         |
| deeper and narrower MLP| L2               |          0.9617            |
| NeuralODE       | L2               |          0.94860            |
---



