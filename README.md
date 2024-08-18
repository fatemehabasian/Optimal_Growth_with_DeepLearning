### Importance and Novelty of Deep Learning in Dynamic Programming Problems

Dynamic programming is a critical tool for solving optimization problems in economics, particularly in the context of models with time-dependent decisions. However, traditional approaches to dynamic programming are often plagued by the "curse of dimensionality," where the computational complexity grows exponentially with the number of state variables. This limitation makes it challenging to apply dynamic programming to high-dimensional problems, which are increasingly common in modern economic modeling.

Deep learning offers a novel solution to this problem. By leveraging the universal approximation capabilities of deep neural networks, researchers can efficiently approximate value and policy functions even in high-dimensional spaces. This approach significantly reduces the computational burden associated with traditional methods and provides a flexible framework for solving complex dynamic programming problems that were previously infeasible. The continuous-time formulation and the ability to generate and utilize large amounts of synthetic data further enhance the effectiveness of deep learning in this domain.

---

# Dynamic Programming Problem Solved with Deep Learning

## Overview

This repository contains the implementation of a project focused on solving dynamic programming problems using deep learning techniques. The project is part of a Quantitative Economics course at Sharif University of Technology and aims to explore how neural networks can be applied to find policy functions in dynamic programming models.

## Project Structure

- **`DL_Project.html`**: This HTML file contains the detailed report and code used to solve the dynamic programming problem using deep learning. It includes explanations of the methodologies, neural network architecture, and results.
- **`DP_Problem_Project.pdf`**: The PDF document outlines the problem statement, the theoretical background, and the instructions for implementing the solution using both traditional dynamic programming methods and deep learning techniques.

## Project Description

### Problem Statement

The project addresses a representative agent's problem in a dynamic setting:

$$\max_{x_t} \sum_{t=0}^{\infty} \beta^t F(x_t, x_{t+1})$$

subject to the constraint:

$$x_{t+1} \in \Gamma(x_t), \forall t \geq 0$$

This problem can be reformulated using the Bellman equation as:

$$V(x) = \max_{y \in \gamma(x)} \left\{ F(x, y) + \beta V(y) \right\}$$

The goal is to solve this problem by finding the policy function $g(x)$ that maximizes the agent's utility over time.

### Neural Network Approach

A neural network is employed to approximate the policy function $g(x)$ by minimizing the residuals of the Euler equation:

$$\sum_{i=1}^{N} \epsilon_i^2 = \sum_{i=1}^{N} \left[ F_y(x_i, g(x_i)) + \beta F_x(g(x_i), g(g(x_i))) \right]^2$$

The neural network is trained using samples of $\x_i$ to approximate the policy function.

### Case Study: Neoclassical Growth Model

The project applies the above method to solve a neoclassical growth model, where the value function is defined as:

$$V(k) = \max_{k'} \left\{ U\left(f(k) + (1-\delta)k - k'\right) + \beta V(k') \right\}$$

This involves defining the utility function $U(\cdot)$, production function $f(k)$, depreciation rate $\delta$, and discount factor $\beta$. The project compares the deep learning approach with traditional methods like value function iteration, policy function iteration, and Euler equation iteration.

### Results

The project showcases the effectiveness of neural networks in solving dynamic programming problems and compares the results with those obtained using traditional methods. The neural network approach demonstrates a high degree of accuracy in approximating the policy function, providing a novel method for tackling such economic models.

## How to Use

1. **Viewing the Report**:
   - Open the `DL_Project.html` file in a web browser to view the full report, including the code and results.

2. **Understanding the Problem**:
   - Refer to the `DP_Problem_Project.pdf` for a detailed explanation of the problem statement, theoretical background, and implementation instructions.

## Conclusion

This project demonstrates the potential of deep learning techniques in solving complex dynamic programming problems, providing a modern alternative to traditional methods. The results validate the neural network's capability to approximate policy functions accurately, making it a valuable tool for economic modeling and analysis.

