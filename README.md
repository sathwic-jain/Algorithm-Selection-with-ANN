# Automated Algorithm Selection using Neural Networks

Welcome to our project on Automated Algorithm Selection! This repository explores the fascinating problem of choosing the best algorithm from a collection based on the features of a given instance. We're using artificial neural networks (ANNs) to tackle this challenge.

## What's the Challenge?

The AS problem involves picking the most suitable algorithm from a set of options for a specific task. Here's a breakdown of what we're exploring:

- **Algorithms & Instances**: We've got a bunch of algorithms (let's call them `A = {a1, a2, ..., an}`) and various instances, each with its unique features.
- **Cost Calculation**: For each instance, we've got a way to measure the cost of using a particular algorithm (`c(i, ak)`).
- **Our Goal**: We're after an automated system (`f(v(i))`) that can predict the best algorithm for a given instance while minimizing the overall cost.

## What's Inside?

### Parts Breakdown:

### 1. Regression Task

We'll use ANNs to predict the cost of each algorithm based on instance features. This way, given an instance, we'll choose the algorithm with the lowest predicted cost.

### 2. Multi-class Classification

- **Basic Approach**: Each algorithm is treated as a separate class, and we'll build a classification model using ANNs.
- **Advanced Approach**: Here, we're more interested in minimizing the average cost metric rather than simply focusing on classification accuracy. We'll factor in the "regret" of our predictions, comparing the true best algorithms with our predicted ones.

### 3. Extensions

#### Extension 1: Cost-Sensitive Pairwise Classification

We'll set up binary classifiers for each pair of algorithms. Taking into account the "prediction regret," the algorithm with the most votes across all classifiers will be our selection.

#### Extension 2: Random Forests

We're not limiting ourselves to just neural networks! Here, we'll delve into Random Forest models for handling the AS problem and compare these approaches with our neural network-based solutions.

## How to Use this Repository

### Installation

- Clone this repository.
- Install the required dependencies listed in `requirements.txt`.

### Running the Code

- Visit each part's directory.
- Follow the instructions in the respective READMEs to execute the code for that particular section.

## Structure of the Project

- `Part1/`: Code and instructions for Part 1
- `Part2/`: Code and instructions for Part 2
- `Part3_Extension1/`: Details for Extension 1
- `Part3_Extension2/`: Details for Extension 2

## Results and Discussion

Each part's directory holds detailed reports and discussions on the results. For a deeper understanding of each approach and comparison of results, dive into the respective directories.
