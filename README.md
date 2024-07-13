# n-step Advantage Actor-Critic (A2C) for CartPole

This repository contains an implementation of the n-step Advantage Actor-Critic (A2C) algorithm for the CartPole environment. The project explores both discrete and continuous action spaces, and investigates the effects of various hyperparameters on the learning process.

## Project Structure

- `K=1-n=1-disc/`: Contains an animation of the CartPole when K=1, n=1 in the discrete environment.
- `imgs/`: Contains plots used in the report.
- `lists/`: Contains data for each agent to reproduce plots without retraining.
- `CS_456_MP2_A2C.pdf`: The project report detailing methodology and results.
- `MP2_A2C.pdf`: The project handout with specifications.
- `train.py`: Implementation of the A2C algorithm and supporting functions.
- `Solution.ipynb`: Jupyter notebook to run the A2C algorithm and generate plots.

## Features

- Implementation of n-step A2C for both discrete and continuous action spaces
- Support for multiple workers (K) and n-step returns
- Evaluation and logging functionalities
- Visualization of training progress, value functions, and agent performance

## Getting Started

1. Clone this repository
2. Install the required dependencies, see `requirements.txt`.
3. Run cells in `Solution.ipynb` to train the agent or reproduce plots using pre-saved data

## Results

The project explores various configurations of the A2C algorithm, including:
- Basic A2C version in CartPole
- Stochastic rewards
- Multiple workers (K-workers)
- n-step returns
- K × n batch learning

Detailed results and analysis can be found in `CS_456_MP2_A2C.pdf`.

## Usage

To train a new agent or reproduce results:

1. Open `Solution.ipynb`
2. Adjust hyperparameters as needed
3. Run the cells to train the agent or generate plots from pre-saved data


## Acknowledgments

This project was completed as part of the EPFL Artificial Neural Networks and Reinforcement Learning course, in collaboration with [@eliashornberg](https://github.com/eliashornberg).
