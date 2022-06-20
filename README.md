# **Autonomous Robotic System**
This coursework is about implementing heuristic and reinforcement learning algorithms to solve different environment problems, namely CartPole and MountainCar problem.

In CartPole problem, the goal is to keep the cartpole balanced by applying appropriate forces to a pivot point. ***cartpole_DQN*** solves the problem using Deep Q Network while ***cartpole_HL*** uses Hill Climbing method to balance the pole.

In MountainCar problem, the goal is to drive an under-powered car up to the top of the mountain. ***mountaincar_HC*** solves the problem using Hill Climbing with deterministic policy while ***mountaincar_QL*** uses Q-Learning.

## **Environment**
- Windows 10 64-bit
- Python 3.9.7
- Visual Studio Code

## **Packages required**
- gym
- numpy
- matplotlib
- tensorflow

> If you do not have it installed, type the following command in the terminal to download the packages.
>```
>pip install gym
>```
>
>```
>pip install numpy
>```
>
>```
>pip install matplotlib
>```
>
>```
>pip install tensorflow
>```
>

Then, feel free to run the files..

To visualize the simulation, uncomment the `self.env.render()` line.

A pretrained weights file, cartpole_DQN(BEST).h5 is available to be used. To use it, uncomment the `self.load('cartpole_DQN(BEST).h5')` line in the `test()` function of DQN file. 


