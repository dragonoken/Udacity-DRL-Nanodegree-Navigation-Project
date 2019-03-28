# Navigation

This project was done as a part of [Deep Reinforcement Learning Udacity Nanodegree Program](https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Environment Description

<table class="unchanged rich-diff-level-one">
  <thead><tr>
      <th align="left">Agent View</th>
      <th align="left">Top View</th>
  </tr></thead>
  <tbody>
    <tr>
      <td align="left"><img src="https://github.com/dragonoken/Udacity-DRL-Nanodegree-Navigation-Project/blob/master/images/banana_collector_firstview.gif" alt="first-view" style="max-width:100%;"></td>
      <td align="left"><img src="https://github.com/dragonoken/Udacity-DRL-Nanodegree-Navigation-Project/blob/master/images/banana_collector_topview.gif" alt="top-view" style="max-width:100%;"></td>
    </tr>
  </tbody>
</table>

<sub>Although there are multiple agents present in the image above on the right, only a single agent is used to solve this environment.</sub>

For this project, an agent to learn to navigate in a large, square world while collecting (giant, delicious) bananas!

In this environment, yellow bananas and blue bananas are constantly generated, and the agent must collect as many yellow bananas as possible while avoiding blue bananas.

Everytime the agent collects a yellow banana, it receives a reward of `+1`.\
Everytime the agent collects a blue banana, it receives a reward of `-1`.

At each time step, the agent receives an observation of its nearby environment: namely, its local ray-cast perception on nearby object as a `37` dimensional vector (state space with size `37`).

the agent must make one of `4` discrete actions:
* `0` - move forward.
* `1` - move backward.
* `2` - turn left.
* `3` - turn right.

The task is episodic, and in order to solve the environment for this project, the agent must get an average score of `+13` over 100 consecutive episodes, or trials, each consisting of 300 time steps.

## Instructions

### 1. Setting up a Python environment

* Assuming that you have already installed either [Miniconda or Anaconda](https://www.anaconda.com/distribution/):\
please, follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment.

* You might need to install PyTorch separately [here](https://pytorch.org/get-started/locally/).

* Also, if you haven't yet, clone this repository and save it wherever you want!

### 2. Download the Unity Environment

You do __Not__ need to install Unity to run this code—the Udacity team has already built the environment for this project, and you can download it from one of the links below. You need only select the environment that matches your operating system (the download will start right after you click the link):

* Linux \[46.3 MB]: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX \[22.7 MB]: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit) \[17.5 MB]: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit) \[20.0 MB]: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Then, extract the content folder in the repository.

Please, do not rename the folder or put multiple files for different OS as my code will try to find the folder automatically using the exact name matching in the same directory—unless you want to specify the file path directly in the notebook.\
When automatic search is used, if there are 32-bit version and 64-bit version available, it will use the 64-bit version by default. If you'd like to use the 32-bit version, you can do also by specifying the file path manually.

### 3. Run the Notebook!

By now, you should be all set to run my notebook file! Run the code and have fun!

* The first section demonstrates the enviroment with an agent performing random moves.
* The second section includes training codes for training a smart agent. (This will take some time.)
* The last section loads a pre-trained smart agent and shows its performance!
