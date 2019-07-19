# cartpole_RL

This is a classic problem related to reinforcement learning. Through learning the policy of the motion of the cartpole, we try to maintain the pole straight as many steps as possible.

This project is similar to the ones that use OpenAI ```Gym``` but the environment is based on Robot Operating System. There is a robot which is in the ```cartpole_robot.py``` for collecting training data. In ```learn_dqn.py```, the policy is learned. ```executive.py``` is a script that will test the model. 

MDP: 
The state of the cartpole is \[x, theta, x', theta'\], where x is the position of the robot and theta is the pole angle. The primed are the corresponding velocity of the cart and pole. The limit for x is \[-1.2, 1.2\] meters, for theta is \[-3, 3\] degrees. 

To run the robot:
```
roslanch robot_sim project3.launch
```

Then run the training script:
```
rosun robot_sim learn_dqn.py
```

After the  training process:
```
rosun robot_sim executive.py
```
to check the model.

A GUI is also provided to visulize the model. Before running ```executive.py```, run 
```
rosrun robot_sim cartpole_gui.py
```

![cartpole system](https://github.com/zyx124/cartpole_RL/blob/master/cartpole_system.png)

