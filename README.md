RL agents with keras and TF for openAI gym 
------------------------------------------
This repo contains toy solutions for the openAI gym environment implementing Q-networks in Keras and TensorFlow.


Requirements
------------
tensorflow==1.0.0
tensorflow-gpu==1.0.0
Keras==1.2.2
numpy==1.12.0
matplotlib==2.0.0
gym==0.8.0


How To
------

To train a dueling network on the cart-pole environment and render when the average reward is above 50:
```sh
python q_learn.py CartPole-v0 --render 50
```


