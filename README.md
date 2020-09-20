## RL agents with keras and TF for openAI gym

This repo contains toy solutions for the openAI gym environment implementing Q-networks in Keras and TensorFlow.

## Requirements

*   tensorflow==1.0.0
*   tensorflow-gpu==1.0.0
*   Keras==1.2.2
*   numpy==1.12.0
*   matplotlib==2.0.0
*   gym==0.8.0

## How To

To train a dueling network on the cart-pole with succesfull configuration:

```sh
python q_learn.py CartPole-v0 \
    --render 50 \
    --batch_size 100 \
    --hidden_size 150 \
    --replay_size 100 \
    --train_repeat 10 \
    --gamma 0.99 \
    --lr 1e-3 \
    --epsilon 0.1 \
    --exploration_decay 1e-5 \
    --max_episodes 200 \
    --nn_mode max \
    --model_path cart-pole-dueling-max
```

To train a dueling network on the MountainCar-v0 environment, a succesfull configuration could be:

```sh
python q_learn.py MountainCar-v0 \
    --render -180 \
    --batch_size 10 \
    --hidden_size 10 \
    --replay_size 10000 \
    --train_repeat 4 \
    --gamma 0.95 \
    --lr 1e-3 \
    --epsilon 0.15 \
    --exploration_decay 0.01 \
    --max_episodes 1000 \
    --nn_mode max \
    --model_path mountain-car-dueling-max
```
