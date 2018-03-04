# 10-703 Homework 2 DQN Implementation

## Linear Q-network without experience replay

```
python3 DQN_Implementation.py --env CartPole-v0 --render 0 --net_type linear --burn_in 0 --memory_size 1 --lr 0.0001 --gamma 0.99 --e_start 0.5 --e_end 0.05 --e_decay 5e-6 --n_episode 2000 --test_interval 2000 --video_dir ./video/Q1/cp

python3 DQN_Implementation.py --env MountainCar-v0 --render 0 --net_type linear --burn_in 0 --memory_size 1 --lr 0.0001 --gamma 1 --e_start 0.5 --e_end 0.05 --e_decay 5e-6 --n_episode 3000 --test_interval 20000 --video_dir ./video/Q1/mc

```

## Linear Q-network with experience replay

```
python3 DQN_Implementation.py --env CartPole-v0 --render 0 --net_type linear --burn_in 100 --memory_size 2000 --lr 0.0001 --gamma 0.9 --e_start 0.5 --e_end 0.05 --e_decay 5e-6 --n_episode 2000 --test_interval 2000 --video_dir ./video/Q2/cp

python3 DQN_Implementation.py --env MountainCar-v0 --render 0 --net_type linear --burn_in 100 --memory_size 100 --lr 0.0001 --gamma 1 --e_start 0.5 --e_end 0.05 --e_decay 5e-6 --n_episode 3000 --test_interval 20000

```


## DQN with experience replay

```
python3 DQN_Implementation.py --env CartPole-v0 --render 0 --net_type DQN --burn_in 10000 --memory_size 50000 --lr 0.0001 --gamma 0.99 --e_start 0.5 --e_end 0.05 --e_decay 5e-6 --n_episode 2000 --test_interval 2000 --video_dir ./video/Q3/cp

python3 DQN_Implementation.py --env MountainCar-v0 --render 0 --net_type DQN --burn_in 10000 --memory_size 50000 --lr 0.0005 --gamma 1 --e_start 0.5 --e_end 0.05 --e_decay 5e-6 --n_episode 3000 --test_interval 20000 --video_dir ./video/Q3/mc

```

## Dueling network with experience replay

```
python3 DQN_Implementation.py --env CartPole-v0 --render 0 --net_type Dueling --burn_in 10000 --memory_size 50000 --lr 0.0001 --gamma 0.99 --e_start 0.5 --e_end 0.05 --e_decay 5e-6 --n_episode 2000 --test_interval 2000 --video_dir ./video/Q4/cp

python3 DQN_Implementation.py --env MountainCar-v0 --render 0 --net_type Dueling --burn_in 10000 --memory_size 50000 --lr 0.0005 --gamma 1 --e_start 0.5 --e_end 0.05 --e_decay 5e-6 --n_episode 3000 --test_interval 20000 --video_dir ./video/Q4/mc

```

work

```
python3 DQN_Implementation.py --env CartPole-v0 --render 0 --net_type DQN --burn_in 200 --memory_size 2000 --lr 0.0001 --gamma 0.9 --e_start 0.5 --e_end 0.05 --e_decay 1e-5 --n_episode 1000 --test_interval 2000

python3 DQN_Implementation.py --env MountainCar-v0 --render 0 --net_type DQN --burn_in 200 --memory_size 2000 --lr 0.001 --gamma 0.9 --e_start 0.5 --e_end 0.05 --e_decay 1e-5 --n_episode 3000 --test_interval 20000

```
python3 DQN_Implementation.py --env CartPole-v0 --render 0 --net_type DQN --burn_in 200 --memory_size 2000 --lr 0.0001 --gamma 0.9 --e_start 0.5 --e_end 0.05 --e_decay 1e-5 --n_episode 800 --test_interval 2000 --video_dir ./video/Q1

python3 DQN_Implementation.py --env MountainCar-v0 --render 0 --net_type linear --burn_in 0 --memory_size 1 --lr 0.01 --gamma 1 --e_start 0.1 --e_end 0.1 --e_decay 5e-6 --n_episode 3000 --test_interval 100000