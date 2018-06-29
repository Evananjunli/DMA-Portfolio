#importing necessary modules
import gym
import gym.spaces
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#sorting method for lists
def bubble_sort(x):
    changed = True
    while changed:
        changed = False
        for i in range(len(x) - 1):
            if x[i] > x[i+1]:
                x[i], x[i+1] = x[i+1], x[i]
                changed = True
#maximum method
def maximum(x):
    bubble_sort(x)
    return x[len(x)-1]

#median method
def median(x):
    bubble_sort(x)
    if len(x) % 2 == 1:
        return x[(len(x)+1)/2]
    else:
        return (x[(len(x))/2]+x[(len(x)/2)+1])/2

#initializing variables
LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 200
score_requirement = 50
#method to calculate mean of a list

def mean(x):
    total = 0
    for count in range(0,len(x)):
        total += x[count]
    return total/len(x)


#method to run random games
def random_games(games):
    for episode in range(games):
        env.reset()
        for t in range(200):
            #env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

#run 5 example random games
#random_games(5)

#method to move randomly left or right
#finds what random moves give a score of at least 50
def initial_set(sample_size):
    training = []
    scores = []
    accepted = []
    for _ in range(sample_size):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randint(0, 1)
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training.append([data[0], output])
        env.reset()
        scores.append(score)
    training_save = np.array(training)
    np.save('saved.npy', training_save)

    #displays results to screen
    print('Average accepted score:', mean(accepted))
    print('Median accepted score:', median(accepted))
    print('Highest score:', maximum(accepted))
    print('Number of accepted scores:', len(accepted))

    #returns data as a list to run in a neural network model
    return training

#test the data collection with 10000 runs
#initial_set(10000)

#connecting to neural network to create a model
def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

#training the model
def train_model(training, model=False):
    X = np.array([i[0] for i in training]).reshape(-1, len(training[0][0]), 1)
    y = [i[1] for i in training]

    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=100, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

training = initial_set(10000)
model = train_model(training)


#running the game through the trained model
def final_runs(x):
    scores = []
    choices = []
    for each_game in range(x):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()
            if len(prev_obs) == 0:
                action = random.randrange(0, 2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done: break
        scores.append(score)
    print('Average Score:', sum(scores) / len(scores))
    print('Highest Score:', maximum(scores))

final_runs(10)