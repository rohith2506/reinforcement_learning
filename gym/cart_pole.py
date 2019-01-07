import gym
import numpy as np
from flask import Flask
import json

env = gym.make('CartPole-v1')

def play(env, policy):
    observation = env.reset()
    done, score, observations = False, 0, []

    for _ in range(5000):
        observations += [observation.tolist()]
        if done: break
        outcome = np.dot(policy, observation)
        action = 1 if outcome < 0 else 0
        observation, reward, done, info = env.step(action)
        score += reward

    return score, observations


max = (0, [], [])

for _ in range(100):
    policy = np.random.rand(1, 4) - 0.5
    score, observations = play(env, policy)

    if score > max[0]:
        max = (score, observations, policy)


app = Flask(__name__, static_folder='.')

@app.route("/data")
def data():
    return json.dumps(max[1])

@app.route("/")
def root():
    return app.send_static_file("./index.html")

app.run(host='0.0.0.0', port='3000')
