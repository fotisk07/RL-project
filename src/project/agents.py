import numpy as np


class MCAgent:
    def __init__(self, height, width, alpha, gamma, epsilon, rng, epsilon_decay=0.001, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = rng
        self.Q = np.zeros((height, width, 2))
        self.returns = {}
        self.episode = 0

    def step(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.choice([0, 1])
        return int(np.argmax(self.Q[state]))

    def decay_epsilon(self):
        self.episode += 1
        self.epsilon = max(self.min_epsilon, self.epsilon / (1 + self.epsilon_decay * self.episode))

    def learn(self, trajectory):
        G = 0
        visited = set()

        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t][0], trajectory[t][1], trajectory[t][-1]
            G = self.gamma * G + r_t

            key = (s_t, a_t)
            if key not in visited:
                visited.add(key)
                self.returns[key] = self.returns.get(key, 0) + 1
                self.Q[s_t[0], s_t[1], a_t] += self.alpha * (G - self.Q[s_t[0], s_t[1], a_t])

        self.decay_epsilon()


import numpy as np


class SarsaLambdaAgent:
    def __init__(
        self, height, width, alpha, gamma, epsilon, rng, lam=0.9, epsilon_decay=0.001, min_epsilon=0.01
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.lam = lam
        self.rng = rng
        self.episode = 0

        self.Q = np.zeros((height, width, 2))
        self.E = np.zeros((height, width, 2))  # eligibility traces

    def step(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.choice([0, 1])
        return int(np.argmax(self.Q[state]))

    def decay_epsilon(self):
        self.episode += 1
        self.epsilon = max(self.min_epsilon, self.epsilon / (1 + self.epsilon_decay * self.episode))

    def reset_traces(self):
        self.E.fill(0)

    def update(self, state, action, reward, next_state, next_action, done):
        s0, s1 = state
        ns0, ns1 = next_state

        if done:
            td_error = reward - self.Q[s0, s1, action]
        else:
            td_error = reward + self.gamma * self.Q[ns0, ns1, next_action] - self.Q[s0, s1, action]

        # Increment trace for visited pair
        self.E[s0, s1, action] += 1

        # Update all Q values and decay all traces
        self.Q += self.alpha * td_error * self.E
        self.E *= self.gamma * self.lam

    def learn(self, trajectory):
        self.reset_traces()

        for t in range(len(trajectory) - 1):
            state, action, reward = trajectory[t][0], trajectory[t][1], trajectory[t][-1]
            next_state, next_action = trajectory[t + 1][0], trajectory[t + 1][1]
            self.update(state, action, reward, next_state, next_action, done=False)

        # Last step (terminal)
        last = trajectory[-1]
        self.update(last[0], last[1], last[-1], last[0], 0, done=True)

        self.reset_traces()
        self.decay_epsilon()
