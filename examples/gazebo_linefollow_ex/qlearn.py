import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''

        with open(filename, 'rb') as file:
            self.q = pickle.load(file)

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        with open(filename, 'wb') as file:
             pickle.dump(self.q, file)
        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''

        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ  = min(q)
            mag = max(abs(minQ), abs(maxQ))

            q = [q[i] + random.random() * mag - 0.5 * mag
                 for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        
        if return_q:
            return action, q
        return action
            

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        maxQ2 = max([self.getQ(state2, a) for a in self.actions])

        oldValue = self.q.get((state1, action1), None)

        if oldValue is None:
            self.q[(state1, action1)] = reward
        else:
            self.q[(state1, action1)] = oldValue + self.alpha * (self.gamma * maxQ2 +  reward - oldValue)

