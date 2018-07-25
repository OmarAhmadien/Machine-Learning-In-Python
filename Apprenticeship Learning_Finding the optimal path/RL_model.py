import operator
__author__ = 'jarethmoyo'

# First we create a state class that holds the current state and value of a node


class State(object):
    def __init__(self, node):
        self.node=node
        self.children=[]

    def get_possible_states(self,terminal_states):
        # deal with terminal states
        if self.node in terminal_states:
            self.children.extend([self.node])
            return
        pos_states=[(self.node[0]+1, self.node[1]), (self.node[0]-1, self.node[1]),
                    (self.node[0], self.node[1]+1), (self.node[0],self.node[1]-1)]
        pos_states=filter(lambda x: 0<=x[0]<=7 and 0<=x[1]<=7, pos_states)  # remove values out of grid
        self.children.extend(pos_states)

# 0:North, 1:East, 2:South, 3:West,


class Grid(object):
    def __init__(self,rows,columns, term_states_vals,terminals):
        self.state_values=dict()  # values for every state
        self.policy=dict()  # optimal policy for a given reward
        self.actions=[0,1,2,3]  # 0:North, 1:East, 2:South, 3:West
        self.transition_dict=dict()
        self.terminal_states=term_states_vals # reward values for terminal states
        self.all_states=[State((i,j)) for i in range(rows) for j in range(columns)]
        for s in self.all_states:
            s.get_possible_states(terminals)
        self.all_states_vals= map(lambda x: x.node,self.all_states)
        for s in self.all_states_vals:
            self.state_values[s]=0
        self.create_transition_model()

    def create_transition_model(self):
        """This method basically creates our transition model by assigning probabilities to every possible
           action """
        for a in self.actions:
            nxt, prev = a+1, a-1
            if nxt>=len(self.actions):  # we have gone beyond len of list
                nxt=self.actions[0]  # go back to the beginning
            elif prev<0:
                prev=self.actions[-1]  # go to the end
            self.transition_dict[a]={a: 0.7, nxt: 0.15, prev: 0.15}
            for i in range(4):
                if i not in self.transition_dict[a]:
                    self.transition_dict[a][i]=0

    def compute_q_value(self, state,children, action, r):
        """This method is for computing the q value of a given state as well as a given action"""
        q=0
        # if terminal state, return its value
        prob=self.transition_dict[action]
        if state in self.terminal_states:
            return self.terminal_states[state]
        state_probs=self.compute_state_probs(state, prob)
        for child in children:
            q+=state_probs[child]*(r+self.state_values[child])
        return q

    def compute_v_values(self,r):
        """Given a reward this method computes the values and policy of different states
           It incorporates value iteration hence the reason for the AS^2 complexity"""
        current=self.state_values.values()
        valid=False
        while not valid:
            previous=current[:]
            for state in self.all_states:
                qs=[]
                for action in self.actions:
                    qs.append((action,self.compute_q_value(state.node,state.children, action, r)))
                b_move=max(qs, key=lambda x:x[1])
                self.state_values[state.node]=b_move[1]
                self.policy[state.node]=b_move[0]
            current=self.state_values.values()
            diff=map(operator.sub, previous, current)
            valid=all(x==0 for x in diff)  # value iteration has converged if True
        return self.state_values

    def compute_state_probs(self, state, probs):
        # assign a probability to every possible state
        fin_probs=dict()
        for dr in probs:
            if dr == 0:
                ns=(state[0]+1, state[1])
            elif dr == 1:
                ns=(state[0], state[1]+1)
            elif dr == 2:
                ns=(state[0]-1, state[1])
            else:
                ns=(state[0], state[1]-1)

            if ns not in self.all_states_vals:
                ns=state

            if ns in fin_probs:
                fin_probs[ns]+=probs[dr]  # incr probability of staying in the same state
            else:
                fin_probs[ns]=probs[dr]
        return fin_probs

    def convert_policy_to_path(self,policy, goals, start=(6,1)):
        path=[start]
        next_state = self.get_next_state(policy, start)
        while next_state not in goals:
            if next_state in path:
                rep_state = path[-1]
                path.extend([next_state, rep_state]*10)
                break
            path.append(next_state)
            next_state = self.get_next_state(policy, next_state)
        path.append(next_state)
        return path

    def get_next_state(self, policy, c_state):
        # given a policy and a state, determine the next state
        dr = policy[c_state]
        if dr == 0:
            ns = (c_state[0]+1, c_state[1])
        elif dr == 1:
            ns = (c_state[0], c_state[1]+1)
        elif dr == 2:
            ns = (c_state[0]-1, c_state[1])
        else:
            ns = (c_state[0], c_state[1]-1)
        return ns


