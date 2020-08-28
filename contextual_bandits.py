import numpy as np
from numpy import random, linalg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


arm_dimension = 2
hiddenSize = 3
outputSize = 1

num_rounds = 5
num_arms_per_round = 4

#batch_size = 1000
num_steps = 1
lamdba = 1
gamma = 1



total_params = arm_dimension * hiddenSize + hiddenSize*outputSize
#need to fix this
#U = lamdba * torch.ones((total_params,))
U = lamdba * torch.eye(total_params)

def random_ball(num_points, radius=1):
	# First generate random directions by normalizing the length of a
	# vector of random-normal values (these distribute evenly on ball).
	random_directions = random.normal(size=(arm_dimension,num_points))
	random_directions /= linalg.norm(random_directions, axis=0)
	# Second generate a random radius with probability proportional to
	# the surface area of a ball with a given radius.
	random_radii = random.random(num_points) ** (1/arm_dimension)
	# Return the list of random (direction & length) points.
	return torch.tensor(radius * (random_directions * random_radii).T, dtype=torch.float32)


def createAMatrix():
	return torch.tensor(random.normal(size=(arm_dimension, arm_dimension)), dtype=torch.float32)

def getReward(arm_vec, A_matrix):
	inner= torch.matmul(A_matrix.transpose(0,1), A_matrix)
	next_inner = torch.matmul(arm_vec, inner)
	arm_vector_transpose = torch.reshape(arm_vec, (arm_dimension,1))
	last = torch.matmul(next_inner, arm_vector_transpose)
	noise = random.normal()
	last = last + noise

	return last

###make the neural network


class NeuralUCBModel(nn.Module):
	def __init__(self):
		super(NeuralUCBModel, self).__init__()
		# parameters
		# TODO: parameters can be parameterized instead of declaring them here

		self.l1 = nn.Linear(arm_dimension, hiddenSize, bias=False)
		self.l2 = nn.Linear(hiddenSize, outputSize, bias = False)
		
	def forward(self, X):
		z = self.l1(X)
		z2 = torch.nn.functional.relu(z)
		z3 = self.l2(z2)
		mult_tensor = torch.tensor(np.sqrt(hiddenSize))
		mult_tensor.reshape(1,1)
		o = torch.mul(mult_tensor, z3)
		return o


class banditData(Dataset):
    def __init__(self):
        self.inp_data = torch.empty(1, arm_dimension)
        self.out_data = torch.empty(1,1)
        #self.out_data = torch.FloatTensor(get_categorical(self.out_data).astype('float'))
    def __len__(self):
        return len(self.inp_data)

    def __getitem__(self, index):
        #target = self.out_data[ind]
        #data_val = self.data[index] [:-1]
        return self.inp_data[index],self.out_data[index]

def regularized_mse_loss(input, target):
	return ((input - target) ** 2).sum()

def train(t):
	model = NeuralUCBModel()
	model.load_state_dict(torch.load("model" + str(t)), strict=False)
	model.train()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=lamdba)
	#batch_size_clipped = min(batch_size, len(data))
	for i in range(num_steps):
		#indices = random.choice(np.arange(0, len(data)), batch_size_clipped, replace = False)
		#train_data = Subset(data, indices)
		optimizer.zero_grad()

		#input = model(train_data[:][0])
		input = model(data.inp_data)
		# loss = nn.MSELoss()
		# output = loss(input, train_data[:][1])
		#output = regularized_mse_loss(input, train_data[:][1])
		output = regularized_mse_loss(input, data.out_data)
		output.backward()
		optimizer.step()

	torch.save(model.state_dict(), 'model'+str(t+1))
	#

def getValueAndGradient(arm_vec, t):
	model = NeuralUCBModel()
	model.load_state_dict(torch.load("model" + str(t)), strict=False)
	output = model(arm_vec)
	output.backward()
	g = torch.cat([p.grad.flatten().detach() for p in model.parameters()])

	return output, g

def chooseArm(arms, t):
	global U

	ucb_values = []
	g_list = []

	for a in arms:
		value, g = getValueAndGradient(a, t)
		g_row = torch.reshape(g, (1, total_params))
		inner = torch.matmul(g_row, torch.inverse(U))
		g_col= torch.reshape(g, (total_params,1))
		last = torch.matmul(inner, g_col)
		#sigma2 = lamdba * gamma * g * g / U
		sigma2 = lamdba * gamma * last
		sigma = torch.sqrt(torch.sum(sigma2))
		ucb_values.append(value + sigma)

		g_list.append(g_col)

	chosen_index = np.argmax(ucb_values)
	g_row = torch.reshape(g_list[chosen_index], (1, total_params))
	U += torch.matmul(g_list[chosen_index], g_row)
	#U += g_list[chosen_index] * g_list[chosen_index]


	return chosen_index



# def selectArm():

###############

arms_chosen=torch.empty(1, arm_dimension)
rewards_observed = torch.empty(1,1)

#arms_chosen = torch.zeros([1, arm_dimension])
#rewards_observed = torch.zeros([1, 1])
data = banditData()

per_period_regret = []
cum_regret = []

def NeuralUCB():
	global data
	A_matrix = createAMatrix()
	model = NeuralUCBModel()
	torch.save(model.state_dict(), 'model'+str(0))

	
	for i in range(num_rounds):
		#model.load_state_dict(torch.load("model" + str(i)), strict=False)

		arms = random_ball(num_arms_per_round)

		#values = model(arms)

		#put ucb code here
		chosen_index = chooseArm(arms, i)

		reward_vec = []
		for a in arms:
			ar = torch.reshape(a, (1,arm_dimension))
			ar_reward = getReward(ar, A_matrix)
			reward_vec.append(ar_reward)
		best_reward = np.max(reward_vec)
		reward = reward_vec[chosen_index]
		regret = (best_reward - reward)[0][0]
		per_period_regret.append(regret)
		if i == 0:
			cum_regret.append(regret)
		else:
			cum_regret.append(regret + cum_regret[i-1])

		chosen_arm = arms[chosen_index]
		
		chosen_arm = torch.reshape(chosen_arm, (1, arm_dimension))
		if (i == 0):
			data.inp_data[0] = chosen_arm
			data.out_data[0] = reward
		else:
			data.inp_data = torch.cat([data.inp_data, chosen_arm], dim=0)
			data.out_data = torch.cat([data.out_data, reward], dim=0)

		train(i)

		if (i % 100 == 0):
			plt.plot(cum_regret)
			plt.savefig('regret_' + str(i)+ '.png')



			# model = NeuralUCBModel()
	# params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
 #    model.load_state_dict(params['state_dict'])
 #    #fix this
 #    outputs = model(arms)
 #    ucb = 

NeuralUCB()
print(per_period_regret)
print(cum_regret)




