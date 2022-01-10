from utils import parse
import gym
import gym_sokoban
import argparse
from mcts import MCTS
from time import time, sleep
from pathlib import Path
from Node import Node
import operator
import random
from tqdm import tqdm

LEGAL_ACTIONS = [1,2,3,4]

ACTION_MAP = {
	'push up': "U",
	'push down': "D",
	'push left': "L",
	'push right': "R",
}

CHANGE_COORDINATES = {
	0: (-1, 0),
	1: (1, 0),
	2: (0, -1),
	3: (0, 1)
}

class Q_solver:
	"""docstring for Q_solver"""
	def __init__(self, env,actions,depth=200, epochs=100 ):
		self.env = env
		self.actions = actions
		self.penalty_for_step = env.penalty_for_step
		self.reward_finished = env.reward_finished + env.reward_box_on_target
		self.room_fixed = env.room_fixed
		self.depth = depth
		self.epochs = epochs
		self.lr = 0.01
		self.discount=0.99
		self.epsilon = 0.1
		self.qvalues={}
		self.env.reset()
		self.train =1


	def get_state_string(self, env_state):
		state_string = ''.join([''.join([str(element) for element in row]) for row in env_state[3]])
		return state_string
	def uct(self, state):
        best_child = max(tree.children, key = lambda child: ((child.utility / child.rollouts) + (sqrt(2) * log(tree.rollouts) / child.rollouts)))
        return best_child
    def back_propagate(self, history, reward):
        while node is not None:
            node.utility += result
            node.rollouts += 1
            result += self.penalty_for_step
            node = node.parent
	def qtrain(self,  env_state):
		count =0
		for _ in tqdm(range(self.epochs)):
			state = env_state
			for _ in range(self.depth):
				state_str = self.get_state_string(state)
				actions = self.sensible_actions(state)
				rand = random.uniform(0,1)
				if rand < self.epsilon:
					action = random.choices(list(self.actions))[0]
				else:
					state_scores = self.qvalues[self.get_state_string(state)]
					action = max(state_scores.items(), key=operator.itemgetter(1))[0]
				new_state, observation, reward, done, info = self.env.simulate_step(action=action, state=state)
				new_state_str = self.get_state_string(new_state)
				if new_state_str not in self.qvalues.keys():
					new_state_actions = self.sensible_actions(new_state)
					self.qvalues[new_state_str]={}
					for new_action in new_state_actions:
						self.qvalues[new_state_str][new_action] = 0
				if done and info["all_boxes_on_target"]:
					count = count +1
					self.qvalues[state_str][action] = (1-self.lr) * (self.qvalues[state_str][action]) + self.lr * reward
					break
				else:
					self.qvalues[state_str][action] = (1-self.lr) * (self.qvalues[state_str][action]) + self.lr * (reward + self.discount*max(self.qvalues[new_state_str].values()))
				state = new_state
		if(count>10):
			self.depth=20
	def qtrain_back(self,  env_state):
		count =0
		for _ in tqdm(range(self.epochs)):
			state = env_state
			history=[]
			for _ in range(self.depth):
				state_str = self.get_state_string(state)
				actions = self.sensible_actions(state)
				rand = random.uniform(0,1)
				action =  self.uct()
				if rand < self.epsilon:
					action = random.choices(list(actions))[0]
				else:
					state_scores = self.qvalues[self.get_state_string(state)]
					action = max(state_scores.items(), key=operator.itemgetter(1))[0]
				new_state, observation, reward, done, info = self.env.simulate_step(action=action, state=state)
				new_state_str = self.get_state_string(new_state)
				if new_state_str not in self.qvalues.keys():
					new_state_actions = self.sensible_actions(new_state)
					self.qvalues[new_state_str]={}
					for new_action in new_state_actions:
						self.qvalues[new_state_str][new_action] = 0

				
				if done and info["all_boxes_on_target"]:
					count = count +1
					self.qvalues[state_str][action] = (1-self.lr) * (self.qvalues[state_str][action]) + self.lr * reward
					break
				else:
					history.append((state_str,new_state_str,action,reward))
					#self.qvalues[state_str][action] = (1-self.lr) * (self.qvalues[state_str][action]) + self.lr * (reward + self.discount*max(self.qvalues[new_state_str].values()))
				state = new_state

			heuristic_reward = -self.heuristic(env_state[3]) + self.heuristic(state[3])
			step = history.pop()
			state_str,new_state_str,action,reward =  step
			reward = reward+heuristic_reward
			history.append((state_str,new_state_str,action,reward))
			for step in history[::-1]:
				state_str,new_state_str,action,reward =  step
				#print(state_str,self.qvalues[state_str],action)
				self.qvalues[state_str][action] = (1-self.lr) * (self.qvalues[state_str][action]) + self.lr * (reward + self.discount*max(self.qvalues[new_state_str].values()))
		print(count,self.depth)
		if(count>10):
			self.depth=20
		print(count,self.depth)

	def heuristic(self, room_state):
		total = 0
		arr_goals = (self.room_fixed == 2)
		arr_boxes = ((room_state == 4) + (room_state == 3))
		# find distance between each box and its nearest storage
		for i in range(len(arr_boxes)):
			for j in range(len(arr_boxes[i])):
				if arr_boxes[i][j] == 1: # found a box
					min_dist = 9999999
					# check every storage
					for k in range(len(arr_goals)):
						for l in range(len(arr_goals[k])):
							if arr_goals[k][l] == 1: # found a storage
								min_dist = min(min_dist, abs(i - k) + abs(j - l))
					total = total + min_dist
		return total * self.penalty_for_step * 0.9

	def take_best_action(self,observation_mode="rgb_array"):
		env_state = self.env.get_current_state()
		new_state_str = self.get_state_string(env_state)
		new_state_actions = self.sensible_actions(env_state)
		if new_state_str not in self.qvalues.keys():
			self.qvalues[new_state_str]={}
			for new_action in new_state_actions:
				self.qvalues[new_state_str][new_action] = 0
		if(self.train):
			self.qtrain_back(env_state);
		else:
			sleep(1)

		state_scores = self.qvalues[self.get_state_string(env_state)]
		action_key = max(state_scores.items(), key=operator.itemgetter(1))[0]

		observation, reward, done, info = self.env.step(action_key)			
		return observation, reward, done, info

	def sensible_actions(self, state):
		player_position = state[2]
		room_state = state[3]
		def sensible(action, room_state, player_position):
			change = CHANGE_COORDINATES[action - 1] 
			new_pos = player_position + change
			#if the next pos is a wall
			if room_state[new_pos[0], new_pos[1]] == 0:
				return False
			new_box_position = new_pos + change
			# if a box is already at a wall
			if new_box_position[0] >= room_state.shape[0] \
				or new_box_position[1] >= room_state.shape[1]:
					return False
			can_push_box = room_state[new_pos[0], new_pos[1]] in [3, 4]
			can_push_box &= room_state[new_box_position[0], new_box_position[1]] in [1, 2]
			if can_push_box:
				#check if we are pushing a box into a corner
				if self.room_fixed[new_box_position[0], new_box_position[1]] != 2:
					box_surroundings_walls = []
					for i in range(4):
						surrounding_block = new_box_position + CHANGE_COORDINATES[i]
						if self.room_fixed[surrounding_block[0], surrounding_block[1]] == 0:
							box_surroundings_walls.append(True)
						else:
							box_surroundings_walls.append(False)
					if box_surroundings_walls.count(True) >= 2:
						if box_surroundings_walls.count(True) > 2:
							return False
						if not ((box_surroundings_walls[0] and box_surroundings_walls[1]) or (box_surroundings_walls[2] and box_surroundings_walls[3])):
							return False
			# trying to push box into wall
			if room_state[new_pos[0], new_pos[1]] in [3, 4] and room_state[new_box_position[0], new_box_position[1]] not in [1, 2]:
				return False
			return True
		return [action for action in self.actions if sensible(action, room_state, player_position)] 


def q_solve(args,file):
	if args.render_mode == "raw":
		observation_mode = "raw"
	elif "tiny" in args.render_mode:
		observation_mode = "tiny_rgb_array"
	else:
		observation_mode = "rgb_array"
	dim_room, n_boxes, map = parse(filename= file)
	actions = []
	env = gym.make("MCTS-Sokoban-v0", dim_room=dim_room, num_boxes=n_boxes, original_map=map, max_steps=args.max_steps)
	solver = Q_solver(env=env, actions=LEGAL_ACTIONS)
	allocated_time = args.time_limit * 60
	start_time = time()
	while True:
		now = time()
		if now - start_time > allocated_time:
			break
		env.render(mode=args.render_mode)
		observation, reward, done, info = solver.take_best_action(observation_mode=observation_mode)
		if "action.name" in info:            
			actions.append(ACTION_MAP[info["action.name"]])
		if done and "mcts_giveup" in info:
			env.reset()
			actions.clear()
		elif done and info["all_boxes_on_target"]:
			actions.append("Solved in {:.0f} mins".format((now - start_time)/60))
			break
		elif done and info["maxsteps_used"]:
			env.reset()
			actions.clear()
	env.render(mode=args.render_mode)
	sleep(3)
	env.close()
	log_dir = Path(args.log_dir)
	log_dir.mkdir(exist_ok=True)
	with open(log_dir / "{}.log".format(file.stem), mode="w") as log:
		print("{}".format(len(actions)), file=log, end="")
		for action in actions:
			print(" {}".format(action), file=log, end="")

def main(args):
	if args.file:
		for file in args.file:
			q_solve(args, Path(file))
	else:
		for file in Path(args.folder).iterdir():
			q_solve(args, file)
			
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--file", nargs = "+", help= "file that defines the sokoban map")
	group.add_argument("--folder", help= "folder that contains files which define the sokoban map")
	parser.add_argument("--render_mode", help="Obversation mode for the game. Use human to see a render on the screen", default="raw")
	parser.add_argument("--max_rollouts", type=int, help="Number of rollouts per move", default=4000)
	parser.add_argument("--max_depth", type=int, help="Depth of each rollout", default=30)
	parser.add_argument("--max_steps", type=int, help="Max moves before game is lost", default=120)
	parser.add_argument("--time_limit", type=int, help="Allocated Time (in minutes) per board", default=60)
	parser.add_argument("--log_dir", type=str, help="Directory to log solve information", default="./solve_log")
	args = parser.parse_args()
	main(args)