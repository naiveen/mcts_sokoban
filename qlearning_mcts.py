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
from math import sqrt, log
import numpy as np
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
	def __init__(self, env,actions,depth=50, epochs=4096*2 ):
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
		self.nvalues={}
		self.last_pos = self.env.get_current_state()[2]
		self.move_box = False
		self.env.reset()
		self.train =1


	def get_state_string(self, env_state):
		state_string = ''.join([''.join([str(element) for element in row]) for row in env_state[3]])
		return state_string

	def uct(self, state,alpha =0.1):
		state_str = self.get_state_string(state)
		actions = self.sensible_actions(state)
		"""
		for action in actions:
			print(action ,self.qvalues[state_str][action] ,self.nvalues[state_str][action],  (self.qvalues[state_str][action] / self.nvalues[state_str][action]) + (sqrt(2 * log(self.nvalues[state_str]["count"]) / self.nvalues[state_str][action])))
		"""
		best_action = max(actions, key = lambda action: (self.qvalues[state_str][action]  + (sqrt(2) * log(self.nvalues[state_str]["count"]) / self.nvalues[state_str][action])))
		return best_action
		
	def mcts_select(self, state, history):
		done = False
		actions = self.sensible_actions(state)
		#print(actions, state)
		state_str = self.get_state_string(state)
		loop_history = []
		loop_history.append(state_str)
		while not done:
			sensible_actions = self.sensible_actions(state)
			explored_actions = self.qvalues[state_str].keys()
			unexplored_actions = set(sensible_actions) - set(explored_actions)
			if(len(unexplored_actions)!=0):
				return state,done
			action =  self.uct(state)
			new_state, observation, reward, done, info = self.env.simulate_step(action=action, state=state)
			#print(new_state,action)
			self.last_pos = state[2]
			self.move_box = info["action.moved_box"]
			actions = self.sensible_actions(new_state)
			new_state_str = self.get_state_string(new_state)
			if not actions:
				history.append((state_str,action,-10))
				return state,True
			if new_state_str in loop_history:
				history.append((state_str,action,-10))
				return state,True
			else:
				loop_history.append(new_state_str)
			history.append((state_str,action,reward))
			state = new_state
			state_str = new_state_str
		if self.env.num_boxes == state[0]:
			return state, done
		state_str,action,reward = history.pop()
		history.append((state_str,action,-10))
		return state,done

	def mcts_expand(self,state):
		sensible_actions = self.sensible_actions(state)
		state_str = self.get_state_string(state)
		explored_actions = self.qvalues[state_str].keys()
		unexplored_actions = set(sensible_actions) - set(explored_actions)
		action = random.choice(list(unexplored_actions))
		self.qvalues[state_str][action] = 0
		self.nvalues[state_str][action] = 0
		new_state, observation, reward, done, info = self.env.simulate_step(action=action, state=state)
		self.last_pos = state[2]
		self.move_box = info["action.moved_box"]
		new_state_str = self.get_state_string(new_state)
		if new_state_str not in self.qvalues.keys():
			self.qvalues[new_state_str]={}
			self.nvalues[new_state_str]={}
			self.nvalues[new_state_str]["count"] = 0;
		#print(state_str,new_state,action)
		return state_str,new_state,action

	def mcts_simulate(self, env_state):
		simulation_reward=0
		state = env_state
		for _ in range(self.depth):
			actions = self.sensible_actions(state)
			if not actions:
				simulation_reward = -10
				break
			action = random.choice(actions)
			new_state, observation, reward, done, info = self.env.simulate_step(action=action, state=state)
			simulation_reward += reward
			state = new_state
			self.last_pos = state[2]
			self.move_box = info["action.moved_box"]
		heuristic_reward = -self.heuristic(env_state[3]) + self.heuristic(state[3])
		return simulation_reward + heuristic_reward

	def back_propagate(self, history):
		R = 0
		for step in history[::-1]:
			state_str,action,reward =  step
			self.nvalues[state_str][action] +=1;
			self.nvalues[state_str]["count"] +=1;
			Q = self.qvalues[state_str][action]
			R = reward + self.discount * R
			self.qvalues[state_str][action] = Q + (R - Q)/self.nvalues[state_str][action]

	def qtrain_mcts(self,  env_state):
		last_pos = self.last_pos
		move_box = self.move_box
		for _ in tqdm(range(self.epochs)):
			history=[]
			state = env_state
			self.last_pos = last_pos
			self.move_box = move_box    
			leafState ,done = self.mcts_select(state,history)
			if not done:
				leafStateStr, exploreState , action = self.mcts_expand(leafState)
				simulation_reward = self.mcts_simulate(exploreState)
				#print(simulation_reward)
				history.append((leafStateStr,action, simulation_reward))
			self.back_propagate(history)

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


	def sensible_actions(self, state):
		player_position = state[2]
		room_state = state[3]
		last_pos = self.last_pos
		move_box = self.move_box
		def sensible(action, room_state, player_position, last_pos, move_box):
			change = CHANGE_COORDINATES[action - 1] 
			new_pos = player_position + change
			#if the next pos is a wall
			if room_state[new_pos[0], new_pos[1]] == 0:
				return False
			if np.array_equal(new_pos, last_pos) and not move_box:
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
		return [action for action in self.actions if sensible(action, room_state, player_position, last_pos, move_box)]



	def take_best_action(self,observation_mode="rgb_array"):
		env_state = self.env.get_current_state()
		self.last_pos = env_state[2]
		state_str = self.get_state_string(env_state)
		self.qvalues[state_str]={}
		self.nvalues[state_str]={}
		self.nvalues[state_str]["count"] = 0
		self.qtrain_mcts(env_state)
		print(len(self.qvalues.keys()))
		self.epochs = 100
		state_counts = self.nvalues[self.get_state_string(env_state)]
		action_key = max(state_counts.items(), key=operator.itemgetter(1))[0]
		action_key = self.uct(env_state)
		
		observation, reward, done, info = self.env.step(action_key)
		self.move_box = info["action.moved_box"]        
		return observation, reward, done, info

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
	parser.add_argument("--max_steps", type=int, help="Max moves before game is lost", default=1000)
	parser.add_argument("--time_limit", type=int, help="Allocated Time (in minutes) per board", default=60)
	parser.add_argument("--log_dir", type=str, help="Directory to log solve information", default="./solve_log")
	args = parser.parse_args()
	main(args)