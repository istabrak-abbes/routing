"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
from model import DRL4TSP, Encoder
import pickle
import GA
from collections import OrderedDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output

class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):

        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)


def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = np.inf
    1.5806
    1.8143
    
    for epoch in range(10):

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):

            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)
            
            print ("tour indices = ",tour_indices)
            print ("tour logp \n ", tour_logp)

            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static, tour_indices)
            print("reward = ", reward)
            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)
            print("the critic estimate equal to = ",critic_est)
            advantage = (reward - critic_est)
            print("the advantage equal to  = ",advantage)
            
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)
            print("actor loss: ",actor_loss)
            print("critic loss : ",critic_loss)
            
            actor_optim.zero_grad()
            print(actor_optim.zero_grad())
            actor_loss.backward()
            #print(actor_loss.backward())
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            print(torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm))
            actor_optim.step()
            print(actor_optim.step())

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            print("critic rewards ",critic_rewards)
            rewards.append(torch.mean(reward.detach()).item())
            print("rewards equal",rewards)
            losses.append(torch.mean(actor_loss.detach()).item())
            print("losses equal to", losses)
            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])
                

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))
        mean_loss = np.mean(losses)
        print("mean loss= ",mean_loss)
        mean_reward = np.mean(rewards)
        print("mean reward ",mean_reward)
        

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pkl')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pkl')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)

        mean_valid = validate(valid_loader, actor, reward_fn, render_fn,
                              valid_dir, num_plot=5)
        print ("mean valid ",mean_valid)

        # Save best model parameters
        if mean_valid < best_reward:

            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pkl')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pkl')
            torch.save(critic.state_dict(), save_path)

        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
              np.mean(times)))
        print("--------------------------------------------------------------------------------------------------------------------------------------------")

def train_tsp(args):

    # Goals from paper:
    # TSP20, 3.97
    # TSP50, 6.08
    # TSP100, 8.44

    from tasks import tsp
    from tasks.tsp import TSPDataset

    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 1 # dummy for compatibility

    train_data = TSPDataset(args.num_nodes, args.train_size, args.seed)
    valid_data = TSPDataset(args.num_nodes, args.valid_size, args.seed + 1)
    test_data  = TSPDataset(args.num_nodes, args.train_size, args.seed + 2)

    update_fn = None

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    update_fn,
                    tsp.update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)
    

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = tsp.reward
    kwargs['render_fn'] = tsp.render

    if args.checkpoint:
    
        path = os.path.join(args.checkpoint, 'actor.pkl')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pkl')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)


    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size,True, num_workers=0)
    out = validate(test_loader, actor, tsp.reward, tsp.render, test_dir, num_plot=5)





    print('Average tour length: ', out)
    
    
   
  


def train_vrp(args):

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    from tasks import vrp
    from tasks.vrp import VehicleRoutingDataset

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {6:15, 8: 18,  10: 20, 12: 23, 15: 24, 18: 25,  20: 30,22:35,  50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 2 # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]

    train_data = VehicleRoutingDataset(args.train_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed)

    valid_data = VehicleRoutingDataset(args.valid_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed + 1)

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    train_data.update_dynamic,
                    train_data.update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = vrp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pkl')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pkl')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)

    print('Average tour length: ', out)
    
    
#     sol_per_pop = 8
#     num_parents_mating = 4
#     num_generations = 5
#     mutation_percent = 50
# # # # #     # f =  torch.load(r"C:\Users\MY HP\Desktop\pallet loading\pytorch-drl4vrp-master (3) (1)\pytorch-drl4vrp-master\vrp\initial_pop\6\21_41_41.887402\actor.pkl", map_location ='cpu')

#     initial_pop_weights = []
#     initial_pop_weights1 = []
#     initial_pop_weights2 = []
#     initial_pop_weights3 = []
#     initial_pop_weights4 = []


#     for curr_sol in np.arange(0, sol_per_pop):
#         input_HL1_weights = np.random.uniform(low=-0.1, high=0.1, 
#                                               size=(1, 128, 384))
#         HL1_HL2_weights = np.random.uniform(low=-0.1, high=0.1, 
#                                               size=(1, 1, 128))
#         conv_weights = np.random.uniform(low=0.1, high=0.1, size=(128,2,1))
#         static_conv_weights = np.random.uniform(low=0.1, high=0.1, size=(128,2,1))
#         decoder_conv_weight = np.random.uniform(low=0.1, high=0.1, size=(128,2,1))

# #         # HL2_output_weights = np.random.uniform(low=-0.1, high=0.1, 
# #         #                                       size=(128,2,1))

#         initial_pop_weights.append(np.array([input_HL1_weights, 
#                                                 ]))
        
#         initial_pop_weights1.append(np.array([ HL1_HL2_weights, 
#                                                 ]))
#         initial_pop_weights2.append(np.array([ conv_weights 
#                                                 ]))
#         initial_pop_weights3.append(np.array([ static_conv_weights 
#                                                 ]))
        
#         initial_pop_weights4.append(np.array([ decoder_conv_weight 
#                                                 ]))
#     pop_weights_mat = np.array(initial_pop_weights)
#     pop_weights_vector = GA.mat_to_vector(pop_weights_mat)
    
#     pop_weights_mat1 = np.array(initial_pop_weights1)
#     pop_weights_vector1 = GA.mat_to_vector(pop_weights_mat1)
    
#     pop_weights_mat2 = np.array(initial_pop_weights2)
#     pop_weights_vector2 = GA.mat_to_vector(pop_weights_mat2)
    
#     pop_weights_mat3 = np.array(initial_pop_weights3)
#     pop_weights_vector3 = GA.mat_to_vector(pop_weights_mat3)
    
#     pop_weights_mat4 = np.array(initial_pop_weights4)
#     pop_weights_vector4 = GA.mat_to_vector(pop_weights_mat4)


#     for generation in range(num_generations):
#         print("Generation : ", generation)

# #     # converting the solutions from being vectors to matrices.
#         pop_weights_mat = GA.vector_to_mat(pop_weights_vector, 
#                                         pop_weights_mat)
        
#         pop_weights_mat1 = GA.vector_to_mat(pop_weights_vector1, 
#                                         pop_weights_mat1)
        
#         pop_weights_mat2 = GA.vector_to_mat(pop_weights_vector2, 
#                                         pop_weights_mat2)
        
#         pop_weights_mat3 = GA.vector_to_mat(pop_weights_vector3, 
#                                         pop_weights_mat3)
        
#         pop_weights_mat4 = GA.vector_to_mat(pop_weights_vector4, 
#                                         pop_weights_mat4)

#     # Measuring the fitness of each chromosome in the population.
#     # fitness = ANN.fitness(pop_weights_mat, 
#     #                       data_inputs, 
#     #                       data_outputs, 
#     #                       activation="sigmoid")
#     # accuracies[generation] = fitness[0]
#     # print("Fitness")
#     # print(fitness)

# #     # Selecting the best parents in the population for mating.
#         parents = GA.select_mating_pool(pop_weights_vector, 
#                                     out.copy(), 
#                                     num_parents_mating)
        
#         parents1 = GA.select_mating_pool(pop_weights_vector1, 
#                                     out.copy(), 
#                                     num_parents_mating)
        
#         parents2 = GA.select_mating_pool(pop_weights_vector2, 
#                                     out.copy(), 
#                                     num_parents_mating)
#         parents3 = GA.select_mating_pool(pop_weights_vector3, 
#                                     out.copy(), 
#                                     num_parents_mating)
        
#         parents4 = GA.select_mating_pool(pop_weights_vector4, 
#                                     out.copy(), 
#                                     num_parents_mating)
    
    
#         print("Parents:")
#         print(parents.shape)
#         print("initial")
#         print(pop_weights_vector.shape)
#     # Generating next generation using crossover.
#         offspring_crossover = GA.crossover(parents,
#                                         offspring_size=(pop_weights_vector.shape[0]-parents.shape[0], pop_weights_vector.shape[1]))
#         offspring_crossover1 = GA.crossover(parents1,
#                                         offspring_size=(pop_weights_vector1.shape[0]-parents1.shape[0], pop_weights_vector1.shape[1]))
        
#         offspring_crossover2 = GA.crossover(parents2,offspring_size=(pop_weights_vector2.shape[0]-parents2.shape[0], pop_weights_vector2.shape[1]))
        
#         offspring_crossover3 = GA.crossover(parents3,
#                                         offspring_size=(pop_weights_vector3.shape[0]-parents3.shape[0], pop_weights_vector3.shape[1]))
        
        
#         offspring_crossover4 = GA.crossover(parents4,
#                                         offspring_size=(pop_weights_vector4.shape[0]-parents4.shape[0], pop_weights_vector4.shape[1]))
        
        
#         print("Crossover")
#         print(offspring_crossover)

#     # Adding some variations to the offsrping using mutation.
#         offspring_mutation = GA.mutation(offspring_crossover, 
#                                       mutation_percent=mutation_percent)
        
#         offspring_mutation1 = GA.mutation(offspring_crossover1, 
#                                       mutation_percent=mutation_percent)
        
        
#         offspring_mutation2 = GA.mutation(offspring_crossover2, 
#                                       mutation_percent=mutation_percent)
          
#         offspring_mutation3 = GA.mutation(offspring_crossover3, 
#                                       mutation_percent=mutation_percent)
        
#         offspring_mutation4 = GA.mutation(offspring_crossover4, 
#                                       mutation_percent=mutation_percent)
#         print("Mutation")
#         print(offspring_mutation)

#     # Creating the new population based on the parents and offspring.
#         pop_weights_vector[0:parents.shape[0], :] = parents
#         pop_weights_vector[parents.shape[0]:, :] = offspring_mutation
        
#         pop_weights_vector1[0:parents1.shape[0], :] = parents1
#         pop_weights_vector1[parents1.shape[0]:, :] = offspring_mutation1
        
#         pop_weights_vector2[0:parents2.shape[0], :] = parents2
#         pop_weights_vector2[parents2.shape[0]:, :] = offspring_mutation2
        
#         pop_weights_vector3[0:parents3.shape[0], :] = parents3
#         pop_weights_vector3[parents3.shape[0]:, :] = offspring_mutation3
        
#         pop_weights_vector4[0:parents4.shape[0], :] = parents4
#         pop_weights_vector4[parents4.shape[0]:, :] = offspring_mutation4
        
#     pop_weights_mat = GA.vector_to_mat(pop_weights_vector, pop_weights_mat)
#     best_weights = pop_weights_mat [0, :]
    
#     pop_weights_mat1 = GA.vector_to_mat(pop_weights_vector1, pop_weights_mat1)
#     best_weights1 = pop_weights_mat1 [0, :]
    
#     pop_weights_mat2 = GA.vector_to_mat(pop_weights_vector2, pop_weights_mat2)
#     best_weights2 = pop_weights_mat2 [0, :]
    
#     pop_weights_mat3 = GA.vector_to_mat(pop_weights_vector3, pop_weights_mat3)
#     best_weights3 = pop_weights_mat3[0, :]
    
    
#     pop_weights_mat4 = GA.vector_to_mat(pop_weights_vector4, pop_weights_mat4)
#     best_weights4 = pop_weights_mat4[0, :]
    
#     x1 = torch.from_numpy(np.asarray(best_weights1))
#     y1= torch.reshape(x1, (1, 1,128))
    
#     x = torch.from_numpy(np.asarray(best_weights))
#     y= torch.reshape(x, (1, 128, 384))
    
#     x2 = torch.from_numpy(np.asarray(best_weights2))
#     y2 = torch.reshape(x2, ( 128,2, 1))
    
#     x3 = torch.from_numpy(np.asarray(best_weights2))
#     y3 = torch.reshape(x3, ( 128,2, 1))
    
    
#     x4 = torch.from_numpy(np.asarray(best_weights2))
#     y4 = torch.reshape(x4, ( 128,2, 1))
    
#     print("best weights : ", y)
#     print("best weights1 : ", y1)
#     print("best weights1 : ", y2)
#     print("best weights1 : ", y3)


#     model_1 = torch.load(r"C:\Users\MY HP\Desktop\pallet loading\pytorch_vrp\pytorch-drl4vrp-master\vrp\10\12_02_45.085968\checkpoints\1\actor.pkl", map_location ='cpu')
#     d = OrderedDict(('pointer.encoder_attn.W',y) if key == 'pointer.encoder_attn.W' else (key, value) for key, value in model_1.items())
#     # d1 = OrderedDict(('pointer.encoder_attn.v',y1) if key == 'pointer.encoder_attn.v' else (key, value) for key, value in model_1.items())
#     d2 = OrderedDict(('dynamic_encoder.conv.weight',y) if key == 'dynamic_encoder.conv.weight' else (key, value) for key, value in model_1.items())
#     d3 = OrderedDict(('static_encoder.conv.weight',y) if key == 'static_encoder.conv.weight' else (key, value) for key, value in model_1.items())
#     d4 = OrderedDict(('decoder.conv.weight',y) if key == 'decoder.conv.weight' else (key, value) for key, value in model_1.items())

    
#     actor = open(r"C:\Users\MY HP\Desktop\pallet loading\pytorch_vrp\pytorch-drl4vrp-master\vrp\10\12_02_45.085968\checkpoints\1\actor.pkl", "wb")
#     torch.save(d, actor)
#     torch.save(d2, actor)
#     torch.save(d3, actor)
#     torch.save(d4, actor)

#     # torch.save(d1, actor)

#     actor.close()
  
    
    
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=500, type=int)
    parser.add_argument('--valid-size', default=500, type=int)

    args = parser.parse_args()

    # print('NOTE: SETTTING CHECKPOINT: ')
    # args.checkpoint = os.path.join('vrp', '10', '12_02_45.085968' + os.path.sep)
    # print(args.checkpoint)

    if args.task == 'tsp':
        train_tsp(args)
    elif args.task == 'vrp':
        train_vrp(args)
    else:
        raise ValueError('Task <%s> not understood'%args.task)
