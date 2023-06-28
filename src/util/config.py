import torch
import torch.optim as optim

root_path = "/home/wooseok/fedsageRec-main/"
no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
#cuda=False

dataset = "gowalla"
num_owners = 10
delta = 20
load_data = root_path+"dataowner/"+dataset
global_data_ratio = 0.5

use_early_stopping = True
early_stopping_patience = 10
early_stopping_policy = "ndcg" # recall, ndcg, hit
test_every = 2
print_every = 30

num_samples = [5, 5]
batch_size = 512
latent_dim = 64
steps = 10
epochs_local = 200
lr = 0.001
regs = 1e-5
weight_decay = 1e-4
hidden = 32
dropout = 0.1
n_layers = 2
mess_dropout = [0.10]*n_layers
layers = [64,64]
optimizer = optim.Adam
bpr_decay = 1e-3

gen_epochs = 10
num_pred = 40
hidden_portion = 0.15
top_k = 100

epoch_recommendation = 200
recommendation_layer_sizes = [64, 32]

communication_round = 40
classifier_valid_ratio = 0.1
communication_test_every = 1
communication_print_every = 4
aggregate_exception = ["user_emb", "item_emb"]

local_test_acc_dir = root_path+'local_result/test_acc/' + dataset+"_"+str(num_owners)
global_test_acc_file = root_path+'global_result/test_acc/' + dataset+"_"+str(num_owners)
global_recommendation_file = root_path+'global_result/model/' + dataset+"_"+str(num_owners)+"recommendation.h5"
local_gen_dir = root_path+'local_result/model/' + dataset+"_"+str(num_owners)
global_gen_rec_acc_file = root_path+'global_result/test_acc/' + dataset+"_rec_"+str(num_owners)+".txt"
local_downstream_task_dir = root_path+'local_result/recommendation_info/' + dataset+"_"+str(num_owners)
server_info_dir = root_path+'global_result/server_info/' + dataset+"_"+str(num_owners)+'.h5'
local_dataowner_info_dir=root_path+'dataowner/' + dataset+"_"+str(num_owners)

a = 1
b = 1
c = 1
k = 20