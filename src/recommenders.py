from stellargraph.core.graph import StellarGraph
from pandas import Series
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Model
from sklearn import preprocessing, model_selection
import os
import dill as pickle
from src.util import config, mending_graph


class Recommender:
    def __init__(self, hasG:StellarGraph,
                 acc_path:str, recommender_path:list, downstream_task_path:str):
        
        self.downstream_task_path = downstream_task_path
        self.hasG = hasG
        
        '''
        self.train_subjects, self.test_subjects = model_selection.train_test_split(
            self.has_node_subjects,train_size=0.5,test_size=0.2, stratify=self.has_node_subjects
        )

        self.all_classes=all_classes

        self.train_targets = preprocessing.label_binarize(y=self.train_subjects,classes=all_classes)
        self.test_targets = preprocessing.label_binarize(y=self.test_subjects,classes=self.all_classes)
        self.has_node_targets = preprocessing.label_binarize(y=self.has_node_subjects,classes=self.all_classes)
        
        # Train Test Set을 어떻게 분리할 것 인가????
        
        '''
        
        self.batch_size = config.batch_size
        self.num_samples = config.num_samples
        
        self.acc_path = acc_path
        self.recommender_path = recommender_path
        self.feat_shape = len(self.hasG.node_features()[0])
        
    def set_recommenders(self, recommender_path, dataowner, hasG_hide:StellarGraph):
        self.recommender_path = recommender_path
        
        self.hasG_hide = hasG_hide
        self.hasG_node_gen = GraphSAGENodeGenerator(self.hasG, self.batch_size, self.num_samples)
        # self.all_train_gen = self.hasG_node_gen.flow(self.train)
        # 
        
        self.locLGCN = self.build_recommender(self.hasG_node_gen)
        self.acc_path = dataowner.test_acc_path
        
        self.fedLGCN = None
        self.fedLGCNPC = None
        
    def save_recommender_instance(self):
        pickle.dump(self, open(self.downstream_task_path, "wb"))
        return
        
    
    def build_recommender(self, fillG_node_gen):
        graphsage_model = GraphSAGE(layer_sizes=config.recommendation_layer_sizes, generator=fillG_node_gen,
                                    n_samples=config.num_samples)
        x_inp, x_out = graphsage_model.in_out_tensors()
        # train_targets[0]에 대해서 확인
        prediction = layers.Dense(len(self.train_targets[0]),activation='softmax')(x_out)
        model = Model(inputs=x_inp, outputs=prediction)
        model.compile(
            optimizer=optimizers.Adam(config.lr),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )
    
        
    # 보충해야함
    def train_locLGCN(self):
        history = self.locLGCN.fit(
            self.all_train_gen, epochs=config.epoch_recommendation
        )
        self.print_test(self.locLGCN, "LocLGCN")
        
    def save_locLGCN(self):
        print("saving LocLGCN recommender ...")
        self.locLGCN.save_weights(self.recommender_path[0])
    
    def load_locLGCN(self):
        print("loading LocLGCN recommender")
        self.locLGCN.load_weights(self.recommender_path[0])
        
    def print_test(self, recommender, name='LocLGCN'):
        all_test_metrics_all = recommender.evaulate(self.all_test_gen)
        print("\n"+name)
        print("\nLocal Test Set Metrics:")
        
        with open(self.acc_path, 'a') as f:
            f.write("\n" + name)
            f.write("\nLocal Test Set Metrics:")
        for name,val in zip(recommender.metrics_names, all_test_metrics_all):
            print("\t{}: {:0.4f}".format(name, val))
            with open(self.acc_path, 'a') as f:
                f.write("\t{}: {:0.4f}".format(name, val))
                
    def pred_missing_neigh(self, generator_model, all_feat, edges, adj):
        
        pred_missing, pred_feat, _ = generator_model(all_feat, edges, adj)
        pred_feat.view(-1, config.num_pred, self.feat_shape)
        pred_feat = pred_feat.cpu().detach().numpy()
        pred_missing = np.around(pred_missing.cpu().detach().numpy()).astype(int)
        
        return pred_missing, pred_feat
    
    
    def eval_pred_Gnew(self, generator_model, all_feat, edges, adj, acc_path,
                       impaired_graph:StellarGraph, org_graph:StellarGraph,
                       test_flag=False, save_flag=False,
                       global_task=None):
        pred_missing, pred_feats = \
            self.pred_missing_neigh(generator_model, all_feat, edges, adj)
            
        if test_flag==True:
            fill_nodes, fill_G = mending_graph.fill_graph(impaired_graph,
                                                          org_graph, pred_missing, pred_feats,
                                                          self.feat_shape)
            
            '''
            fillG_node_gen = GraphSAGENodeGenerator(fill_G, self.batch_size, self.num_samples)
            fill_train_gen = fillG_node_gen.flow(self.train)
            
            self.locLGCNPC = self.build_recommender(fillG_node_gen)
            if os.path.isfile(self.recommender_path[-1]+"locLGCNPC.h5") == False:
                history = self.locLGCNPC.fit(
                    fill_train_gen, epochs=config.epoch_recommendation, verbose=2, shuffle=False
                )
                
                if save_flag:
                    self.locLGCNPC.save_weights(self.recommender_path[-1]+"locLGCNPC.h5")
            else:
                self.locLGCNPC.load_weights(self.recommender_path[-1]+"locLGCNPC.h5")
            if global_task.test_only_gen is not None:
                self.test_global(global_task, self.locLGCNPC, acc_path, "LocLGCNPlusC", "")
                self.test_global(global_task, self.locLGCN, acc_path, "LocLGCN", "")
                
            '''

            with open(acc_path, 'a') as f:
                f.write("\nlocal #nodes = " + str(len(self.hasG.node_features())))
                f.write("\nlocal #edges = " + str(len(list(self.hasG.edges()))) + "\n\n\n")
            
        return pred_missing, pred_feats
    
    
    def test_global(self, global_taskrecommender, acc_path, name='MD', prefix=''):
        test_metrics_org_fill = recommender.evaulate(global_task.test_only_gen)
        
        with open(acc_path, 'a') as f:
            f.write("\n"+prefix+" "+name+" Global Org Test Set Metrics:")
        for name, val in zip(recommender.metrics_names, test_metrics_org_fill):
            with open(acc_path, 'a') as f:
                f.write("\t{}: {:0.4f}".format(name, val))
        return test_metrics_org_fill[-1]
    
    
    def save_fedLGCN(self):
        self.fedLGCN.save_weights(self.recommender_path[-1]+"fedLGCN.h5")
        
    def load_fedLGCN(self, test_gen):
        self.fedLGCN=self.build_recommender(test_gen)
        self.fedLGCN.load_weights(self.recommender_path[-1]+"fedLGCN.h5")
        
    def save_fedLGCNPC(self):
        self.fedLGCNPC.save_weights(self.recommender_path[-1]+"fedLGCNPlusC.h5")
    
    def load_fedLGCNPC(self, test_gen):
        self.fedLGCNPC=self.build_recommender(test_gen)
        self.fedLGCNPC.load_weights(self.recommender_path[-1]+"fedLGCNPlusC.h5")