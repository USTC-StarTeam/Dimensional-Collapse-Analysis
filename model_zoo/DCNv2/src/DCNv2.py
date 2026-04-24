

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetMix
import torch.nn.functional as F

class DCNv2(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DCNv2", 
                 gpu=-1,
                 model_structure="stacked_parallel",
                 use_low_rank_mixture=False,
                 low_rank=32,
                 num_experts=4,
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 stacked_dnn_hidden_units=[], 
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None,
                 net_regularizer=None, 
                 **kwargs):
        super(DCNv2, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        
        input_dim = feature_map.sum_emb_out_dim()
        self.feature_gating = nn.Sequential(
            nn.Linear(feature_map.sum_emb_out_dim(), feature_map.sum_emb_out_dim()),
        )
    
        self.analyzing = kwargs.get("analyzing", False)



        self.crossnet = CrossNetV2(input_dim, num_cross_layers, embedding_dim, self.nonlinear, self.concat_emb, self.gamma, self.symmetric, kwargs)
        
        self.model_structure = model_structure
        assert self.model_structure in ["crossnet_only", "stacked", "parallel", "stacked_parallel"], \
               "model_structure={} not supported!".format(self.model_structure)
        if self.model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnn = MLP_Block(input_dim=input_dim,
                                         output_dim=None, # output hidden layer
                                         hidden_units=stacked_dnn_hidden_units,
                                         hidden_activations=None,
                                         output_activation=None, 
                                         dropout_rates=net_dropout,
                                         batch_norm=batch_norm)
            final_dim = stacked_dnn_hidden_units[-1]

        if self.model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = MLP_Block(input_dim=input_dim,
                                         output_dim=None, # output hidden layer
                                         hidden_units=parallel_dnn_hidden_units,
                                         hidden_activations=None,
                                         output_activation=None, 
                                         dropout_rates=net_dropout,
                                         batch_norm=batch_norm)
            final_dim = input_dim + parallel_dnn_hidden_units[-1]

        if self.model_structure == "stacked_parallel":
            final_dim = stacked_dnn_hidden_units[-1] + parallel_dnn_hidden_units[-1]
            
        if self.model_structure in ["crossnet_only"]: # only CrossNet 
            final_dim = input_dim


        self.fc = nn.Linear(final_dim, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def init_record(self):
        self.record_final_out = []
        self.record_feature_emb = []
        self.record_gating = []
        self.record_gating_linear = []
        self.record_final_representation = []


    def forward(self, inputs):
        self.grad_var_list = []
        X = self.get_inputs(inputs)

        feature_emb = self.embedding_layer(X, flatten_emb=True)

        if self.analyzing:
            self.record_feature_emb.append(feature_emb.detach().clone().cpu())
        if self.training and self.analyzing:
            feature_emb.retain_grad()
            
        self.feature_embedding_grad = feature_emb

        gating = feature_emb
        gating_linear = feature_emb
        feature_emb = gating
        
        if self.analyzing:
            self.record_gating.append(gating.detach().clone().cpu())
            self.record_gating_linear.append(gating_linear.detach().clone().cpu())
        if self.training and self.analyzing:
            gating.retain_grad()
            gating_linear.retain_grad()
        self.grad_var_list.append(gating)
        self.grad_var_list.append(gating_linear)

        cross_out = self.crossnet(feature_emb, gating=None)


        if self.model_structure == "crossnet_only":
            final_out = cross_out
            if self.analyzing:
                self.record_final_out.append(final_out.detach().clone().cpu())
                self.record_final_representation.append(final_out.detach().clone().cpu())

                
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)
            if self.analyzing:
                self.record_final_out.append(final_out.detach().clone().cpu())
                self.record_final_representation.append(final_out.detach().clone().cpu())


        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
            if self.analyzing:
                self.record_final_out.append(final_out.detach().clone().cpu())
                self.record_final_representation.append(final_out.detach().clone().cpu())
            
        
        elif self.model_structure == "stacked_parallel":
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(feature_emb)], dim=-1)


        y_pred = self.fc(final_out)
        if self.analyzing:
            self.record_final_out.append(final_out.detach().clone().cpu())

        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers, embedding_dim=None, nonlinear='relu', concat_emb=False, gamma=1, symmetric=False, kwargs=None):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
                                          for _ in range(self.num_layers))
        self.embedding_dim = embedding_dim
        self.num_field = input_dim // embedding_dim
        self.analyzing = kwargs.get("analyzing", False)
    
    def forward(self, feature_embedding, gating=None):
        self.grad_var_list = []
        X_0 = feature_embedding
        if gating is not None:
            X_0 = gating
        X_i = feature_embedding # b x dim
        for i in range(self.num_layers):
            tmp = self.cross_layers[i](X_i)
            X_i = X_i + X_0 * tmp

        return X_i
