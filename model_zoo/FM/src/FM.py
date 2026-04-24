import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, FactorizationMachine, InnerProductInteraction, LogisticRegression


class DeepFM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DeepFM", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 model_structure="DeepFM",
                 stacked_dnn_hidden_units=[],
                 parallel_dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(DeepFM, self).__init__(feature_map, 
                                     model_id=model_id, 
                                     gpu=gpu, 
                                     embedding_regularizer=embedding_regularizer, 
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        
        # Embedding Layer
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
    
        input_dim = feature_map.sum_emb_out_dim()
        self.model_structure = model_structure
        
        self.fm_layer = InnerProductInteraction(feature_map.num_fields, output="product_sum")
        self.lr_layer = LogisticRegression(feature_map, use_bias=True)
        
        
        
        assert self.model_structure in ["FM_only", "DeepFM", "NFM"], \
               "model_structure={} not supported!".format(self.model_structure)
               
        if self.model_structure in ["FM_only"]:
            final_dim = embedding_dim
        
        if self.model_structure in ["NFM"]:
            self.stacked_dnn = MLP_Block(input_dim=embedding_dim,
                                         output_dim=None, # output hidden layer
                                         hidden_units=stacked_dnn_hidden_units,
                                         hidden_activations=dnn_activations,
                                         output_activation=None, 
                                         dropout_rates=net_dropout,
                                         batch_norm=batch_norm)
            
            final_dim = stacked_dnn_hidden_units[-1]
        
        if self.model_structure in ["DeepFM"]:
            self.parallel_dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                                          output_dim=None, # output hidden layer
                                          hidden_units=parallel_dnn_hidden_units,
                                          hidden_activations=dnn_activations,
                                          output_activation=None, 
                                          dropout_rates=net_dropout, 
                                          batch_norm=batch_norm)
            final_dim = embedding_dim + parallel_dnn_hidden_units[-1]
        
        self.fc = nn.Linear(final_dim, 1)
        self.analyzing = kwargs.get("analyzing", False)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def init_record(self):
        self.record_feature_emb = []
        
        self.record_bi_pooling_vec = []
        self.record_gating = []
        self.record_gating_linear = []
        self.record_final_representation = []
        

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        self.grad_var_list = []
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)   #[B,F,D]
        if self.training and self.analyzing:
            feature_emb.retain_grad()
        self.feature_embedding_grad = feature_emb
        
        if self.analyzing:
            self.record_feature_emb.append(feature_emb.detach().clone().cpu())
        
        gating = feature_emb


        row, col = torch.triu_indices(feature_emb.shape[1], feature_emb.shape[1], offset=1)
        rst = gating[:, row] * feature_emb[:, col]
        bi_pooling_vec = rst.sum(-2)
        if self.analyzing:
            self.record_bi_pooling_vec.append(bi_pooling_vec.detach().clone().cpu())
            self.record_final_representation.append(bi_pooling_vec.detach().clone().cpu())
        if self.training and self.analyzing:
            bi_pooling_vec.retain_grad()
        self.grad_var_list.append(bi_pooling_vec)

        if self.model_structure == "FM_only":
            final_out = bi_pooling_vec.sum(dim=-1, keepdim=True)
            
        elif self.model_structure == "NFM":
            final_out = self.stacked_dnn(bi_pooling_vec)
                
        elif self.model_structure == "DeepFM":
            dnn_out = self.parallel_dnn(feature_emb.flatten(start_dim=1))
            final_out = torch.cat([bi_pooling_vec, dnn_out], dim=-1)
            
            
        if self.model_structure == "FM_only":
            y_pred = final_out
        else:
            y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

