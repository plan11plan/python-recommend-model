import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, GMF, MLP, num_factor):
        super(NeuMF, self).__init__()
        self.gmf_user_emb = GMF.user_emb
        self.gmf_item_emb = GMF.item_emb

        self.mlp_user_emb = MLP.user_emb
        self.mlp_item_emb = MLP.item_emb

        self.mlp_layer = MLP.MLP_layers
        for i in self.mlp_layer:
            if isinstance(i, nn.Linear):
                out_features = i.out_features

        self.predict_layer = nn.Sequential(
            nn.Linear(num_factor + out_features, 1, bias = False),
        )

        self._init_weight_()

    def _init_weight_(self):
        for m in self.predict_layer:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, user, item):
        gmf_user_emb = self.gmf_user_emb(user)
        gmf_item_emb = self.gmf_item_emb(item)
        gmf_output = gmf_user_emb * gmf_item_emb

        mlp_user_emb = self.mlp_user_emb(user)
        mlp_item_emb = self.mlp_item_emb(item)
        mlp_cat_emb = torch.cat((mlp_user_emb, mlp_item_emb), -1)
        mlp_output = self.mlp_layer(mlp_cat_emb)

        cat_output = torch.cat((gmf_output, mlp_output), -1)

        output = self.predict_layer(cat_output)

        return output.view(-1)
    
    
class BPR_Loss(nn.Module):
    def __init__(self):
        super(BPR_Loss, self).__init__()

    def forward(self, pos, neg):
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos - neg)))
        return bpr_loss