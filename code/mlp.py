import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_user, num_item, num_factor, num_layers, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.user_emb = nn.Embedding(num_user, num_factor)
        self.item_emb = nn.Embedding(num_item, num_factor)

        MLP_modules = []
        input_size = num_factor * 2
        for i in range(num_layers):
            MLP_modules.append(nn.Dropout(p = self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
            input_size = input_size // 2
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Sequential(
            nn.Linear(input_size, 1, bias = False),
        )

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        for m in self.predict_layer:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)

        cat_emb = torch.cat((user_emb, item_emb), -1)

        output = self.MLP_layers(cat_emb)

        output = self.predict_layer(output)

        return output.view(-1)