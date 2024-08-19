import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_user, num_item, num_factor):
        super(GMF, self).__init__()
        self.user_emb = nn.Embedding(num_user, num_factor)
        self.item_emb = nn.Embedding(num_item, num_factor)

        self.predict_layer = nn.Sequential(
            nn.Linear(num_factor, 1, bias = False)
        )

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for m in self.predict_layer:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)

        output = self.predict_layer(user_emb * item_emb)

        return output.view(-1)