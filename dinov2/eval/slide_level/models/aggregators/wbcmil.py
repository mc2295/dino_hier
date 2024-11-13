import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aggregators.aggregator import BaseAggregator

class WBCMIL(BaseAggregator):
    def __init__(self, input_dim, num_classes, multi_attention=True, attention_latent_dim=128):
        super(BaseAggregator, self).__init__()

        self.mil_latent_dim = input_dim
        self.attention_latent_dim = attention_latent_dim
        self.num_classes = num_classes
        self.multi_attention = multi_attention

        # single attention network
        self.attention = nn.Sequential(
            nn.Linear(self.mil_latent_dim, self.attention_latent_dim),
            nn.Tanh(),
            nn.Linear(self.attention_latent_dim, 1)
        )

        # multi attention network
        self.attention_multi_column = nn.Sequential(
            nn.Linear(self.mil_latent_dim, self.attention_latent_dim),
            nn.Tanh(),
            nn.Linear(self.attention_latent_dim, self.num_classes)
        )

        # single head classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.mil_latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )

        # multi head classifier
        self.classifier_multi_column = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.mil_latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(self.num_classes)
        ])

    def forward(self, x):
        prediction = []
        bag_feature_stack = []

        # Assuming features come from some external encoder
        features = x  # Uncomment if needed: self.mil_encoder(x.squeeze().unsqueeze(dim=2))
        
        attention = torch.transpose(self.attention_multi_column(features), 1, 0)

        if self.multi_attention:
            for cls in range(self.num_classes):
                # Multi-head attention aggregation
                att_softmax = F.softmax(attention.T[cls, ...], dim=-1)
                bag_features = torch.mm(att_softmax, features.squeeze())

                bag_feature_stack.append(bag_features)

                # Multi-head classification
                pred = self.classifier_multi_column[cls](bag_features)
                prediction.append(pred)

            prediction = torch.stack(prediction).view(1, self.num_classes)
            bag_feature_stack = torch.stack(bag_feature_stack).squeeze()

            # return {
            #     "prediction": prediction,
            #     "attention": attention,
            #     "att_softmax": att_softmax,
            #     "bag_features": bag_features
            # }
            return prediction

        else:
            # Single-head attention aggregation
            att_softmax = F.softmax(attention, dim=1)
            bag_features = torch.mm(att_softmax, features)

            # Single-head classification
            prediction = self.classifier(bag_features)

            return {
                "prediction": prediction,
                "attention": attention,
                "att_softmax": att_softmax,
                "bag_features": bag_features
            }
