import torch
import torch.nn as nn

class Point2Views(nn.Module):
    def __init__(self,in_features,dim) -> None:
        super().__init__()
        self.in_features = in_features+dim    # 512+3
        self.dropout = 0.075

        self.project = nn.Sequential(
            #nn.Linear(in_features+dim, in_features),
            nn.BatchNorm1d(self.in_features),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.in_features ,
                        out_features= in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features= in_features, out_features= in_features)

        )

    def forward(self,point_feat,a,e,d):
        #print(point_feat.shape,a.shape,e.shape,d.shape)
        # todo   a e d normalization
        feat = torch.cat((point_feat,a,e,d),dim=1)  # 16*(512+3)
        #print(feat.shape)
        out = self.project(feat)
        #print(out.shape)
        return out


