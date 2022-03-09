from torch import nn

class Softmax_Reg( nn.Module ):
    def __init__( self, n_features,n_classes ):
        super(Softmax_Reg, self).__init__()
        #self.linear1 = nn.Linear(n_features,n_features//2)
        #self.sigmoid = nn.Sigmoid()
        #self.linear2 = nn.Linear(n_features//2,n_classes)
        #self.softmax = nn.Softmax()

        self.mlp = nn.Sequential(
            nn.Linear( n_features, n_features*2),
            nn.Sigmoid(),
            nn.Linear( n_features*2, n_features ),
            nn.Sigmoid(),
            nn.Linear( n_features, n_classes),
            nn.Softmax()
        )

    def forward( self, x ):
        # y = self.linear1(x)
        # y = self.sigmoid(y)
        # y = self.linear2(y)
        # y = self.softmax(y)
        y = self.mlp(x)
        return y