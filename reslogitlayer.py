from pycmtensor.expressions import Weights
from pycmtensor import aet as aet

class ResLogitLayer:
    def __init__(self, u, w_in, w_out, activation=None):
        
        assert w_in.shape[1].eval() == w_out.shape[0].eval()
        assert isinstance(w_in, (Weights)), "w_in must be of type Weights"
        assert isinstance(w_out, (Weights)), "w_out must be of type Weights"
        
        if isinstance(u, (list, tuple)):
            assert len(u) == w_in.shape[0].eval(), f"index.0 of w_in must be of the same length as u"
            self.U = aet.stacklists(u).flatten(2)
        else:
            self.U = u

        self.w_in = w_in()
        self.w_out = w_out()
        if activation == None:
            activation = aet.sigmoid
            
        h = activation(aet.dot(self.U.T, self.w_in))
        output = activation(aet.dot(h, self.w_out)).T
        self.params = [self.w_in, self.w_out]
        self.output = output + self.U

    
    def __repr__(self):
        return f"ResLogitLayer([{self.w_in.shape.eval()}, {self.w_out.shape.eval()}])"