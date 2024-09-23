class Flatten():

    def __init__(self):
        self._parameters = None
        
    
    def zero_grad(self):
        pass

    def forward(self, X):
        """
        Args :
            X:(batch,input,chan_in)
            out:(batch,input*chan_in)
        """
        return X.reshape(X.shape[0], -1)
        

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        """
        Args :
            input: (batch,input,chan_in)
            delta: (batch, input * chan_in)
            out: (batch,input,chan_in)
        """
        self._delta = delta.reshape(input.shape)
        return self._delta
    
    def backward_update_gradient(self, input, delta):
        pass
    