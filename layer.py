# Build a general Layer class, specific layers will have to inherit from this class
class Layer: 
    def __init__(self):
        self.input=None
        self.output=None
    
    # Inherited classes will have to implement functions for forward and backward propagation
    def forward_prop(self,input):
        # Returns output
        pass 
    
    def backward_prop(self,output_gradient, alpha=0.01):
        # Update parameters and return input gradient
        pass
