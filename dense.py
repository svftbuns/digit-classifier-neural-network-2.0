# Fully-connected layer
class Dense(Layer):
    def __init__(self,input_units,output_units, activation):
        # randn: Generates random numbers from a standard normal distribution
        # Scaled by sqrt(2/input_units) also known as Xavier initialization
        self.weights=np.random.randn(output_units,input_units) / np.sqrt(2 / input_units)
        self.input_units=input_units
        # Second parameter 1 represents the number of columns in the output array
        self.bias=np.random.randn(output_units,1)
        self.activ_func=None
        
        # Figure out which activation is used
        if activation=='Sigmoid':
            self.activ_func=Sigmoid()
        elif activation=='Softmax':
            self.activ_func=Softmax()           
        
    def forward_prop(self,input):
        self.original_shape=input.shape
        self.input_flattened=input.flatten()
        self.input_flattened=self.input_flattened.reshape(self.input_units,1)
        
        Z1=np.dot(self.weights,self.input_flattened)+self.bias
        return self.activ_func.forward_prop(Z1)
    
    def backward_prop(self,output_gradient,alpha=0.01):
        activ_gradient=self.activ_func.backward_prop(output_gradient,alpha)
        weights_gradient=np.dot(activ_gradient,self.input_flattened.T)
        input_gradient=np.dot(self.weights.T,activ_gradient)
        
        self.weights-=alpha*weights_gradient
        self.bias-=alpha*activ_gradient
        
        input_gradient=np.reshape(input_gradient,self.original_shape) # Reshape to feed into the previous layer
        return input_gradient   
