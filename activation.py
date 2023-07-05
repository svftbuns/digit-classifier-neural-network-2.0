# Define individual activation layers
# ReLU layer
class ReLU(Layer):
    
    def forward_prop(self, input):
        self.input=input
        return np.maximum(0,input)
    
    # Compute the gradient of the loss function wrt input of the activation layer
    # output_gradient: Gradient of lost wrt output of the activation layer, passed from subsequent layer
    def backward_prop(self, output_gradient,alpha):
        return np.multiply(output_gradient, self.input>0)
    
# Sigmoid Layer
class Sigmoid(Layer):
    
    def forward_prop(self,input):
        self.input=input
        return 1/(1+np.exp(-input))
    
    def backward_prop(self,output_gradient,alpha):
        sigmoid=self.forward_prop(self.input)
        sigmoid_derivative=sigmoid*(1-sigmoid)
        return np.multiply(output_gradient,sigmoid_derivative)
    
# Softmax Layer
class Softmax(Layer):
    
    # log-sum-exp trick to avoid overflow issues when exponentiating large numbers
    def forward_prop(self,input):
        max_value = np.max(input)
        shifted_input = input - max_value
        top = np.exp(shifted_input)
        self.output = top / np.sum(top)
        return self.output
    
    # When used in conjuction with cross-entropy loss, derivative of softmax is just the output gradient
    def backward_prop(self,output_gradient,alpha):
        return output_gradient

        """n = np.size(self.output)
        matrix=np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        return np.multiply(output_gradient,matrix)"""
        
