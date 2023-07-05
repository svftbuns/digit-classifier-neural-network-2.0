# Define individual activation layers 
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

    def backward_prop(self,output_gradient,alpha):
        n = np.size(self.output)
        matrix=np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        return np.multiply(output_gradient,matrix)
        
