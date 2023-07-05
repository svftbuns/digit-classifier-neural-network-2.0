# Dropout layer: Regularization technique which stochastically sets a fraction of the input values to zero
# Helps to prevent overfitting and improve generalization ability of the model
class Dropout(Layer):
    def __init__(self,dropout_rate):
        self.dropout_rate=dropout_rate
        self.mask=None
        
    def forward_prop(self,input):
        # Generates an array of random numbers drawn from a binomial distribution with same shape as the input
        # If value on mask < dropout_rate, set to 0 otherwise set to 1
        self.input=input
        self.mask=np.random.binomial(1,1-self.dropout_rate,size=input.shape)/(1-self.dropout_rate)
        output=self.input*self.mask
        return output
    
    # Multiply element-wise with mask
    # Ensure dropped out elements do not contribute to the gradient
    def backward_prop(self,output_gradient,learning_rate):
        input_gradient=output_gradient*self.mask
        return input_gradient
