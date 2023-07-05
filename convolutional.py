from scipy import signal

# Now define a Convolutional Layer that inherits the Layer class
class Convolutional(Layer):
    # input_shape: Tuple containing the depth, height and width of input
    # kernel_size: Size of matrix inside each kernel
    # depth: Number of kernel thus is the depth of the output
    def __init__(self, input_shape, kernel_size, depth, activation):
        input_depth, input_height,input_width=input_shape # Three-dimensional
        self.depth=depth
        self.input_shape=input_shape
        self.input_depth=input_depth
        self.output_shape=(depth, input_height-kernel_size+1,input_width-kernel_size+1) # Four-dimensional, follows shape of valid correlation
        
        # Define shape of the kernel tensor
        self.kernel_shape=(depth,input_depth,kernel_size,kernel_size) 
        
        self.kernel=np.random.randn(*self.kernel_shape) / (kernel_size**2) # Divide by (kernel_size**2) for weights normalization
        self.biases=np.random.randn(*self.output_shape) # Random numbers with mean of 0 and S.D of 1
        self.activ_func=None
        
        # Figure out which activation is used
        if activation=='Sigmoid':
            self.activ_func=Sigmoid()
        elif activation=='Softmax':
            self.activ_func=Softmax() 

    
    def forward_prop(self,input):
        self.input=input
        self.output=np.copy(self.biases) # Same shape as biases
        # Loop through the number of kernels
        for i in range(self.depth):
            for j in range(self.input_depth): 
                self.output[i]+=signal.correlate2d(self.input[j], self.kernel[i,j],"valid") # Cross-correlation
            self.output[i]+=self.biases[i] # Add the bias, bias and output have the same shape
        return self.activ_func.forward_prop(self.output) # Pass output through activation function
    
    
    def backward_prop(self,output_gradient,alpha=0.01): # Output_gradient is deriv of E wrt output Y
        output_gradient=self.activ_func.backward_prop(output_gradient,alpha)
        kernel_gradient=np.zeros(self.kernel_shape) # Create a multi-dimensional array filled with zeros
        input_gradient=np.zeros(self.input_shape)
        
        # output_gradient: deriv of E wrt to output Y
        # kernel_gradient: deriv of E wrt to kernels -> Valid correlation of input X and output_gradient
        # input_gradient: deriv of E wrt to input X -> Full convolution of output_gradient and kernels
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernel_gradient[i,j]=signal.correlate2d(self.input[j],output_gradient[i],"valid")
                input_gradient[j]+=signal.convolve2d(output_gradient[i], self.kernel[i,j], "full")  
                
        # Update the kernel and bias values
        # bias_gradient: deriv of E wrt to biases -> Equals to output_gradient
        self.kernel-=alpha*kernel_gradient
        self.biases-=alpha*output_gradient
        
        return input_gradient
    
    
    '''def correlate2d(self,input,kernel,mode='valid'): #Default mode is valid
        self.input=input.reshape(self.input_shape)
        input_depth,input_height,input_width=self.input.shape
        kernel_depth,kernel_height,kernel_width=1,self.kernel.shape
        
        if mode=='valid':
            output_height=input_height-kernel_height+1
            output_width=input_width-kernel_width+1
        elif mode=='same':
            output_height=input_height
            output_width=input_width
        elif mode=='full':
            output_height=input_height+kernel_height-1
            output_width=input_width+kernel_width-1
        else:
            raise ValueError("Invalid mode. Only 'valid','same' and 'full' modes are supported")
        
        output=np.zeros((output_height,output_width)) # Initialize output matrix with zeros
        
        # Perform 2D correlation
        for i in range(output_height):
            for j in range(output_width):
                output[i,j]=np.sum(input[:,i:i+kernel_height,j:j+kernel_width]*kernel)
                
        return output
    
    
    def convolve2d(self,input,kernel, mode='valid'):
        input_depth,input_height,input_width=input.shape
        kernel_depth,kernel_height,kernel_width=kernel.shape
        
        if mode=='valid':
            output_height=input_height-kernel_height+1
            output_width=input_width-kernel_width+1
        elif mode=='same':
            output_height=input_height
            output_width=input_width
        elif mode=='full':
            output_height=input_height+kernel_height-1
            output_width=input_width+kernel_width-1
        else:
            raise ValueError("Invalid mode. Only 'valid','same' and 'full' modes are supported")
            
        output=np.zeros((output_height,output_width))
        
        # Flip the kernel along both axes, 180 degrees rotation
        # axis=1: Elements within each row are reversed
        # axis=2: Elements within each column are reversed
        flipped_kernel=np.flip(np.flip(kernel,axis=1),axis=2)
        
        # Perform 2D convolution
        for i in range(output_height):
            for j in range(output_width):
                output[i,j]=np.sum(input[:, i:i+kernel_height, j:j+kernel_width]*flipped_kernel)
                
        return output'''
