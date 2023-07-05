# Max pooling layer: Reduces the spatial dimensions of the input while preserving the most salient features 
class MaxPooling(Layer):
    # pool_size: Downsized dimensions
    # stride: step size used to slide the pooling window across the input feature map
    def __init__(self,pool_size, stride=1,padding=0):
        self.pool_size=pool_size
        self.stride=stride
        self.padding=padding
        self.input=None
        self.output=None
    
    def forward_prop(self,input):
        self.input=input
        depth,height,width=input.shape
        
        # Padding: add extra border values, usually zeros, around input data before pooling
        # Useful when we want to retain the spatial size of the input feature map or when the input size is not evenly divisible by the stride
        padded_input=np.pad(input,((0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant')
        
        pooled_depth=depth
        pooled_height=(height-self.pool_size+2*self.padding) // self.stride + 1
        pooled_width=(width-self.pool_size+2*self.padding) // self.stride + 1
        
        self.output=np.zeros((pooled_depth,pooled_height,pooled_width))
        
        # Perform MaxPooling
        for d in range(pooled_depth):
            for h in range(pooled_height):
                for w in range(pooled_width):
                    z_start=d
                    z_end=z_start+1
                    y_start=h*self.stride
                    y_end=y_start+self.pool_size
                    x_start=w*self.stride
                    x_end=x_start+self.pool_size
                    region=padded_input[z_start:z_end,y_start:y_end,x_start:x_end]
                    self.output[d,h,w]=np.max(region)
        return self.output
    
    
    def backward_prop(self,output_gradient,alpha):
        pooled_depth,pooled_height,pooled_width=self.output.shape
        input_gradient=np.zeros_like(self.input)
        padded_output_gradient=np.pad(output_gradient,((0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant')
        
        for d in range(pooled_depth):
            for h in range(pooled_height):
                for w in range(pooled_width):
                    z_start=d
                    z_end=z_start+1
                    y_start=h*self.stride
                    y_end=y_start+self.pool_size
                    x_start=w*self.stride
                    x_end=x_start+self.pool_size
                    region=self.input[z_start:z_end,y_start:y_end, x_start:x_end]
                    # Find the indices of the maximum value within a region of input
                    # np.argmax(region): Finds the index of the maximum value within the flattened region array
                    # np.unravel_index: Converts a flattened index to its corresponding multidimensional indices based on the shape of the array
                    max_idx=np.unravel_index(np.argmax(region), region.shape)
                    input_gradient[z_start:z_end,y_start:y_end,x_start:x_end][max_idx]=padded_output_gradient[d,h,w]
                    
        return input_gradient      
