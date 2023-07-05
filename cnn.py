# ConvolutionalNN class
# Include calculation for execution time
import time

class ConvolutionalNN:
    def __init__(self,layers):
        self.layers=layers
        
    def train(self, x_train,y_train,validation,epochs=20,learning_rate=0.01, verbose=True):
        # Include calculations for training loss and validation accuracy
        train_loss=[]
        val_accuracy=[]
        
        for i in range(1,epochs+1):
            start_time=time.time() # Start the timer
            train_l=0 # Initialize training loss to zero first
            for x,y in zip(x_train,y_train):
                output=x
                for layer in self.layers:
                    output=layer.forward_prop(output)
                
                one_hot_y=self.one_hot_encode(y,10)
                grad=output-one_hot_y
                
                train_l+=cross_entropy_loss(output,one_hot_y)
    
            
                for layer in reversed(self.layers):
                    grad=layer.backward_prop(grad,learning_rate)
            
            # Calculating validation accuracy and error
            val_acc=0
            for x_val, y_val in zip(validation[0], validation[1]):
                val_output = self.predict(x_val)
                if self.get_label(val_output)==y_val:
                    val_acc+=1
            
            train_loss.append(train_l/len(x_train)) # Append mean training error
            val_accuracy.append(val_acc/len(validation[0])) # Append mean validation accuracy
             
            end_time=time.time() # End the timer
            execution_time=round(((end_time-start_time)/60),4)
        
            if verbose:
                print(f'Epoch {i}/{epochs}, Train Loss={train_loss[-1]}, Val Accuracy={val_accuracy[-1]}, Time Elapsed={execution_time} mins')

        return train_loss, val_accuracy
    
    def predict(self, input):
        output=input
        for layer in self.layers:
            output=layer.forward_prop(output)
            
        return output
    
    def get_label(self,Y):
        # Finds the class with the highest probability
        # Return the index of the class, 0-9
        return np.argmax(Y,0)

    # Convert output values as a single-column matrix with 10 rows, each corresponding to the possible digits 0-9
    def one_hot_encode(self, arr, num_classes=10):
        one_hot = np.zeros((num_classes, 1))
        one_hot[arr] = 1
        return one_hot
    
        
