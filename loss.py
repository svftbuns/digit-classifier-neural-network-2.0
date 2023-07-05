# Cross-entropy loss for multi-class classification
# Tells how correct a particular prediction is
def cross_entropy_loss(y_pred,y_true):
    epsilon=1e-10 # Small constant to avoid division by zero
    num_examples=y_pred.shape[0]
    log_probabilities=-np.log(y_pred+epsilon)*y_true  # y_true is hot-one encoded
    loss=np.mean(log_probabilities) # Average loss
    return loss


def cross_entropy_loss_prime(y_pred,y_true): # y_true is one-hot encoded
    return y_pred-y_true
    
