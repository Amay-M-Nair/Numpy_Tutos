import numpy as np
"""
Neural Network from Scratch — Pure NumPy

A 2-layer fully connected network trained on MNIST.
Achieves ~97.7% test accuracy with no deep learning frameworks.

Architecture: 784 → 256 (ReLU) → 10 (Softmax)
Optimiser:    SGD with Momentum
Loss:         Cross-Entropy

"""
np.random.seed(42)

##-----SECTION 1: ACTIVATIONS-----------------

def relu(x):
    return np.maximum(0,x)

def relu_backward(grad_out, x):
    # Gradient of ReLU: 1 where x > 0, 0 where x <= 0
    # grad_out is the gradient flowing in from the next layer
    return grad_out*(x>0).astype(int) 

def softmax(x):
    x-= np.max(x,axis=-1,keepdims=True)
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x,axis=-1,keepdims=True)

##-----SECTION 2: LOSS FUNCTIONS--------------

def cross_entropy_loss(probs,labels):
    #Negative log likelihood
    batch_size = probs.shape[0]
    correct_probs = probs[np.arange(batch_size),labels] #Gathering prob for correct class
    loss = -np.log(correct_probs + 1e-8) #1e-8 for stability
    return np.mean(loss)

def cross_entropy_backward(probs,labels):
    # Combined gradient of softmax + cross entropy
    # d(Loss)/d(logits) = probs - one_hot(labels)
    batch_size = probs.shape[0]
    grad = probs.copy()
    grad[np.arange(batch_size),labels] -= 1 
    grad/=batch_size #average over batch

    return grad

##----SECTION 3: LAYER OPERATIONS------------

def linear_forward(x,W,b):
    # x: (batch, fan_in)
    # W: (fan_in, fan_out)
    # b: (fan_out,)
    # out: (batch, fan_out)
    return x@W + b

def linear_backward(grad_out,x, W):
    #gradients for x, W, and b , matrix calculus
    grad_x = grad_out @ W.T #passing further back
    grad_W = x.T @ grad_out #for updating W
    grad_b = np.sum(grad_out,axis=0) #sum over batch

    return grad_x,grad_W,grad_b

##-----SECTION 4: INITIALIZATION--------------

def kaiming_init(fan_in,fan_out):
    #For ReLU networks
    std = np.sqrt(2.0/fan_in)

    return np.random.randn(fan_in,fan_out)*std

def init_network(layer_size):
    params = {}
    for i in range(len(layer_size)-1):
        fan_in = layer_size[i]
        fan_out = layer_size[i+1]
        params[f'W{i+1}']= kaiming_init(fan_in,fan_out)
        params[f'b{i+1}']= np.zeros(fan_out)
    return params


##-----SECTION 5: FORWARD PASS---------------

def forward(X,params):
    """
    X:      (batch, 784)
    params: dict with W1, b1, W2, b2
    
    Returns probs and cache.
    Cache stores everything needed for backprop.
    """
    #Layer 1
    z1 = linear_forward(X,params['W1'],params['b1'])
    a1 = relu(z1)

    #Layer 2
    z2 = linear_forward(a1,params['W2'],params['b2'])

    #Outputs
    probs = softmax(z2)

    cache = {
        'X': X,
        'z1':z1,
        'z2':z2,
        'a1':a1,
    }

    return probs, cache

def predict(X,params):
    #Returns predicted class for each sample
    probs, _ = forward(X,params)
    return np.argmax(probs,axis=1)

def accuracy(X, labels, params):
    preds = predict(X,params)
    return np.mean(preds==labels)

##------SECTION 6: BACKWARD PASS-----------------

def backward(probs, labels, cache, params):
    """
    probs:  (batch, 10)  — softmax output from forward pass
    labels: (batch,)     — true class labels
    cache:  dict         — saved values from forward pass
    params: dict         — current weights

    Returns grads dict with gradients for W1, b1, W2, b2
    """
    grads = {}

    # Combined softmax + cross-entropy gradient
    # gradient of loss w.r.t. z2, dL/dz2
    dz2 = cross_entropy_backward(probs,labels)

    # gradient through linear layer 2 
    # a1 was the input to layer 2, dL/da1, dL/dW2, dL/db2
    da1, grads['W2'], grads['b2'] = linear_backward(dz2, cache['a1'],params['W2'])

    # gradient through ReLU
    # z1 was the input to ReLU, dL/dz1
    dz1 = relu_backward(da1,cache['z1'])

    # gradient through linear layer 1
    # X was the input to layer 1, dL/dX, dL/dW1, dL/db1
    dX, grads['W1'], grads['b1'] = linear_backward(dz1, cache['X'], params['W1'])

    return grads

##------SECTION 7: OPTIMIZER------------------

def init_velocity(params):
    """Initialise momentum velocity to zeros — same shape as params."""
    velocity = {}
    for key in params:
        velocity[key] = np.zeros_like(params[key])
    return velocity

def sgd_momentum(params, grads, velocity, lr=0.01, momentum=0.9):
    """
    v = momentum * v - lr * grad
    W = W + v
    """
    for key in params:
        velocity[key] = momentum*velocity[key]- lr*grads[key]
        params[key] +=velocity[key]
    return params,velocity


##----SECTION 8: GRADIENT CHECK----------------

def gradient_check(params, X, labels, epsilon=1e-5):
    """
    Compare analytical gradients to numerical gradients.
    Should match to within 1e-5.
    """

    probs,cache = forward(X,params)
    grads = backward(probs,labels,cache,params)

    print("Gradient check (should be < 1e-5):")

    for param_name in ['W1', 'b1', 'W2', 'b2']:
        param = params[param_name]
        grad = grads[param_name]

        #for first five elements
        num_checks = 5
        max_error = 0
        for idx in range(num_checks):
            orig_val = param.flat[idx] #flatten to 1D

            param.flat[idx] = orig_val + epsilon #f(x+epsilon)
            probs_plus,_ = forward(X,params)
            loss_plus = cross_entropy_loss(probs_plus,labels)

            param.flat[idx] = orig_val - epsilon #f(x-epsilon)
            probs_minus,_ = forward(X,params)
            loss_minus = cross_entropy_loss(probs_minus,labels)

            param.flat[idx] = orig_val #restore

            numerical_grad = (loss_plus-loss_minus)/(2*epsilon)  #Numerical gradient
            analytical_grad = grad.flat[idx]                     #Analytical gradient

            error = abs(numerical_grad-analytical_grad)
            max_error = max(max_error,error)
        print(f"{param_name}: max error = {max_error:.2e}")


##---------SECTION 9: TRAINING------------------

def train(X_train, y_train, X_val, y_val,
          layer_sizes = [784,256,10],
          lr=0.01, momentum = 0.9,
          epochs = 20, batch_size = 64,
          print_every = 1):
    #Initialize
    params = init_network(layer_sizes)
    velocity = init_velocity(params)
    n = X_train.shape[0]

    print("Training....")

    history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}
    for epoch in range(epochs):
        idx = np.random.permutation(n) #shuffle training data each epoch
        X_train = X_train[idx]
        y_train = y_train[idx]

        epoch_loss = 0
        n_batches = 0

        #Mini-batch loop
        for i in range(0,n,batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            #Forward
            probs, cache = forward(X_batch, params)
            loss = cross_entropy_loss(probs,y_batch)

            #Backward
            grads = backward(probs, y_batch, cache, params)

            #Update
            params, velocity = sgd_momentum(params, grads, velocity,lr, momentum)

            epoch_loss+=loss
            n_batches +=1

        #Epoch metrics
        avg_loss = epoch_loss/n_batches
        val_probs,_ = forward(X_val,params)
        val_loss = cross_entropy_loss(val_probs, y_val)
        val_acc = accuracy(X_val, y_val, params)

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % print_every == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"train loss: {avg_loss:.4f} | "
                  f"val loss: {val_loss:.4f} | "
                  f"val acc: {val_acc:.4f}")
            
    return params, history

##-----------------OVERFIT TEST FOR SANITY----------------------
"""
Starts at 2.3149
Accuracy hits near 1.00
Loss decreases smoothly
No spikes, no instability
"""
np.random.seed(42)
params   = init_network([784, 256, 10])
velocity = init_velocity(params)
X_tiny   = np.eye(10, 784)
y_tiny   = np.arange(10)

print("Gradient Check....")
for step in range(200):
    probs, cache = forward(X_tiny, params)
    loss = cross_entropy_loss(probs, y_tiny)
    grads = backward(probs, y_tiny, cache, params)
    params, velocity = sgd_momentum(params, grads, velocity, lr=0.01)
    if step == 199 or step == 0:
        print(f"Step {step:3d} | loss: {loss:.4f} | "
              f"acc: {accuracy(X_tiny, y_tiny, params):.2f}")
        

##----------FINAL PROJECT: MNIST---------------
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

print("Loading MNIST...")
mnist = fetch_openml('mnist_784',version=1,as_frame=False,parser='auto')
X,y = mnist.data, mnist.target.astype(int)

X= X/255.0 #normalize pixels to [0,1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=10000,random_state=42) #standard split

print(f"X_train: {X_train.shape} y_train:{y_train.shape}")
print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}")
print(f"Pixel range: [{X_train.min():.1f},{X_train.max():.1f}]")
print(f"Classes:{np.unique(y_train)}")

np.random.seed(42)

params, history = train(
    X_train, y_train,
    X_test,  y_test,
    layer_sizes = [784, 256, 10],
    lr          = 0.01,
    momentum    = 0.9,
    epochs      = 20,
    batch_size  = 64,
    print_every = 1
) #training

test_acc = accuracy(X_test, y_test, params)
print(f"\nFinal test accuracy: {test_acc:.4f}") #final accuracy

print("\nPer-class accuracy:")
for digit in range(10):
    mask = y_test == digit
    class_acc = accuracy(X_test[mask],y_test[mask],params)
    bar = '█' * int(class_acc*100)
    print(f"Digit {digit}:{class_acc:.4f} {bar}") #Per class accuracy



