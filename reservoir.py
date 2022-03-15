import torch

class Reservoir():
    # Note: std originally .05
    def __init__(self, Nu=2_352, Nx=5_000, density=0.001, activation=torch.tanh, std=.5, batch_size=128):

        # Neurons
        self.x = torch.zeros((Nx,batch_size))
        # Activation function
        self.activation = activation
        # Number of inputs
        self.Nu = Nu
        # Number of internal neurons
        self.Nx = Nx
        # Batch size
        self.batch_size = batch_size
        # Weights - same for all the batches
        self.W = self.init_w(density, std=std)  # Note: std is standard deviation of weight value
        self.Win = self.init_win()

    """
    method: type of weight initialization
        'uniform' -> uniformly selected b/w -1 and 1
        'normal' -> normally distributed w mean 0 and std = std (.5 by default)
        'binary' -> weights are either 1 or -1
    """
    def init_w(self, density, std=.5, method="uniform"):

        # Get random connections
        probs = torch.tensor([density]).repeat((self.Nx, self.Nx))
        connections = torch.bernoulli(probs)  # Tensor of 1s and 0s

        if method == 'normal':
            std_matrix = torch.tensor([std]).repeat((self.Nx, self.Nx))
            weights = torch.normal(mean=0, std=std_matrix)
        elif method == 'uniform':
            weights = torch.rand((self.Nx, self.Nx)) * 2 - 1  # From -1 to 1
        elif method == 'binary':
            halves = torch.ones((self.Nx, self.Nx)) * .5
            weights = torch.bernoulli(halves) * 2 - 1  # Everything is either -1 or 1 with .5 probability

        weights = weights * connections  # Note: element wise multiplication
        return weights

    def init_win(self):
        diag_square = torch.diag(torch.ones((self.Nu,)))
        try:
            filler = torch.zeros((self.Nx-self.Nu,self.Nu))
        except RuntimeError:
            print("Input length longer than size of reservoir; error creating W_in")
            exit(1)
        win = torch.vstack((diag_square, filler))
        return win

    def get_states(self, transpose=True):
        return self.x

    # Move forward one timestep with input tensor u
    def evolve(self, u):
        # Implementing:
        # x(n) = f(Win u(n) + W x(n − 1))

        # ASSUMING u IS SHAPE (batch size, Nu)
        new_u = torch.transpose(u, 0, 1)
        winu = torch.matmul(self.Win, new_u)
        wx = torch.matmul(self.W, self.x)
        self.x = self.activation(winu + wx)

    def clear(self):
        self.x = torch.zeros((self.Nx,self.batch_size))