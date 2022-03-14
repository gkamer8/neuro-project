import torch

class Reservoir():
    # Note: std originally .05
    def __init__(self, Nu=2_352, Nx=5_000, density=.2, activation=torch.tanh, std=.1, batch_size=128):

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
        self.W = self.init_w(density, std)  # Note: std is standard deviation of weight value
        self.Win = self.init_win()

    def init_w(self, density, std=.5):
        # Get random connections
        probs = torch.tensor([density]).repeat((self.Nx, self.Nx))
        connections = torch.bernoulli(probs)  # Tensor of 1s and 0s

        # Get random weights
        std_matrix = torch.tensor([std]).repeat((self.Nx, self.Nx))
        weights = torch.normal(mean=0, std=std_matrix)
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