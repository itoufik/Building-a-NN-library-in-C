import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define constants
BITS = 4
RATE = 1e-1
TRAIN_EPOCHS = 70000

# Define the Neural Network model with one hidden layer
class AdderNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2 * BITS, 2 * BITS + 1)
        self.fc2 = nn.Linear(2 * BITS + 1, BITS + 1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x
    
def main():
    s = time.time()
    n = 1 << BITS      # n = 2^BITS
    rows = n * n       # total number of examples

    # Create tensors for inputs (X) and targets (Y)
    X = torch.zeros((rows, 2 * BITS))
    Y = torch.zeros((rows, BITS + 1))
    for i in range(rows):
        x_val = i // n
        y_val = i % n
        z_val = x_val + y_val
        overflow = int(z_val >= n)
        for j in range(BITS):
            X[i, j] = (x_val >> j) & 1
            X[i, j + BITS] = (y_val >> j) & 1
            # If there's overflow, target bits are forced to 0
            Y[i, j] = 0 if overflow else (z_val >> j) & 1
        Y[i, BITS] = overflow
    # Initialize the network
    model = AdderNN()
    optimizer = optim.SGD(model.parameters(), lr=RATE)
    criterion = nn.MSELoss()

    # Training loop using full gradient descent (full batch each update)
    for epoch in range(TRAIN_EPOCHS):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0 or epoch == TRAIN_EPOCHS - 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # Evaluation: Test the network on every combination of x and y in [0, n-1]
    with torch.no_grad():
        for x_val in range(n):
            for y_val in range(n):
                input_vec = torch.zeros((1, 2 * BITS))
                for j in range(BITS):
                    input_vec[0, j] = (x_val >> j) & 1
                    input_vec[0, j + BITS] = (y_val >> j) & 1
                
                out = model(input_vec)
                # Check for overflow: the overflow bit is at index BITS
                if out[0, BITS] > 0.5:
                    print(f"{x_val} + {y_val} = OVERFLOW")
                else:
                    z = 0
                    for j in range(BITS):
                        bit = 1 if out[0, j] > 0.5 else 0
                        z |= bit << j
                    print(f"{x_val} + {y_val} = {z}")

    e = time.time()
    print(f"Time of execution {e -s}")

if __name__ == "__main__":
    main()

