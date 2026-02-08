import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time

start = time.time()

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

print('\n-------------- DATA SHAPE ----------------')
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
print('------------------------------------------\n')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: USING {device} DEVICE\n')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

neural_net = NeuralNetwork()
model = neural_net.to(device)
print('---------------------- MODEL ATTRIBUTES ---------------------')
print(model)
print('-------------------------------------------------------------\n')

loss_fn = nn.CrossEntropyLoss() # the loss function --------------------------------------------
optimizer = torch.optim.SGD(model.parameters(), lr=5e-1) # change learning rate here ----------*

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 4 # change num of epochs here --------------------------------
for t in range(epochs):
    print(f"Epoch {t+1}:\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

end = time.time()
time_elapsed = end-start
print(f'Time spent: {time_elapsed} seconds')

# r = torch.randn(2, 3, requires_grad=True)
# r1 = torch.rand(2, 3)
# print(r, r1, sep='\n')

# r = torch.sin(r)

# print(r.grad_fn)

# torch.manual_seed(1)
# ones = torch.ones(5, 2, 3)
# twos = torch.rand(1, 3)*2
# ans = ones*twos
# print(ans)

# for i in ans:
#     for j in i:
#         print(i, j)

# array = np.ones(5, dtype=np.int16)
# print(array)

# array = np.ones(4)
# print(array)

# ex = pd.DataFrame([['a', 'b', 'c', 'd']], [1], [1, 2, 3, 4])
# print(ex)



# class C {
#     private:
#         int x = 0;
#     public:
#         void run(int x);
# }

# #include "C.h"
# C::run(int x) {
#     std::cout << ''
# }
# C x = new C()

# public class Example {
#     nums = []
#     public void main (String args[]) {
#     }

#     public Example(int nums[]) {
#         self.nums = nums;
#     }

#     private int sum(list[]) {
#         ans = 0;
#         for (int i = 0; i < list.size(); i++) {
#             ans += sum[i];
#         }
#         for int i : self.nums {
#             ans += i;
#         }
#         return ans;
#     }
# }
# Example ex = new Example([1, 2, 3]);