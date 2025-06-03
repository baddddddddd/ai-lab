import torch

class Trainer:
    def __init__(self, train_dataloader, test_dataloader, model, loss_fn, optimizer):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer


    def fit(self, epochs, verbose=True):
        for t in range(epochs):
            print(f"Epoch {t + 1}\n----------------------------------")
            self.train()
            loss, accuracy = self.test()


    def train(self, verbose=True):
        self.model.train()
        size = len(self.train_dataloader.dataset)

        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if verbose:
                if batch % 100 == 0:
                    loss_val, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")


    def test(self, verbose=True):
        self.model.eval()
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        correct = 0
        total_loss = 0.0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                total_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        average_loss = total_loss / num_batches
        accuracy = correct / size

        if verbose:
            print(f"Test Error: \n Accuracy: {accuracy * 100:>0.2f}%, Avg Loss: {average_loss:>8f}\n")

        return average_loss, accuracy