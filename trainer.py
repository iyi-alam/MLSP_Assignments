import torch
from tqdm import tqdm
import os

class Accuracy:
    def accuracy_bce(preds, targets):
        preds = (preds > 0).to(torch.long)
        targets = targets.to(torch.long)
        acc = torch.sum(preds == targets)
        return acc.item()
    
    def accuracy_ce(preds, targets):
        preds = torch.argmax(preds, dim = 1)
        acc = torch.sum(preds == targets)
        return acc.item()

class Trainer:
    def __init__(self, model, criterion, optimizer, accuracy, save_dir, scheduler = None, use_tqdm = True):

        self.model            = model
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.criterion        = criterion
        self.accuracy = accuracy
        self.loss_history     = { 'train': [], 'val': [] }
        self.acc_history      = { 'train': [], 'val': [] }
        self.device = model.device
        self.use_tqdm = use_tqdm
        self.save_dir = save_dir

    # def accuracy(self, preds, targets):
    #     #preds = torch.argmax(preds, dim = 1)
    #     #probs = torch.nn.Sigmoid()(preds)
    #     preds = (preds > 0).to(torch.long)
    #     targets = targets.to(torch.long)
    #     acc = torch.sum(preds == targets)
    #     return acc.item()

    def train_step(self, train_loader):
        self.model.train()
        train_loss = 0.0
        train_acc = 0
        samples = 0
        generator = tqdm(train_loader, desc = 'Processing Training Batch...') if self.use_tqdm else train_loader
        for x, y in generator:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.model(x)
            loss = self.criterion(preds,y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_acc += self.accuracy(preds, y)
            samples += len(x)
        
        if self.scheduler is not None:
            self.scheduler.step()
        train_loss /= len(train_loader)
        train_acc /= samples
        return train_loss, train_acc
    
    def eval_step(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_acc = 0
        samples = 0
        generator = tqdm(val_loader, desc = 'Processing Val Batch...') if self.use_tqdm else val_loader
        with torch.no_grad():
            for x,y in generator:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x)
                val_loss += self.criterion(preds, y).item()
                val_acc += self.accuracy(preds, y)
                samples += len(x)
            
            val_loss /= len(val_loader)
            val_acc /= samples
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, num_epochs):
        best_acc = 0
        model_path = os.path.join(self.save_dir, 'best.pth')
        for epoch in range(num_epochs):

            # train step
            train_loss, train_acc = self.train_step(train_loader)
            self.loss_history['train'].append(train_loss)
            self.acc_history['train'].append(train_acc)

            # Val step
            val_loss, val_acc = self.eval_step(val_loader)
            self.loss_history['val'].append(val_loss)
            self.acc_history['val'].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), model_path)


            print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\
                   | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        

        save_metric = dict(
            loss = self.loss_history,
            acc = self.acc_history,
            model_path = model_path,
        )
            
        return save_metric
