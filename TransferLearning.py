import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import analysis 

def train_mlp(name, model, train_data, test_data, params):
    train_loss = np.zeros(params['n_epochs'])
    test_loss = np.zeros(params['n_epochs'])

    train_loader = DataLoader(analysis.MyDataSet(train_data[0], train_data[1]), 
                            batch_size=params['batch_size'], 
                            shuffle=True)

    optimizer = optim.Adam([
        {'params': model.mlp.parameters(), 'lr': params['learning_rate']}
    ])
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                patience=3, threshold=0.01)
    criterion = nn.MSELoss()
    
    patience = 5  
    min_delta = 1e-4  
    best_test_loss = float('inf')
    patience_counter = 0
    final_epoch = len(train_loss)
    
    # Training loop
    for epoch in range(params['n_epochs']):
        model.train()
        epoch_loss = 0
        
        for X, y in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(X)
            loss = criterion(pred, y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            
        train_loss[epoch] = epoch_loss/len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_pred = model(test_data[0])
            test_loss[epoch] = criterion(test_pred, test_data[1]).item()
        
        scheduler.step(test_loss[epoch])
        print(f"Epoch {epoch+1}: Train Loss={train_loss[epoch]:.4f}, Test Loss={test_loss[epoch]:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'learning_rates': {
                'mlp': params['learning_rate']
            }
        }, f'Models/Transferred_{name}.pth')
        
        # Early stopping check
        if epoch > 2 and abs(test_loss[epoch]-test_loss[epoch-1]) < min_delta:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Converged: Loss hasn't improved by {min_delta} for {patience} iterations")
                final_epoch = epoch + 1
                break
        else:
            patience_counter = 0

    return train_loss, test_loss, final_epoch


def main():
    for name in ['bonde', 'kuo_arti', 'kuo_dmsc', 'kosuri']:
        print(name)

        X_train, X_test, y_train, y_test = analysis.Create_Model_Input(dataset = name, simple_onehot = True)
        
        tX_train = torch.from_numpy(X_train).to(torch.float32)
        tX_test = torch.from_numpy(X_test).to(torch.float32)
        tY_train = torch.from_numpy(y_train).to(torch.float32).reshape(-1, 1)
        tY_test = torch.from_numpy(y_test).to(torch.float32).reshape(-1, 1)
        
        # Load the checkpoint
        checkpoint = torch.load('Models/Backbone_fepB.pth', map_location=torch.device('cpu'), weights_only=False)
        model = analysis.HybridModel() 
        model.load_state_dict(checkpoint['model_state_dict'])
        learning_rates = checkpoint['learning_rates']
        optimizer = optim.Adam([
            {'params': model.cnn_lstm.parameters(), 'lr': learning_rates['cnn_lstm']},
            {'params': model.lstm.parameters(), 'lr': learning_rates['lstm']},
            {'params': model.mlp.parameters(), 'lr': learning_rates['mlp']}
        ])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.eval()
        pred_train = model(tX_train).detach().numpy()
        pred_test = model(tX_test).detach().numpy()
        print("pre-transfer, mse train:", mean_squared_error(y_train, pred_train ) )
        print("pre-transfer, mse test:", mean_squared_error(y_test, pred_test ) )

        fig1, axs = plt.subplots(1, 1, figsize = (10, 10))
        axs.scatter(y_test, pred_test, color = 'blue', alpha = 0.7)
        axs.scatter([], [], color = 'blue', label = "Pre-Transferred R$^2$: " + str(round(r2_score(y_test, pred_test),2)))
        start = min(min(y_test), min(pred_test))
        end = max(max(y_test), max(pred_test))
        line = np.arange(start, end, 0.01)
        axs.plot(line, line, color = 'black')
        #axs.plot(line, line + 0.17, color = 'black', linestyle = '--')
        #axs.plot(line, line - 0.17, color = 'black', linestyle = '--')
        axs.set_xlabel("True Values", fontsize = 24)
        axs.set_ylabel("Predictions", fontsize = 24)
        axs.legend(loc = 'lower right', fontsize = 20)
        axs.tick_params(axis='both', labelsize=18)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.tight_layout()
        fig1.savefig('Images/' + name + '_R2_Pre_Transfer.png')
        

        params = {
        'n_epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.05
        } 

        train_loss, test_loss, n_epochs = train_mlp(name, model, (tX_train, tY_train), (tX_test, tY_test), params)
        checkpoint = torch.load(f'Models/Transferred_{name}.pth', map_location=torch.device('cpu'), weights_only=False)
        model = analysis.HybridModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam([
            {'params': model.mlp.parameters(), 'lr': learning_rates['mlp']}
        ])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        n_epochs = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        test_loss = checkpoint['test_loss']

        fig2, axs = plt.subplots(1, 1, figsize = (10, 10))
        axs.plot(range(n_epochs),train_loss[:n_epochs], label="training")
        axs.plot(range(n_epochs),test_loss[:n_epochs], label="test")
        axs.set_xlabel("N_epoch", fontsize = 24)
        axs.set_ylabel("Loss", fontsize = 24)
        axs.legend(loc = 'upper right', fontsize = 20)
        axs.tick_params(axis='both', labelsize=18)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.tight_layout()
        fig2.savefig('Images/' + name + '_MSE_Transfer_Learning.png')

        model.eval()
        pred_train = model(tX_train).detach().numpy()
        pred_test = model(tX_test).detach().numpy()
        print("post-transfer, mse train:", mean_squared_error(y_train, pred_train ) )
        print("post-transfer, mse test:", mean_squared_error(y_test, pred_test ) )

        fig3, axs = plt.subplots(1, 1, figsize = (10, 10))
        axs.scatter(y_test, pred_test, color = 'blue', alpha = 0.7)
        axs.scatter([], [], color = 'blue', label = "Post-Transferred R$^2$: " + str(round(r2_score(y_test, pred_test),2)))
        start = min(min(y_test), min(pred_test))
        end = max(max(y_test), max(pred_test))
        line = np.arange(start, end, 0.01)
        axs.plot(line, line, color = 'black')
        #axs.plot(line, line + 0.17, color = 'black', linestyle = '--')
        #axs.plot(line, line - 0.17, color = 'black', linestyle = '--')
        axs.set_xlabel("True Values", fontsize = 24)
        axs.set_ylabel("Predictions", fontsize = 24)
        axs.legend(loc = 'lower right', fontsize = 20)
        axs.tick_params(axis='both', labelsize=18)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.tight_layout()
        fig3.savefig('Images/' + name + '_R2_Post_Transfer.png')
        



if __name__ == "__main__":
    main()