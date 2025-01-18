import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import analysis 

def compute_uncertainty(model, data_loader, n_samples=10, acquisition='bald'):
    model.eval()
    all_uncertainties = []

    with torch.no_grad():
        for X, _ in data_loader:
            predictions = torch.stack([F.softmax(model(X), dim=1) for _ in range(n_samples)], dim=0)
            mean_predictions = predictions.mean(dim=0)
            
            if acquisition == 'bald':
                entropy = -torch.sum(mean_predictions * torch.log(mean_predictions + 1e-10), dim=1)
                expected_entropy = -torch.mean(torch.sum(predictions * torch.log(predictions + 1e-10), dim=2), dim=0)
                uncertainties = entropy - expected_entropy
            elif acquisition == 'entropy':
                uncertainties = -torch.sum(mean_predictions * torch.log(mean_predictions + 1e-10), dim=1)
            
            all_uncertainties.append(uncertainties)

    return torch.cat(all_uncertainties)


def train_mlp_with_active_learning(model, train_data, test_data, pool_data, params, acquisition_fn, n_iterations, name):
    labeled_X, labeled_y = train_data
    pool_X, pool_y = pool_data
    test_X, test_y = test_data
    
    all_train_loss = []
    all_test_loss = []

    outer_patience = params.get('outer_patience', 3)
    outer_min_delta = params.get('outer_min_delta', 1e-4)
    outer_patience_counter = 0
    best_outer_test_loss = float('inf')
    
    for iteration in range(n_iterations):
        print(f"Active Learning Iteration {iteration + 1}/{n_iterations}")
        
        patience = params.get('patience', 5)
        min_delta = params.get('min_delta', 1e-4)
        patience_counter = 0
        best_test_loss = float('inf')
        

        train_loader = DataLoader(analysis.MyDataSet(labeled_X, labeled_y), batch_size=params['batch_size'], shuffle=True)
        optimizer = optim.Adam([{'params': model.mlp.parameters(), 'lr': params['learning_rate']}])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=0.01)
        criterion = nn.MSELoss()
        
        train_loss = []
        for epoch in range(params['n_epochs']):
            model.train()
            epoch_loss = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            train_loss.append(epoch_loss / len(train_loader))
        
            model.eval()
            with torch.no_grad():
                test_preds = model(test_X)
                test_loss = criterion(test_preds, test_y).item()
            
            scheduler.step(test_loss)
            print(f"Iteration {iteration + 1}, Epoch {epoch + 1}, Train Loss: {train_loss[-1]:.4f}, Test Loss: {test_loss:.4f}")
            
            if epoch > 2 and abs(test_loss - best_test_loss) < min_delta:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered in iteration {iteration + 1}, epoch {epoch + 1}")
                    break
            else:
                patience_counter = 0
                best_test_loss = test_loss
        all_train_loss.append(train_loss[-1])
        all_test_loss.append(test_loss)
        
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': all_train_loss,
            'test_loss': all_test_loss,
            'learning_rates': {
                'mlp': params['learning_rate']
            }
        }, f'Models/AL_1000/AL_Transferred_{name}.pth')
        
        if iteration > 0 and abs(test_loss - best_outer_test_loss) < outer_min_delta:
            outer_patience_counter += 1
            if outer_patience_counter >= outer_patience:
                print(f"Outer early stopping triggered at iteration {iteration + 1}. No significant improvement in test loss.")
                break
        else:
            outer_patience_counter = 0
            best_outer_test_loss = test_loss
        
        pool_loader = DataLoader(analysis.MyDataSet(pool_X, pool_y), batch_size=params['batch_size'], shuffle=False)
        uncertainties = compute_uncertainty(model, pool_loader, n_samples=10, acquisition=acquisition_fn)

        k = min(params['selection_size'], pool_X.size(0))
        selected_indices = torch.topk(uncertainties, k, largest=True).indices

        new_X = pool_X[selected_indices]
        new_y = pool_y[selected_indices]
        labeled_X = torch.cat([labeled_X, new_X], dim=0)
        labeled_y = torch.cat([labeled_y, new_y], dim=0)

        mask = torch.ones(pool_X.size(0), dtype=torch.bool)
        mask[selected_indices] = False
        pool_X = pool_X[mask]
        pool_y = pool_y[mask]

        if pool_X.size(0) == 0:
            print("Data pool exhausted.")
            break

    return all_train_loss, all_test_loss, len(all_train_loss)


def main():
    for name in ['bonde', 'kuo_arti', 'kuo_dmsc', 'kosuri']:
        print(name)

        X_train, X_test, y_train, y_test = analysis.Create_Model_Input(dataset = name, simple_onehot = True)

        tX_train = torch.from_numpy(X_train).to(torch.float32)
        tX_test = torch.from_numpy(X_test).to(torch.float32)
        tY_train = torch.from_numpy(y_train).to(torch.float32).reshape(-1, 1)
        tY_test = torch.from_numpy(y_test).to(torch.float32).reshape(-1, 1)
        
        initial_size = min(1000, tX_train.size(0)) 
        labeled_indices = torch.randperm(tX_train.size(0))[:initial_size]
        pool_indices = torch.ones(tX_train.size(0), dtype=torch.bool)
        pool_indices[labeled_indices] = False

        labeled_X = tX_train[labeled_indices]
        labeled_y = tY_train[labeled_indices]
        pool_X = tX_train[pool_indices]
        pool_y = tY_train[pool_indices]

        train_data = (labeled_X, labeled_y)
        test_data = (tX_test, tY_test)
        pool_data = (pool_X, pool_y)
        
        # Load the checkpoint
        checkpoint = torch.load('Models/Backbone_fepB.pth', map_location=torch.device('cpu'), weights_only=False)
        model = analysis.HybridModel()  # Adjust this line as needed to instantiate your model
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
        axs.plot(line, line + 0.17, color = 'black', linestyle = '--')
        axs.plot(line, line - 0.17, color = 'black', linestyle = '--')
        axs.set_xlabel("True Values", fontsize = 24)
        axs.set_ylabel("Predictions", fontsize = 24)
        axs.legend(loc = 'lower right', fontsize = 20)
        axs.tick_params(axis='both', labelsize=18)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.tight_layout()
        fig1.savefig('Images/AL_' + name + '_R2_Pre_Transfer.png')
        

        params = {
            'n_epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.01,
            'selection_size': 50,  # Number of samples to select per iteration
            'patience': 5,
            'min_delta': 1e-4,
            'outer_patience': 3,
            'outer_min_delta': 1e-3
        }

        train_loss, test_loss, n_iteration = train_mlp_with_active_learning(
            model=model,
            train_data=train_data,
            test_data=test_data,
            pool_data=pool_data,
            params=params,
            acquisition_fn='bald',  # Replace with your acquisition function
            n_iterations=50,
            name=name
        )
        
        
        # Load the checkpoint
        checkpoint = torch.load(f'Models/AL_1000/AL_Transferred_{name}.pth', map_location=torch.device('cpu'), weights_only=False)
        model = analysis.HybridModel()
        model.load_state_dict(checkpoint['model_state_dict'])

        iteration = checkpoint['iteration']
        train_loss = np.array(checkpoint['train_loss'])
        test_loss = np.array(checkpoint['test_loss'])

        fig2, axs = plt.subplots(1, 1, figsize = (10, 10))
        axs.plot(range(len(train_loss)),train_loss, label="training")
        axs.plot(range(len(test_loss)),test_loss, label="test")
        axs.set_xlabel("N_Iteration", fontsize = 24)
        axs.set_ylabel("Final Loss in Each Iteration", fontsize = 24)
        axs.legend(loc = 'upper right', fontsize = 20)
        axs.tick_params(axis='both', labelsize=18)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.tight_layout()
        fig2.savefig('Images/AL_1000/AL_' + name + '_MSE_Transfer_Learning.png')
        
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
        fig3.savefig('Images/AL_1000/AL_' + name + '_R2_Post_Transfer.png')
        



if __name__ == "__main__":
    main()