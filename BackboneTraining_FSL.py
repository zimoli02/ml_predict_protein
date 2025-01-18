import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import analysis


def split_task_data(X, y, split_ratio=0.8):
    split_index = int(len(X) * split_ratio)
    support_X, support_y = X[:split_index], y[:split_index]
    query_X, query_y = X[split_index:], y[split_index:]
    return support_X, support_y, query_X, query_y

def meta_train_maml(model, tasks, meta_params, min_delta=1e-4):
    alpha = meta_params['inner_lr']
    beta = meta_params['outer_lr']
    n_inner_steps = meta_params['n_inner_steps']
    meta_epochs = meta_params['meta_epochs']
    
    meta_optimizer = optim.Adam(model.parameters(), lr=beta)
    criterion = nn.MSELoss()
    meta_loss_history = []
    
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(meta_epochs):
        meta_loss = 0.0

        for task_data in tasks:
            query_X, query_y = task_data

            adapted_model = deepcopy(model)
            inner_optimizer = optim.SGD(adapted_model.mlp.parameters(), lr=alpha)

            for _ in range(n_inner_steps):
                inner_optimizer.zero_grad()
                query_pred = adapted_model(query_X)
                inner_loss = criterion(query_pred, query_y)
                inner_loss.backward()
                inner_optimizer.step()
            
            adapted_model.eval()
            final_pred = adapted_model(query_X)  
            task_meta_loss = criterion(final_pred, query_y)  
            meta_loss += task_meta_loss  

        meta_optimizer.zero_grad()
        meta_loss = meta_loss / len(tasks)  
        meta_loss.backward()
        meta_optimizer.step()

        current_loss = meta_loss.item()
        meta_loss_history.append(current_loss)
        
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{meta_epochs}, Meta Loss: {current_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'meta_optimizer_state_dict': meta_optimizer.state_dict(),
            'meta_loss_history': meta_loss_history,
            'meta_params': meta_params,
            'best_model_state': best_model_state,
            'best_loss': best_loss
        }, 'Models/MAML_MetaTrained_Backbone.pth')

    return model, meta_loss_history


def main():
    # Load datasets
    kuo_fepb_data = analysis.Create_Model_Input(dataset='kuo_fepb', simple_onehot=True)
    kuo_arti_data = analysis.Create_Model_Input(dataset='kuo_arti', simple_onehot=True)
    kuo_dmsc_data = analysis.Create_Model_Input(dataset='kuo_dmsc', simple_onehot=True)

    def prepare_tensors(dataset):
        X, _, y, _ = dataset
        tX = torch.from_numpy(X).to(torch.float32)
        tY = torch.from_numpy(y).to(torch.float32).reshape(-1, 1)
        return tX, tY

    tX_fepb, tY_fepb = prepare_tensors(kuo_fepb_data)
    tX_arti, tY_arti = prepare_tensors(kuo_arti_data)
    tX_dmsc, tY_dmsc = prepare_tensors(kuo_dmsc_data)
    
    # Use 1000 samples each for new tasks
    num_samples = 1000
    selected_indices_fepb = torch.randperm(len(tX_fepb))[:num_samples]
    selected_indices_arti = torch.randperm(len(tX_arti))[:num_samples]
    selected_indices_dmsc = torch.randperm(len(tX_dmsc))[:num_samples]
    support_X_fepb, support_y_fepb = tX_fepb[selected_indices_fepb], tY_fepb[selected_indices_fepb]
    support_X_arti, support_y_arti = tX_arti[selected_indices_arti], tY_arti[selected_indices_arti]
    support_X_dmsc, support_y_dmsc = tX_dmsc[selected_indices_dmsc], tY_dmsc[selected_indices_dmsc]

    # Prepare tasks
    tasks = [
        (support_X_fepb, support_y_fepb),
        (support_X_arti, support_y_arti),
        (support_X_dmsc, support_y_dmsc)
    ]

    # Initialize model and meta-training parameters
    model = analysis.HybridModel()
    meta_params = {
        'inner_lr': 5e-2,
        'outer_lr': 3e-2,
        'n_inner_steps': 5,
        'meta_epochs': 50,
    }

    # Train the model
    meta_trained_model, meta_loss_history = meta_train_maml(model, tasks, meta_params)

    n_epochs = len(meta_loss_history)
    fig1, axs = plt.subplots(1, 1, figsize = (10, 10))
    axs.plot(range(n_epochs),meta_loss_history)
    axs.set_xlabel("N_epoch", fontsize = 24)
    axs.set_ylabel("Meta-Loss", fontsize = 24)
    axs.tick_params(axis='both', labelsize=18)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    plt.tight_layout()
    fig1.savefig('Images/FSL_Backbone_fepB_MSE.png')
    
    # Predict on test data
    tX_test = tX_fepb
    tY_test = tY_fepb
    meta_trained_model.eval()
    with torch.no_grad():
        pred_test = meta_trained_model(tX_test).cpu().numpy() 
        y_test = tY_test.cpu().numpy()
    print("FSL_Backbone_fepB, mse test:", mean_squared_error(y_test, pred_test))

    fig2, axs = plt.subplots(1, 1, figsize = (10, 10))
    axs.scatter(y_test, pred_test, color = 'blue', alpha = 0.7)
    axs.scatter([], [], color = 'blue', label = "R2: " + str(round(r2_score(y_test, pred_test),2)))
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
    fig2.savefig('Images/FSL_Backbone_fepB_R2.png')


if __name__ == "__main__":
    main()