import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #type: ignore
import librosa #type:ignore
import os
import dataset
import models
import trainer
import torch

# %%
def get_device(device_idx):
    return torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')


def load_df(dfpath):
    df = pd.read_csv(dfpath)
    esc10_df = df[df['esc10']]
    train_df = esc10_df[(esc10_df['fold']==1) | (esc10_df['fold']==2) | (esc10_df['fold']==3) ]
    val_df = esc10_df[(esc10_df['fold']==4)]
    test_df = esc10_df[(esc10_df['fold']==5)]
    return train_df, val_df, test_df

# %%
# LOAD AUDIO
def load_audio(filepath, win_shift = 10):
    '''
    win_shift: window shift in ms
    '''
    audio, sr = librosa.load(filepath, sr = None)
    n_fft = sr * win_shift // 1000
    hop_length = sr * win_shift // 1000
    mel_spec = librosa.feature.melspectrogram(y = audio, sr = sr, n_fft = n_fft, hop_length = hop_length)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec

def get_optimizer(choice, lr, momentum, model):

    if choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    elif choice == 'sgd_with_momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum= momentum)
    elif choice == 'rms_prop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)
    else:
        #print('Invalid Choice of Optimizer! Will return Adam')
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    return optimizer

def plot_losses(train_loss, val_loss, save_path, plot_title):
    
    fig = plt.figure(figsize = (10,6))
    epochs = range(1, len(train_loss)+1)
    plt.plot(epochs, train_loss, color = 'blue', label = 'Train Loss')
    plt.plot(epochs, val_loss, color = 'green', label = 'Val Loss')
    plt.xlabel('Epochs')
    plt.xticks(epochs)
    plt.ylabel('Loss')
    plt.title(f'Loss vs Epoch plot for {plot_title}')
    plt.legend()
    save_path = os.path.join(save_path, 'loss_plot.png')
    plt.savefig(save_path)
    return


if __name__ == '__main__':

# %%    # DEFINE HYPERPARAMS
    WINDOW_SHIFT = 10
    BATCH_SIZE = 16
    FILTER_DEPTH = 16
    NUM_CLASSES = 10
    HIDDEN_DIMS = 128
    LR = 0.0005
    MOMENTUM = 0.9
    OPTIM_CHOICE = 'rms_prop'
    NORM_CHOICE = 'no_norm'
    NUM_EPOCHS = 20
    SEED = 746

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Get the device
    device = get_device(4)

# %%    # Load the DFs
    print('Loading the data...')
    df_path = '/data/home/samsadalam/MLSP_Assignment/Assignment3/ESC-50-master/meta/esc50.csv'
    train_df, val_df, test_df =  load_df(df_path)

    # Get the target indices, we need to convert the indices to 0-num_classes range
    train_targets = train_df['target']
    targets_to_index = {target: index for target, index in zip(train_targets.unique(), range(len(train_targets.unique())))}
    #print(targets_to_index)

    # Now prepare dataset and dataloader
    print('Create dataset and dataloaders...')
    audiopath = '/data/home/samsadalam/MLSP_Assignment/Assignment3/ESC-50-master/audio'

    train_dataset = dataset.AudioDataset(audiopath, train_df, targets_to_index, load_audio, win_shift= WINDOW_SHIFT)
    val_dataset = dataset.AudioDataset(audiopath, val_df, targets_to_index, load_audio, win_shift= WINDOW_SHIFT)
    test_dataset = dataset.AudioDataset(audiopath, test_df, targets_to_index, load_audio, win_shift= WINDOW_SHIFT)

    train_loader = dataset.create_dataloader(train_dataset, batch_size= BATCH_SIZE, shuffle = True, collate_fn= None)
    val_loader = dataset.create_dataloader(val_dataset, batch_size= BATCH_SIZE, shuffle = False, collate_fn= None)
    test_loader = dataset.create_dataloader(test_dataset, batch_size= BATCH_SIZE, shuffle = False, collate_fn= None)


# %% BASIC MODEL TRAINING FOR A AND B PART
    model = models.AudioCNN(FILTER_DEPTH, HIDDEN_DIMS, NUM_CLASSES, NORM_CHOICE).to(device)
    optimizer = get_optimizer(OPTIM_CHOICE, LR, MOMENTUM, model)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma = 0.2)
    scheduler = None
    criterion = torch.nn.CrossEntropyLoss()
    accuracy = trainer.Accuracy.accuracy_ce

    os.makedirs('/data/home/samsadalam/MLSP_Assignment/Assignment3/ESC-50-master/q3_cnn_b' + NORM_CHOICE+ '_' +OPTIM_CHOICE, exist_ok= True)
    save_dir = '/data/home/samsadalam/MLSP_Assignment/Assignment3/ESC-50-master/q3_cnn_b' + NORM_CHOICE+ '_'+ OPTIM_CHOICE

    # Begin training
    print('Begin Training...')
    model_trainer = trainer.Trainer(model = model, 
                                    criterion = criterion, 
                                    optimizer = optimizer, 
                                    accuracy = accuracy, 
                                    save_dir= save_dir, 
                                    scheduler = scheduler, 
                                    use_tqdm = True)
    
    train_val_metric = model_trainer.train(train_loader, val_loader, num_epochs = NUM_EPOCHS)

    # Compute test accuracy
    best_model = train_val_metric['model_path']
    model_state_dict = torch.load(best_model, map_location= device, weights_only= True)
    model.load_state_dict(model_state_dict)

    test_loss, test_acc = model_trainer.eval_step(test_loader)
    print('\nTraining finished...\n')

    # Plot train and val losses
    train_loss = train_val_metric['loss']['train']
    val_loss = train_val_metric['loss']['val']
    train_acc = train_val_metric['acc']['train']
    val_acc = train_val_metric['acc']['val']

    plot_losses(train_loss, val_loss, save_dir, f'Adam and {NORM_CHOICE} ')

    with open(os.path.join(save_dir, 'loss_acc.txt'), 'w') as f:
        for i in range(len(train_loss)):
            f.write(f'Epoch: {i+1} | Train Loss: {train_loss[i]:.4f} | Val Loss: {val_loss[i]:.4f} \n')
        f.write(f'Test Acc: {test_acc}')

    print('Test Accuracy: ', test_acc)


# %% Train 3 Models and use them for ensembling

    model_sgd_wo_norm = models.AudioCNN(FILTER_DEPTH, HIDDEN_DIMS, NUM_CLASSES, None).to(device)
    optimizer_sgd_wo_norm = get_optimizer('sgd', 0.01, MOMENTUM, model_sgd_wo_norm)
    os.makedirs('/data/home/samsadalam/MLSP_Assignment/Assignment3/ESC-50-master/q3_cnn_' + 'c_sgd_wo_norm', exist_ok= True)
    save_dir_sgd_wo_norm = '/data/home/samsadalam/MLSP_Assignment/Assignment3/ESC-50-master/q3_cnn_' + 'c_sgd_wo_norm'

    # Begin training
    print('Begin Training SGD without Norm...')
    model_trainer = trainer.Trainer(model = model_sgd_wo_norm, 
                                    criterion = criterion, 
                                    optimizer = optimizer_sgd_wo_norm, 
                                    accuracy = accuracy, 
                                    save_dir= save_dir_sgd_wo_norm, 
                                    scheduler = scheduler, 
                                    use_tqdm = True)
    
    train_val_metric = model_trainer.train(train_loader, val_loader, num_epochs = NUM_EPOCHS)

    # Compute test accuracy
    best_model_sgd_wo_norm = train_val_metric['model_path']
    

# %% CNN with RMS Prop and Layernorm

    model_rms_layern = models.AudioCNN(FILTER_DEPTH, HIDDEN_DIMS, NUM_CLASSES, norm_type = 'layernorm').to(device)
    optimizer_rms_layern = get_optimizer('rms_prop', 0.0005, MOMENTUM, model_rms_layern)
    os.makedirs('/data/home/samsadalam/MLSP_Assignment/Assignment3/ESC-50-master/q3_cnn_' + 'c_rms_layernorm', exist_ok= True)
    save_dir_rms_layern = '/data/home/samsadalam/MLSP_Assignment/Assignment3/ESC-50-master/q3_cnn_' + 'c_rms_layernorm'

    # Begin training
    print('Begin Training RMS Prop with Layernorm...')
    model_trainer = trainer.Trainer(model = model_rms_layern, 
                                    criterion = criterion, 
                                    optimizer = optimizer_rms_layern, 
                                    accuracy = accuracy, 
                                    save_dir= save_dir_rms_layern, 
                                    scheduler = scheduler, 
                                    use_tqdm = True)
    
    train_val_metric = model_trainer.train(train_loader, val_loader, num_epochs = NUM_EPOCHS)

    # Compute test accuracy
    best_model_rms_layern = train_val_metric['model_path']


# %%
# Adam with Batchnorm
# CNN with RMS Prop and Layernorm
    model_adam_bn = models.AudioCNN(FILTER_DEPTH, HIDDEN_DIMS, NUM_CLASSES, norm_type = 'batchnorm').to(device)
    optimizer_adam_bn = get_optimizer('adam', 0.0005, MOMENTUM, model_adam_bn)
    os.makedirs('/data/home/samsadalam/MLSP_Assignment/Assignment3/ESC-50-master/q3_cnn_' + 'c_adam_bn', exist_ok= True)
    save_dir_adam_bn = '/data/home/samsadalam/MLSP_Assignment/Assignment3/ESC-50-master/q3_cnn_' + 'c_adam_bn'

    # Begin training
    print('Begin Training RMS Prop with Layernorm...')
    model_trainer = trainer.Trainer(model = model_adam_bn, 
                                    criterion = criterion, 
                                    optimizer = optimizer_adam_bn, 
                                    accuracy = accuracy, 
                                    save_dir= save_dir_adam_bn, 
                                    scheduler = scheduler, 
                                    use_tqdm = True)
    
    train_val_metric = model_trainer.train(train_loader, val_loader, num_epochs = NUM_EPOCHS)

    # Compute test accuracy
    best_model_adam_bn = train_val_metric['model_path']
    
# %%
# Once the models are trained take their best state dict and load them
    model_sgd_wo_norm.load_state_dict(torch.load(best_model_sgd_wo_norm, weights_only= True, map_location= device))
    model_rms_layern.load_state_dict(torch.load(best_model_rms_layern, weights_only= True, map_location= device))
    model_adam_bn.load_state_dict(torch.load(best_model_adam_bn, weights_only= True, map_location= device))
    os.makedirs('/data/home/samsadalam/MLSP_Assignment/Assignment3/ESC-50-master/q3_cnn_ensemble_non_train', exist_ok= True)
    save_dir = '/data/home/samsadalam/MLSP_Assignment/Assignment3/ESC-50-master/q3_cnn_ensemble_non_train'
    # Create ensemble model
    ensemble_model = models.EnsembleModel(model_sgd_wo_norm, model_rms_layern, model_adam_bn).to(device)
    ensemble_trainer = trainer.Trainer(model = ensemble_model,
                                    criterion= criterion,
                                    optimizer= torch.optim.SGD(ensemble_model.parameters(), lr = 0.01, momentum= MOMENTUM),
                                    accuracy= accuracy,
                                    save_dir = save_dir,
                                    scheduler= None,
                                    use_tqdm = False
                                    )

    # Since we don't need to train, simply evaluate the ensemble model on the test set to get average accuracy
    test_loss, test_acc_no_train = ensemble_trainer.eval_step(test_loader)
    # with open(os.path.join(save_dir, 'test_acc.txt'), 'w') as f:
    #         f.write(f'Test Acc: {test_acc}')

    # print('Test Accuracy of Ensemble Averaging: ', test_acc)


    # Now train the ensemble model for values of alpha but use validation set for training
    train_val_metric = ensemble_trainer.train(val_loader, test_loader, num_epochs= 20)
    ensemble_best = train_val_metric['model_path']
    ensemble_model.load_state_dict(torch.load(ensemble_best, weights_only= True, map_location= device))

    test_loss, test_acc_train = ensemble_trainer.eval_step(test_loader)
    with open(os.path.join(save_dir, 'test_acc.txt'), 'w') as f:
            f.write(f'Test Acc without ensemble train: {test_acc_no_train} \nTest Acc with ensemble train: {test_acc_train}')

    print(f'Test Acc without ensemble train: {test_acc_no_train} \nTest Acc with ensemble train: {test_acc_train}')

