import prep_data #type:ignore
import dataset #type:ignore
import trainer #type:ignore
import models #type:ignore
import os

import numpy as np
import torch
import matplotlib.pyplot as plt


def get_modal_dict(model_name):
    model_dict = dict(
        lstm_without_attention = dict(
            vocab_size = len(vocab),
            embd_dims = 300,
            hidden_size = 256,
            word2vec_embd = vocab_w2v,
            num_layers = 2,
            dropout = 0.3
        ),
        lstm_with_attention = dict(
            vocab_size = len(vocab),
            embd_dims = 300,
            hidden_size = 256,
            word2vec_embd = vocab_w2v,
            num_layers = 2,
            dropout = 0.3
        ),
        transformer_encoder = dict(
            word2vec_embd = vocab_w2v, 
            embed_dim = 300, 
            hidden_dim = 256, 
            mlp_size = 512, 
            max_seq_len = 512,
            num_heads = 4, 
            num_layers = 1, 
            attn_dropout = 0.3, 
            mlp_dropout = 0.3

        )
    )
    return model_dict[model_name]

def plot_losses(train_loss, val_loss, save_path, model_name):
    
    fig = plt.figure(figsize = (6,6))
    epochs = range(1, len(train_loss)+1)
    plt.plot(epochs, train_loss, color = 'blue', label = 'Train Loss')
    plt.plot(epochs, val_loss, color = 'green', label = 'Val Loss')
    plt.xlabel('Epochs')
    plt.xticks(epochs)
    plt.ylabel('Loss')
    plt.title(f'Loss vs Epoch plot for model: {model_name.replace("_", " ")}')
    plt.legend()
    save_path = os.path.join(save_path, 'loss_plot.png')
    plt.savefig(save_path)
    return


if __name__ == '__main__':

    print('All good')
    SEED = 746
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device_idx = 6
    device = torch.device(f'cuda:{device_idx}') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    print('Using device: ', device)

    # prepare the data
    datapath = '/data/home/samsadalam/MLSP_Assignment/Assignment3/IMDB Dataset.csv'
    train_size = 40000
    test_size = 5000
    print_info = True

    [(xtrain, ytrain), (xval, yval), (xtest, ytest)] = prep_data.preprocess(datapath, train_size, test_size, print_info, SEED)

    # Get word2vec embeddings
    print('\nLoading Word2Vec...')
    word_vectors = prep_data.load_word2vec()


    #Now create vocab and vocab embeddings from word2vec
    print('\nCreating Vocab...')
    if os.path.exists('/data/home/samsadalam/MLSP_Assignment/Assignment3/vocab.pt'):
        vocab = torch.load('/data/home/samsadalam/MLSP_Assignment/Assignment3/vocab.pt', weights_only= False)
    else:
        vocab = dataset.create_vocabulary(xtrain, cutoff_freq = 2, word_vectors = word_vectors)
        torch.save(vocab, '/data/home/samsadalam/MLSP_Assignment/Assignment3/vocab.pt' )

    print('\nCreating Word2Vec for Vocab...')
    if os.path.exists('/data/home/samsadalam/MLSP_Assignment/Assignment3/vocab_w2v.pt'):
        vocab_w2v = torch.load('/data/home/samsadalam/MLSP_Assignment/Assignment3/vocab_w2v.pt', weights_only= False)
    else:
        vocab_w2v = dataset.vocab_word2vec(vocab, word_vectors)
        torch.save(vocab_w2v, '/data/home/samsadalam/MLSP_Assignment/Assignment3/vocab_w2v.pt')

    #prepare dataset and dataloader

    train_dataset = dataset.TokenizerDataset(xtrain, ytrain, vocab)
    val_dataset = dataset.TokenizerDataset(xval, yval, vocab)
    test_dataset = dataset.TokenizerDataset(xtest, ytest, vocab)

    train_loader = dataset.create_dataloader(train_dataset, batch_size = 32, shuffle = True, collate_fn= train_dataset.collate_fn)
    val_loader = dataset.create_dataloader(val_dataset, batch_size= 32, shuffle = False, collate_fn= val_dataset.collate_fn)
    test_loader = dataset.create_dataloader(test_dataset, batch_size= 32, shuffle = False, collate_fn= test_dataset.collate_fn)

    model_name = 'transformer_encoder'
    model_dict = get_modal_dict(model_name)

    if model_name == 'lstm_without_attention':
        model = models.LSTMwithoutAttn(**model_dict).to(device)
    elif model_name == 'lstm_with_attention':
        model = models.LSTMwithAttention(**model_dict).to(device)
    elif model_name == 'transformer_encoder':
        model = models.TransformerModel(**model_dict).to(device)
        #print(model)
        max_seq_len = model_dict['max_seq_len']
        train_dataset.max_seq_len = max_seq_len
        val_dataset.max_seq_len = max_seq_len
        test_dataset.max_seq_len = max_seq_len
    else:
        print('Model name error! Model not found.')
        exit()

    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.00005, weight_decay= 0.00001)
    criterion = torch.nn.BCEWithLogitsLoss()
    accuracy = trainer.Accuracy.accuracy_bce

    print('\nStarting the training...\n')
    os.makedirs('/data/home/samsadalam/MLSP_Assignment/Assignment3/'+ model_name, exist_ok = True)
    save_dir = '/data/home/samsadalam/MLSP_Assignment/Assignment3/'+ model_name
    model_trainer = trainer.Trainer(model, criterion, optimizer, accuracy, use_tqdm = False, save_dir = save_dir)
    train_val_metric = model_trainer.train(train_loader, val_loader, num_epochs = 25) 

    # Compute test accuracy
    best_model = train_val_metric['model_path']
    model_state_dict = torch.load(best_model, map_location= device, weights_only= True)
    model.load_state_dict(model_state_dict)

    test_loss, test_acc = model_trainer.eval_step(test_loader)
    print('\nTraining finished...\n')

    # Plot train and val losses
    train_loss = train_val_metric['loss']['train']
    val_loss = train_val_metric['loss']['val']

    plot_losses(train_loss, val_loss, save_dir, model_name.replace("_"," "))

    with open(os.path.join(save_dir, 'test_acc.txt'), 'w') as f:
        f.write(f'Test Acc: {test_acc}')

    



    
        