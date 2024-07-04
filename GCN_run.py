import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn import metrics
import torch
import torch.nn.functional as F
from gcn_model import GCN
from utils import load_data
from utils import accuracy
from matplotlib import pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def train(epoch, optimizer, features, adj, labels, idx_train):
    labels.to(device)

    GCN_model.train()
    optimizer.zero_grad()
    output = GCN_model(features, adj)
    
    # check sample number for distinguish SMOTE or normal input
    if(features.size(0) == 511):
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        
        # output is the one-hot label
        ot = output[idx_train].detach().cpu().numpy()
        # change one-hot label to digit label
        ot = np.argmax(ot, axis=1)
        # original label
        lb = labels[idx_train].detach().cpu().numpy()
    
    else: 
        loss_train = F.cross_entropy(output, labels)
        acc_train = accuracy(output, labels)
        
        # output is the one-hot label
        ot = output.detach().cpu().numpy()
        # change one-hot label to digit label
        ot = np.argmax(ot, axis=1)
        # original label
        lb = labels.detach().cpu().numpy()
 
    # calculate the f1 score
    f = f1_score(ot, lb, average='weighted')
    
    # calculate the precision
    precision_train = metrics.precision_score(ot, lb, average='weighted')
    
    # calculate the recall
    recall_train = metrics.recall_score(ot, lb, average='weighted') 

    # calculate the Matthews Correlation Coefficient (MCC)
    mcc_train = metrics.matthews_corrcoef(ot, lb)
    
    # calculate Mean Squared Error (MSE)
    mse_train = metrics.mean_squared_error(ot, lb)
    
    # calculate confusion matrix
    cm_model = metrics.confusion_matrix(lb, ot)
    
    loss_train.backward()
    optimizer.step()

    # plot confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(cm_model, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix of Best Epoch Model')
    plt.savefig('static/images/best_confusion_matrix_smote_edited_normal{:.1f}.png'.format(epoch+1))
    plt.close()
    
    # if (epoch+1) % 10 == 0:
    print('Epoch: %.2f | acc train: %.4f | f1 train: %.4f | precision train: %.4f | recall train: %.4f | mcc train: %.4f | loss train: %.4f | mse train: %.4f' 
              %(epoch+1, acc_train.item(), f, precision_train.item(), recall_train.item(), mcc_train.item(), loss_train.item(), mse_train.item()))    
    return acc_train.item(), f, precision_train.item(), recall_train.item(), mcc_train.item(), loss_train.item(), mse_train.item()

def test(features, adj, labels, idx_test):
    '''
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param labels: sample labels
    :param idx_test: the index of tested samples
    '''
    GCN_model.eval()
    output = GCN_model(features, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])

    #calculate the accuracy
    acc_test = accuracy(output[idx_test], labels[idx_test])
    
    # output is the one-hot label (predicted)
    ot = output[idx_test].detach().cpu().numpy()
    # change one-hot label to digit label 
    ot = np.argmax(ot, axis=1)
    # original label
    lb = labels[idx_test].detach().cpu().numpy()

    #calculate the f1 score
    f = f1_score(ot, lb, average='weighted')

    # calculate the precision
    precision_test = metrics.precision_score(ot, lb, average='weighted')
    
    # calculate the recall
    recall_test = metrics.recall_score(ot, lb, average='weighted') 

    # calculate the Matthews Correlation Coefficient (MCC)
    mcc_test = metrics.matthews_corrcoef(ot, lb)
    
    # calculate Mean Squared Error (MSE)
    mse_test = metrics.mean_squared_error(ot, lb)
    
    print('Epoch: %.2f | acc test: %.4f | f1 test: %.4f | precision test: %.4f | recall test: %.4f | mcc test: %.4f | loss test: %.4f | mse test: %.4f' 
              %(epoch+1, acc_test.item(), f, precision_test.item(), recall_test.item(), mcc_test, loss_test.item(), mse_test.item()))    

    # return accuracy, f1 score, precision, recall, mcc, loss and mse
    return acc_test.item(), f, precision_test.item(), recall_test.item(), mcc_test, loss_test.item(), mse_test.item()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--featuredata', '-fd', type=str, required=True, help='The vector feature file.')
    parser.add_argument('--adjdata', '-ad', type=str, required=True, help='The adjacency matrix file.')
    parser.add_argument('--labeldata', '-ld', type=str, required=True, help='The sample label file.')
    parser.add_argument('--testsample', '-ts', type=str, help='Test sample names file.')
    parser.add_argument('--mode', '-m', type=int, choices=[0,1], default=0,
                        help='mode 0: 10-fold cross validation; mode 1: train and test a model.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='cpu',
                        help='Training on cpu or gpu, default: cpu.')
    parser.add_argument('--epochs', '-e', type=int, default=150, help='Training epochs, default: 150.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('--weight_decay', '-w', type=float, default=0.01,
                        help='Weight decay (L2 loss on parameters), methods to avoid overfitting, default: 0.01')
    parser.add_argument('--hidden', '-hd',type=int, default=64, help='Hidden layer dimension, default: 64.')
    parser.add_argument('--dropout', '-dp', type=float, default=0.5, help='Dropout rate, methods to avoid overfitting, default: 0.5.')
    parser.add_argument('--threshold', '-t', type=float, default=0.005, help='Threshold to filter edges, default: 0.005')
    parser.add_argument('--nclass', '-nc', type=int, default=4, help='Number of classes, default: 4')
    parser.add_argument('--patience', '-p', type=int, default=20, help='Patience')
    args = parser.parse_args()

    # Check whether GPUs are available 
    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set random seed
    setup_seed(args.seed)

    # load input files
    adj, data, label = load_data(args.adjdata, args.featuredata, args.labeldata, args.threshold)

    # change dataframe to Tensor
    adj = torch.tensor(adj, dtype=torch.float, device=device)
    features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float, device=device)
    labels = torch.tensor(label.iloc[:, 1].values, dtype=torch.long, device=device)

    print('Begin training model...')

    # 10-fold cross validation
    if args.mode == 0:
        skf = StratifiedKFold(n_splits=10, shuffle=True)

        acc_res, f1_res, precision_res, recall_res, loss_res, mcc_res, mse_res = [], [], [], [], [], [], []  #record accuracy and f1 score

        # split train and test data
        for idx_train, idx_test in skf.split(data.iloc[:, 1:], label.iloc[:, 1]):
            # conduct SMOTE test for Selected features data
            sm = SMOTE(k_neighbors=1,random_state=args.seed)
            features_smote, labels_smote = sm.fit_resample(features[idx_train],labels[idx_train])
            print ('Shape of oversampled data: {}'.format(features_smote.shape))
            print ('Shape of Y: {}'.format(labels_smote.shape))
            fea_smote=torch.from_numpy(features_smote)
            lab_smote=torch.from_numpy(labels_smote)
            
            counter = Counter(labels_smote)
            label, values = zip(*counter.items()) 
            
            # Create a bar plot
            plt.bar(label, values)

            # Add labels and title
            plt.xlabel('Class')
            plt.ylabel('Frequency')
            plt.title('Distribution of Classes')

            # Display the plot
            plt.savefig('static/images/SMOTE_selected_features_1300.png')
            plt.close()
            
            # conduct SMOTE test for SNF data integration result
            sm = SMOTE(k_neighbors=1,random_state=args.seed)
            adjacent_smote, labels_smote = sm.fit_resample(adj[idx_train],labels[idx_train])
            print ('Shape of oversampled data: {}'.format(adjacent_smote.shape))
            print ('Shape of Y: {}'.format(labels_smote.shape))
            adj_smote=torch.from_numpy(adjacent_smote)
            lab_smote=torch.from_numpy(labels_smote)
            
            counter = Counter(labels_smote)
            label, values = zip(*counter.items()) 
            
            # Create a bar plot
            plt.bar(label, values)

            # Add labels and title
            plt.xlabel('Class')
            plt.ylabel('Frequency')
            plt.title('Distribution of Classes')

            # Display the plot
            plt.savefig('static/images/SMOTE_adjacent_matrix.png')
            plt.close()
            
            # initialize a model
            GCN_model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, dropout=args.dropout)
            GCN_model.to(device)

            # define the optimizer
            optimizer = torch.optim.Adam(GCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)

            idx_train, idx_test= torch.tensor(idx_train, dtype=torch.long, device=device), torch.tensor(idx_test, dtype=torch.long, device=device)
            for epoch in range(args.epochs):
                train(epoch, optimizer, fea_smote, adj_smote.t(), lab_smote, idx_train)
                # train(epoch, optimizer, features, adj, labels, idx_train)
                
            # calculate the accuracy, f1 score, precision, recall, mcc, loss and mse
            ac, f1, precision, recall, mcc, loss, mse= test(features, adj, labels, idx_test)
            acc_res.append(ac)
            f1_res.append(f1)
            precision_res.append(precision)
            recall_res.append(recall)
            mcc_res.append(mcc)
            loss_res.append(loss)
            mse_res.append(mse)
        
        print('10-fold  Acc(%.4f, %.4f)  F1(%.4f, %.4f)  Precision(%.4f, %.4f)  Recall(%.4f, %.4f)  MCC(%.4f, %.4f)  Loss(%.4f, %.4f)  MSE(%.4f, %.4f)'  % (np.mean(acc_res), np.std(acc_res), np.mean(f1_res), np.std(f1_res), np.mean(precision_res), np.std(precision_res), np.mean(recall_res), np.std(recall_res), np.mean(mcc_res), np.std(mcc_res), np.mean(loss_res), np.std(loss_res), np.mean(mse_res), np.std(mse_res)))

    # train and test model
    elif args.mode == 1:
        # load test samples
        test_sample_df = pd.read_csv(args.testsample, header=0, index_col=None)
        test_sample = test_sample_df.iloc[:, 0].tolist()
        all_sample = data['Sample'].tolist()
        train_sample = list(set(all_sample)-set(test_sample))

        # get index of train samples and test samples
        train_idx = data[data['Sample'].isin(train_sample)].index.tolist()
        test_idx = data[data['Sample'].isin(test_sample)].index.tolist()
        
        # conduct SMOTE test for Selected features data
        sm = SMOTE(k_neighbors=1,random_state=args.seed)
        features_smote, labels_smote = sm.fit_resample(features[train_idx],labels[train_idx])
        print ('Shape of oversampled data: {}'.format(features_smote.shape))
        print ('Shape of Y: {}'.format(labels_smote.shape))
        fea_smote=torch.from_numpy(features_smote)
        lab_smote=torch.from_numpy(labels_smote)
        
        # Class distribution after SMOTE 
        counter = Counter(labels_smote)
        label, values = zip(*counter.items()) 
        
        # Create a bar plot
        plt.bar(label, values)

        # Add labels and title
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.title('Distribution of Classes')

        # Display the plot
        plt.savefig('static/images/SMOTE_selected_features_1300.png')
        plt.close()
        
        # conduct SMOTE test for SNF data integration result
        sm = SMOTE(k_neighbors=1,random_state=args.seed)
        adjacent_smote, labels_smote = sm.fit_resample(adj[train_idx],labels[train_idx])
        print ('Shape of oversampled data: {}'.format(adjacent_smote.shape))
        print ('Shape of Y: {}'.format(labels_smote.shape))
        adj_smote=torch.from_numpy(adjacent_smote)
        lab_smote=torch.from_numpy(labels_smote)
        
        counter = Counter(labels_smote)
        label, values = zip(*counter.items()) 
        
        # Create a bar plot
        plt.bar(label, values)

        # Add labels and title
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.title('Distribution of Classes')

        # Display the plot
        plt.savefig('static/images/SMOTE_adjacent_matrix.png')
        plt.close()
              
        GCN_model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, dropout=args.dropout)
        GCN_model.to(device)
        optimizer = torch.optim.Adam(GCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)
        idx_train, idx_test = torch.tensor(train_idx, dtype=torch.long, device=device), torch.tensor(test_idx, dtype=torch.long, device=device)

        '''
        save a best model (with the minimum loss value)
        if the loss didn't decrease in N epochs，stop the train process.
        N can be set by args.patience 
        '''
        loss_values, acc_values, f1_values, precision_values, recall_values, mcc_values, mse_values= [], [], [], [], [], [], []    
        acc_test_vals, f1_test_vals, precision_test_vals, recall_test_vals, loss_test_vals, mcc_test_vals, mse_test_vals= [], [], [], [], [], [], []
        loss_values_smote, acc_values_smote, f1_values_smote, precision_values_smote, recall_values_smote, mcc_values_smote, mse_values_smote= [], [], [], [], [], [], []    
        acc_test_vals_smote, f1_test_vals_smote, precision_test_vals_smote, recall_test_vals_smote, loss_test_vals_smote, mcc_test_vals_smote, mse_test_vals_smote= [], [], [], [], [], [], []
        
        # record the times with no loss decrease, record the best epoch
        bad_counter, best_epoch, curr_epoch= 0, 0, 0
        best = 1000   
        # record the lowest loss value & obtain the evaluation results from training and testing.
        for epoch in range(args.epochs):
            acc, f1, precision, recall, mcc, loss, mse= train(epoch, optimizer, features, adj, labels, idx_train)
            acc_values.append(acc)
            f1_values.append(f1)
            precision_values.append(precision)
            recall_values.append(recall)
            mcc_values.append(mcc)
            loss_values.append(loss)
            mse_values.append(mse)
            
            ac_test, f1_test, prec_test, rec_test, mcc_test, ls_test, mse_test= test(features, adj, labels, idx_test)
            acc_test_vals.append(ac_test)
            f1_test_vals.append(f1_test)
            precision_test_vals.append(prec_test)
            recall_test_vals.append(rec_test)
            mcc_test_vals.append(mcc_test)
            loss_test_vals.append(ls_test)
            mse_test_vals.append(mse_test)
            
            acc_smote, f1_smote, precision_smote, recall_smote, mcc_smote, loss_smote, mse_smote= train(epoch, optimizer, fea_smote, adj_smote.t(), lab_smote, idx_train)
            acc_values_smote.append(acc_smote)
            f1_values_smote.append(f1_smote)
            precision_values_smote.append(precision_smote)
            recall_values_smote.append(recall_smote)
            mcc_values_smote.append(mcc_smote)
            loss_values_smote.append(loss_smote)
            mse_values_smote.append(mse_smote)
            
            ac_test_smote, f1_test_smote, prec_test_smote, rec_test_smote, mcc_test_smote, ls_test_smote, mse_test_smote= test(features, adj, labels, idx_test)
            acc_test_vals_smote.append(ac_test_smote)
            f1_test_vals_smote.append(f1_test_smote)
            precision_test_vals_smote.append(prec_test_smote)
            recall_test_vals_smote.append(rec_test_smote)
            mcc_test_vals_smote.append(mcc_test_smote)
            loss_test_vals_smote.append(ls_test_smote)
            mse_test_vals_smote.append(mse_test_smote)
            
            if loss_values_smote[-1] < best:
                best = loss_values_smote[-1]
                best_epoch = epoch
                bad_counter = 0
            # if loss_values[-1] < best:
            #     best = loss_values[-1]
            #     best_epoch = epoch
            #     bad_counter = 0
            else:
                bad_counter += 1     # In this epoch, the loss value didn't decrease

            if bad_counter == args.patience:
                curr_epoch = epoch+1
                break
            else: 
                curr_epoch = args.epochs
        

        print('Training finished.')
        print('The best epoch model is ',best_epoch+1)
        
        # get the total epoch displayed in graph
        epoch_plt = range(0, curr_epoch)
        
        
        # training acc graph (SMOTE + Normal)
        plt.subplot()
        plt.plot(epoch_plt, acc_values_smote, 'b', label='Accuracy (SMOTE)')
        plt.plot(epoch_plt, acc_values, 'r', label='Accuracy (Normal)')
        plt.title('SMOTE and Normal Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend() 
        plt.savefig('static/images/training-acc-smote-1300-target-all.png')
        plt.close()
        
        # training f1 graph (SMOTE + Normal)
        plt.subplot()
        plt.plot(epoch_plt, f1_values_smote, 'b', label='F1 score (SMOTE)')
        plt.plot(epoch_plt, f1_values, 'r', label='F1 score (Normal)')
        plt.title('SMOTE and Normal F1 score')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend() 
        plt.savefig('static/images/training-f1-smote-1300-target-all.png')
        plt.close()
        
        # training precision graph (SMOTE + Normal)
        plt.subplot()
        plt.plot(epoch_plt, precision_values_smote, 'b', label='Precision (SMOTE)')
        plt.plot(epoch_plt, precision_values, 'r', label='Precision (Normal)')
        plt.title('SMOTE and Normal Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend() 
        plt.savefig('static/images/training-precs-smote-1300-target-all.png')
        plt.close()
        
        # training recall graph (SMOTE + Normal)
        plt.subplot()
        plt.plot(epoch_plt, recall_values_smote, 'b', label='Recall (SMOTE)')
        plt.plot(epoch_plt, recall_values, 'r', label='Recall (Normal)')
        plt.title('SMOTE and Normal Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend() 
        plt.savefig('static/images/training-recall-smote-1300-target-all.png')
        plt.close()
        
        # training loss graph (SMOTE + Normal)
        plt.subplot()
        plt.plot(epoch_plt, loss_values_smote, 'b', label='Loss (SMOTE)')
        plt.plot(epoch_plt, loss_values, 'r', label='Loss (Normal)')
        plt.title('SMOTE and Normal Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend() 
        plt.savefig('static/images/train-loss-smote-1300-target-all.png')
        plt.close()
        
        # training mse graph (SMOTE + Normal)
        plt.subplot()
        plt.plot(epoch_plt, mse_values_smote, 'b', label='MSE (SMOTE)')
        plt.plot(epoch_plt, mse_values, 'r', label='MSE (Normal)')
        plt.title('SMOTE and Normal MSE')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend() 
        plt.savefig('static/images/train-mse-smote-1300-target-all.png')
        plt.close()
        
        # training MCC graph (SMOTE + Normal)
        plt.subplot()
        plt.plot(epoch_plt, mcc_values_smote, 'b', label='MCC (SMOTE)')
        plt.plot(epoch_plt, mcc_values, 'r', label='MCC (Normal)')
        plt.title('SMOTE and Normal Matthews Correlation Coefficient (MCC)')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend()
        plt.savefig('static/images/train-mcc-smote-1300-target-all.png')
        plt.close()
        
        
        # training and testing loss graph (SMOTE) - check model underfitting / overfitting
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(epoch_plt, loss_values_smote, 'b', label='Training Loss')
        plt.plot(epoch_plt, loss_test_vals_smote, 'r', label='Testing Loss')
        plt.title('Training and Testing Loss (SMOTE)')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(epoch_plt, loss_values, 'b', label='Training Loss')
        plt.plot(epoch_plt, loss_test_vals, 'r', label='Testing Loss')
        plt.title('Training and Testing Loss (Normal)')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend()
        plt.savefig('static/images/train-test-loss-smote-1300-target-all.png')
        plt.close()
        
        
        # training acc + Loss graph (SMOTE + Normal) 
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(epoch_plt, acc_values_smote, 'b', label='Accuracy')
        plt.plot(epoch_plt, loss_values_smote, 'r', label='Loss')
        plt.title('Accuracy and Loss (SMOTE)')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(epoch_plt, acc_values, 'b', label='Accuracy')
        plt.plot(epoch_plt, loss_values, 'r', label='Loss')
        plt.title('Accuracy and Loss (Normal)')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend() 
        plt.savefig('static/images/train-acc-loss-smote-1300-target-all.png')
        plt.close()
       
       
        # train acc + MSE graph (SMOTE + Normal) 
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(epoch_plt, acc_values_smote, 'b', label='Accuracy')
        plt.plot(epoch_plt, mse_values_smote, 'r', label='Mean Square Error (MSE)')
        plt.title('Accuracy and MSE (SMOTE)')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(epoch_plt, acc_values, 'b', label='Accuracy')
        plt.plot(epoch_plt, mse_values, 'r', label='Mean Square Error (MSE)')
        plt.title('Accuracy and MSE (Normal)')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend() 
        plt.savefig('static/images/train-acc-mse-smote-1300-target-all.png')
        plt.close()
        
        
        # training acc + precision graph (SMOTE + Normal)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(epoch_plt, acc_values_smote, 'b', label='Accuracy')
        plt.plot(epoch_plt, precision_values_smote, 'r', label='Precision')
        plt.title('Accuracy and Precision (SMOTE)')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(epoch_plt, acc_values, 'b', label='Accuracy')
        plt.plot(epoch_plt, precision_values, 'r', label='Precision')
        plt.title('Accuracy and Precision (Normal)')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.legend()
        plt.savefig('static/images/train-acc-precision-smote-1300-target-all.png')
        plt.close()
        

    print('Finished!')