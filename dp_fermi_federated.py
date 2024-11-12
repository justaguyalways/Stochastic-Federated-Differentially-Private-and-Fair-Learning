import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm
from pathlib import Path
from math import ceil

from args_fed import Args
from models import *
from dataloader_fed import *
from federated_sampler import *
from federated_dp_sgd import *
from metrics import *
from opacus.grad_sample import GradSampleModule
import statistics
import matplotlib.pyplot as plt
import sys
import json
import os

if __name__ == '__main__':
    # Initialize the arguments from Args class and assign default values
    args = Args()
    args.assign()
    
    # Set environment variables for CUDA devices and world size for distributed learning
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["WORLD_SIZE"] = "1"

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Redirect standard output to a log file in the specified folder
    original_stdout = sys.stdout
    output_file = f"{args.folder_name}/{args.prefix}output_eps_{args.epsilon}_lr_W_{args.lr_W}_lr_theta_{args.lr_theta}_dp_{args.demographic_parity}_C_{args.C}_lip_theta_{args.lipschitz_theta}.txt"
    sys.stdout = open(output_file, "w")

    # Load datasets based on the selected dataset type
    if args.dataset in ["adult", "retired-adult", "credit-card", "parkinsons"]:
        # General dataset loading with specified parameters for federated learning
        full_data = GeneralData(path=args.path, random_state=args.random_state, 
                                num_silos=args.num_silos, silo_attribute=args.silo_attribute,
                                level_of_heterogeneity=args.level_of_heterogeneity, sensitive_attributes=args.sensitive_attributes, 
                                cols_to_norm=args.cols_to_norm, output_col_name=args.output_col_name, split=args.split, 
                                min_per_silo=args.min_per_silo)
        
        # Training and test datasets for federated learning
        dataset_train_silos_list = full_data.getTrain()
        dataset_test = full_data.getTest()
    else:
        # If dataset is not one of the above, load the UTKFace dataset
        dataset_train = UTKFaceDataset(split=args.split)
        dataset_test = UTKFaceDataset(train=False, split=args.split)      

    # Calculate the number of batches per epoch (account for any remaining samples that don't fit into a full batch)
    args.number_of_batches_per_epoch = int(ceil(full_data.n_tilda / args.batch_size))

    # Iterating over different learning rates and epsilon values for hyperparameter tuning
    for args.lr_theta in args.lr_theta_list:
        for args.lr_W in args.lr_W_list:
            for args.epsilon in args.epsilon_list:
                # Recalculate noise scale based on the current epsilon
                args.calculate_noise()
                print(f'''Learning Rate Theta: {args.lr_theta} Learning Rate W: {args.lr_W} Epsilon: {args.epsilon} \n 
                      C: {args.C} lipschitz_theta: {args.lipschitz_theta} Batch size: {args.batch_size} \n
                      Std Theta: {args.std_theta} Std W: {args.std_W} \n ''')
                
                # Display additional parameters
                print(f"Level of heterogeneity: {args.level_of_heterogeneity}", flush=True)
                
                # Dictionary to store results for different lambda values and model numbers
                lambda_model_num_dict = {}
                for args.lambd in args.lambd_list:
                    lambda_model_num_dict[args.lambd] = {}
                    data_table = {}

                    for model_number in range(args.num_models_train):
                        # Reset learning rate for theta after each model training session
                        args.lr_theta = args.lr_theta_initial
                        
                        print(f"Lambda: {args.lambd} \n Model number: {model_number}", flush=True)
                        
                        # Set random seed for model initialization
                        torch.manual_seed(model_number)
                        lambda_model_num_dict[args.lambd][model_number] = {}
                        
                        # Initialize data table for fairness metrics
                        data_table[model_number] = {
                            "demographic_parity_list": [],
                            "equalized_odds_list": [],
                            "equalized_opportunity_list": [],
                            "misclassification_error_list": []
                        }
                        
                        # Federated data loading for training based on the tuning parameter
                        if args.tuning:
                            federated_dataloader_train = FederatedBatchSampler(datasets=dataset_train_silos_list, 
                                                                               batch_size=args.batch_size)
                            dataloader_test = Data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, 
                                                              num_workers=args.num_workers)
                        else:
                            federated_dataloader_train = FederatedBatchSampler(datasets=dataset_train_silos_list, 
                                                                               batch_size=args.batch_size)
                            dataloader_test = Data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, 
                                                              num_workers=args.num_workers)
                            
                        # Initialize the model based on the type of model specified in args
                        if args.model_type == "logistic-regression":
                            model = GradSampleModule(LogisticRegression(args.num_inp_attr))
                        elif args.model_type == "neural-network":
                            model = NeuralNetwork(args.num_inp_attr, args.out_attr, args.num_layers)
                        elif args.model_type == "cnn-classifier":
                            model = FeedForward(args.out_attr)
                        
                        model = model.to(device)

                        # Set loss function based on model type
                        if args.model_type == "logistic-regression":
                            classification_loss_fn = nn.BCEWithLogitsLoss(reduction="none").to(device)
                        else:
                            classification_loss_fn = nn.CrossEntropyLoss(reduction="none").to(device)
                            
                        # Initialize parameter W for gradient calculations
                        W = nn.Parameter(data=torch.zeros(args.out_attr, args.out_attr).to(device), requires_grad=True)
                        if not args.demographic_parity:
                            W_ = nn.Parameter(data=torch.zeros(args.out_attr, args.out_attr).to(device), requires_grad=True)
            
                        # Calculate the common P_s matrix depending on the dataset and demographic parity setting
                        if args.model_type == "logistic-regression":
                            P_s_negative_half = full_data.calculateP_s(args.demographic_parity)
                        else:
                            P_s_negative_half = dataset_train.calculateP_s(args.demographic_parity)
                        
                        if args.demographic_parity:
                            P_s_negative_half = P_s_negative_half.to(device)
                        else:
                            P_s_negative_half[0] = P_s_negative_half[0].to(device)
                            P_s_negative_half[1] = P_s_negative_half[1].to(device)
                        
                        # Initialize dictionaries to store gradients per silo for both theta and W
                        model_grad_silos = {}
                        for name, param in model.named_parameters():
                            model_grad_silos[name] = torch.zeros((args.num_silos, *param.shape)).to(device)
                            
                        W_grad_silos = {"W": torch.zeros((args.num_silos, *W.shape)).to(device)}
                        if not args.demographic_parity:
                            W_grad_silos["W_"] = torch.zeros((args.num_silos, *W.shape)).to(device)
                            
                        # Lists to store loss metrics during training
                        all_epoch_losses, all_epoch_losses_fermi, all_epoch_losses_classification = [], [], []
                        
                        # Start training for the specified number of epochs
                        for epoch in tqdm(range(args.epochs), f"Lambd: {args.lambd} Model {model_number}"):
                            # Reset federated data iterator at the start of each epoch
                            federated_dataloader_train.start_new_epoch()

                            epoch_loss_fermi, epoch_loss_classification = [], []
                            for batch_set_no in range(args.number_of_batches_per_epoch):
                                
                                with torch.no_grad(): 
                                    # Zeroing gradients for W and W_ across silos
                                    if W.grad is not None:
                                        W.grad.zero_()
                                    if not args.demographic_parity:
                                        if W_.grad is not None:
                                            W_.grad.zero_()

                                    # Reset gradient buffers for W and theta for each silo
                                    W_grad_silos["W"] = torch.zeros((args.num_silos, *W.shape)).to(device)
                                    if not args.demographic_parity:
                                        W_grad_silos["W_"] = torch.zeros((args.num_silos, *W.shape)).to(device)
                                    
                                    for name, param in model.named_parameters():
                                        model_grad_silos[name] = torch.zeros((args.num_silos, *param.shape)).to(device)
                                
                                # Zero gradients for theta
                                model.zero_grad()
                                
                                # Lists to store loss values for each silo
                                silo_losses_fermi, silo_losses_classification = [], []
                                
                                # Iterate over silos and compute gradients for each silo
                                for silo_id in range(args.num_silos):
                                    non_sensitive, sensitive, label, _ = federated_dataloader_train.sample_batch_from_dataset(silo_id)
                                    
                                    # Reset model gradients
                                    model.zero_grad()
                                    
                                    with torch.no_grad(): 
                                        # Zeroing W grad actual
                                        if W.grad is not None:
                                            W.grad.zero_()
                                        if not args.demographic_parity:
                                            if W_.grad is not None:
                                                W_.grad.zero_()
                                    
                                    if args.demographic_parity:
                                        W_ = None  # No second W matrix if demographic parity is used
                                        
                                    # Compute gradients for the current silo
                                    model_grad_per_silo, W_grad_per_silo, silo_loss_fermi, silo_loss_classification = compute_grad_per_silo(
                                        args=args, device=device, non_sensitive=non_sensitive, sensitive=sensitive, 
                                        label=label, P_s_negative_half=P_s_negative_half, classification_loss_fn=classification_loss_fn,
                                        model=model, W=W, W_=W_)
                                    
                                    # Accumulate losses for current silo
                                    with torch.no_grad():
                                        silo_losses_classification.append(silo_loss_classification)
                                        silo_losses_fermi.append(silo_loss_fermi)
                                        
                                        # Store per-silo gradients in model_grad_silos and W_grad_silos
                                        for name, param in model.named_parameters():
                                            model_grad_silos[name][silo_id] = model_grad_per_silo[name].to(device)
    
                                        W_grad_silos["W"][silo_id] = W_grad_per_silo["W"].to(device)
                                        if not args.demographic_parity:
                                            W_grad_silos["W_"][silo_id] = W_grad_per_silo["W_"].to(device)

                                # Aggregate losses across all silos for the epoch
                                epoch_loss_classification.append(statistics.mean(silo_losses_classification))
                                epoch_loss_fermi.append(statistics.mean(silo_losses_fermi))
                                
                                # Post-silo gradient updates (Federated Averaging)
                                with torch.no_grad():
                                    # Update theta parameters by averaging gradients across silos
                                    for name, param in model.named_parameters():
                                        model_grad_silos[name] = model_grad_silos[name].to(device)
                                        param.sub_(args.lr_theta * model_grad_silos[name].mean(dim=0))
                                    
                                    # Update W parameter by averaging gradients across silos
                                    W_grad_silos["W"] = W_grad_silos["W"].to(device)
                                    W.add_(args.lr_W * W_grad_silos["W"].mean(dim=0))
                                    
                                    # Update W_ parameter (if applicable)
                                    if not args.demographic_parity:
                                        W_grad_silos["W_"] = W_grad_silos["W_"].to(device)
                                        W_.add_(args.lr_W * W_grad_silos["W_"].mean(dim=0))
                                    
                                    # Project W back into the convex constraint set (norm clipping)
                                    norm_W = torch.norm(W.data)
                                    if norm_W > args.C:
                                        W.copy_(args.C * W.data / norm_W)
                                    
                                    if not args.demographic_parity:
                                        norm_W_ = torch.norm(W_.data)
                                        if norm_W_ > args.C:
                                            W_.copy_(args.C * W_.data / norm_W_)
                                    
                                    # Zero out actual gradients for W and theta after updates
                                    if W.grad is not None:
                                        W.grad.zero_()
                                    if not args.demographic_parity:
                                        if W_.grad is not None:
                                            W_.grad.zero_()
                                            
                                    # Zero out gradient buffers for W and theta
                                    W_grad_silos["W"] = torch.zeros((args.num_silos, *W.shape)).to(device)
                                    if not args.demographic_parity:
                                        W_grad_silos["W_"] = torch.zeros((args.num_silos, *W.shape)).to(device)
                                    
                                    for name, param in model.named_parameters():
                                        model_grad_silos[name] = torch.zeros((args.num_silos, *param.shape)).to(device)
                                
                                # Zero actual model gradients for the next iteration
                                model.zero_grad()
                            
                            # Append the average classification loss and Fermi loss for the current epoch
                            all_epoch_losses_classification.append(statistics.mean(epoch_loss_classification))
                            all_epoch_losses_fermi.append(statistics.mean(epoch_loss_fermi))
                            all_epoch_losses.append(all_epoch_losses_classification[-1] + all_epoch_losses_fermi[-1])

                            # Plot the losses at each epoch
                            plot_losses(args, all_epoch_losses, all_epoch_losses_fermi, all_epoch_losses_classification)
                                                         
                            # Decay the learning rate for theta after a specified step
                            if (epoch + 1) % args.lr_theta_decay_step == 0:
                                args.lr_theta *= args.lr_theta_decay_rate
                        
                            # Evaluate model performance every 'eval_epochs' epochs
                            if (epoch + 1) % args.eval_epochs == 0:
                                model.eval()  # Set model to evaluation mode

                                # Initialize lists to store evaluation metrics
                                sensitive_index_all = []
                                y_hat_all = []
                                label_all = []

                                # Iterate over the test set for evaluation
                                for non_sensitive, sensitive, label, sensitive_index in dataloader_test:
                                    non_sensitive = non_sensitive.to(device)
                                    sensitive = sensitive.to(device)
                                    label = label.to(device)
                                    sensitive_index = sensitive_index.to(device)

                                    # Perform inference without gradient calculations
                                    with torch.no_grad():
                                        y_logit, y_hat = model(non_sensitive.float())

                                    # Store predictions, labels, and sensitive indices
                                    sensitive_index_all.extend(sensitive_index.squeeze().tolist())
                                    if args.model_type == "logistic-regression":
                                        y_hat_all.extend((y_hat.detach().cpu() > 0.5).squeeze().tolist())  # Binarize logits for logistic regression
                                    else:
                                        y_hat_all.extend(y_hat.detach().cpu().squeeze().tolist())
                                    label_all.extend(label.squeeze().tolist())

                                # Compute fairness and performance metrics for the test set
                                if args.model_type == "logistic-regression":
                                    y_hat_all = [1 if u else 0 for u in y_hat_all]
                                    demographic_parity = demographic_parity_violation_binary(sensitive_index_all, y_hat_all, label_all)
                                    equalized_odds = equalized_odds_violation_binary(sensitive_index_all, y_hat_all, label_all)
                                    misclassification_error = 1 - accuracy(y_hat_all, label_all)
                                else:
                                    demographic_parity = demographic_parity_violation_multiple(sensitive_index_all, y_hat_all, label_all)
                                    equalized_odds = equalized_odds_violation_multiple(sensitive_index_all, y_hat_all, label_all)
                                    misclassification_error = 1 - accuracy(y_hat_all, label_all)

                                # Print the computed metrics
                                # print(f"Demographic Parity: {demographic_parity}", flush=True)
                                # print(f"Equalized Odds: {equalized_odds}", flush=True)
                                # print(f"Misclassification Error: {misclassification_error}", flush=True)
                                # print("", flush=True)

                                model.train()  # Switch back to training mode

                                # Store the computed metrics in the data table
                                data_table[model_number]["demographic_parity_list"].append(demographic_parity)
                                data_table[model_number]["equalized_odds_list"].append(equalized_odds)
                                data_table[model_number]["misclassification_error_list"].append(misclassification_error)

                        # After all models are trained, compute and store evaluation metrics
                        demographic_parity_list = torch.tensor(data_table[model_number]["demographic_parity_list"])
                        equalized_odds_list = torch.tensor(data_table[model_number]["equalized_odds_list"])
                        misclassification_error_list = torch.tensor(data_table[model_number]["misclassification_error_list"])
                        

                        # Compute average, minimum values for the metrics
                        average_demographic_parity = torch.mean(demographic_parity_list).item()
                        average_equalized_odds = torch.mean(equalized_odds_list).item()
                        average_misclassification_loss = torch.mean(misclassification_error_list).item()

                        # Store evaluation results for the current lambda and model number
                        lambda_model_num_dict[args.lambd][model_number] = {
                            "average_demographic_parity": average_demographic_parity,
                            "average_equalized_odds": average_equalized_odds,
                            "average_misclassification_loss": average_misclassification_loss,
                            "demographic_parity_list": data_table[model_number]["demographic_parity_list"],
                            "equalized_odds_list": data_table[model_number]["equalized_odds_list"],
                            "misclassification_error_list": data_table[model_number]["misclassification_error_list"],
                        }
                    
                    # Determine the starting index for evaluation
                    eval_index_start = args.eval_start / args.eval_epochs - 1
                    
                    # Compute final evaluation metrics based on mean or median
                    if args.metric_type == "mean":    
                        calculate_final_mean_metrics(lambda_model_num_dict, [args.lambd], eval_index_start)     
                    else:
                        # Choose the fairness metric based on demographic parity or equalized odds
                        fairness_metric = "demographic_parity" if args.demographic_parity else "equalized_odds"
                        calculate_final_median_metrics(lambda_model_num_dict, [args.lambd],
                                                      fairness_metric, args.both, eval_index_start)

                    # Save the results for the current configuration to a JSON file
                    with open(f'{args.folder_name}/dumps/{args.prefix}output_eps_{args.epsilon}_lr_W_{args.lr_W}_lr_theta_{args.lr_theta_initial}_dp_{args.demographic_parity}_C_{args.C}_lip_theta_{args.lipschitz_theta}.json', 'w') as json_file:
                        json.dump(lambda_model_num_dict, json_file, indent=4)

                # Compute final evaluation metrics for all lambda values
                if args.metric_type == "mean":    
                    calculate_final_mean_metrics(lambda_model_num_dict, args.lambd_list, eval_index_start)     
                else:
                    # Choose the fairness metric based on demographic parity or equalized odds
                    fairness_metric = "demographic_parity" if args.demographic_parity else "equalized_odds"
                    calculate_final_median_metrics(lambda_model_num_dict, args.lambd_list,
                                                  fairness_metric, args.both, eval_index_start)
                
                # Save the final results to a JSON file after all lambda values are processed
                with open(f'{args.folder_name}/dumps/{args.prefix}output_eps_{args.epsilon}_lr_W_{args.lr_W}_lr_theta_{args.lr_theta_initial}_dp_{args.demographic_parity}_C_{args.C}_lip_theta_{args.lipschitz_theta}.json', 'w') as json_file:
                    json.dump(lambda_model_num_dict, json_file, indent=4)

