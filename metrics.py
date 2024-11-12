import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

def get_df(sensitive_attributes, output, label):
    df_dict = {}
    df_dict["sensitive"] = sensitive_attributes
    df_dict['y_hat'] = list(output)
    df_dict['y'] = list(label)
    df = pd.DataFrame.from_dict(df_dict)
    return df

def accuracy(output, label):
    return np.mean(np.array(output) == np.array(label))

def demographic_parity_violation_binary(sensitive_attributes, output, label):
    df = get_df(sensitive_attributes, output, label)
    positive_value_rates = []
    for attr in set(sensitive_attributes):
        positive_value_rates.append(len(df.loc[(df['sensitive'] == attr) & (df['y_hat'] == 1)].index)/len(df.loc[df['sensitive'] == attr].index))
    positive_value_rates = sorted(positive_value_rates)
    maximum_diff = positive_value_rates[-1] - positive_value_rates[0]
    return maximum_diff

def equalized_odds_violation_binary(sensitive_attributes, output, label):
    df = get_df(sensitive_attributes, output, label)
    true_positive_rates = []
    false_positive_rates = []
    for attr in set(sensitive_attributes):
        true_positive_rates.append(len(df.loc[(df['sensitive'] == attr) & (df['y_hat'] == 1) & (df['y'] == 1)].index)/len(df.loc[(df['sensitive'] == attr) & (df['y'] == 1)].index))
        false_positive_rates.append(len(df.loc[(df['sensitive'] == attr) & (df['y_hat'] == 1) & (df['y'] == 0)].index)/len(df.loc[(df['sensitive'] == attr) & (df['y'] == 0)].index))
    
    true_positive_rates = sorted(true_positive_rates)
    false_positive_rates = sorted(false_positive_rates)

    max_diff_tpr = true_positive_rates[-1] - true_positive_rates[0]
    max_diff_fpr = false_positive_rates[-1] - false_positive_rates[0]
    return max(max_diff_tpr, max_diff_fpr)

def demographic_parity_violation_multiple(sensitive_attributes, output, label):
    total_unique_labels = set(label)
    demographic_parity = 0
    for i in total_unique_labels:
        labels_temp = (torch.tensor(label) == i).float().tolist()
        output_temp = (torch.tensor(output) == i).float().tolist()
        demographic_parity = max(demographic_parity, demographic_parity_violation_binary(sensitive_attributes, output_temp, labels_temp))
    return demographic_parity

def equalized_odds_violation_multiple(sensitive_attributes, output, label):
    total_unique_labels = set(label)
    equalized_odds = 0
    for i in total_unique_labels:
        labels_temp = (torch.tensor(label) == i).float().tolist()
        output_temp = (torch.tensor(output) == i).float().tolist()
        equalized_odds = max(equalized_odds, equalized_odds_violation_binary(sensitive_attributes, output_temp, labels_temp))
    return equalized_odds


def plot_losses(args,all_epoch_losses, all_epoch_losses_fermi, all_epoch_losses_classification ):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot for all_epoch_losses_classification
    axs[0].plot(all_epoch_losses_classification, marker='o', linestyle='-', color='b', label='Classification Loss')
    axs[0].set_xlabel('Index (Epoch)')
    axs[0].set_ylabel('Value (Loss)')
    axs[0].set_title('Classification Loss per Epoch')
    axs[0].legend()

    # Plot for all_epoch_losses_fermi
    axs[1].plot(all_epoch_losses_fermi, marker='o', linestyle='-', color='g', label='Fermi Loss')
    axs[1].set_xlabel('Index (Epoch)')
    axs[1].set_ylabel('Value (Loss)')
    axs[1].set_title('Fermi Loss per Epoch')
    axs[1].legend()

    # Plot for combined all_epoch_losses
    axs[2].plot(all_epoch_losses, marker='o', linestyle='-', color='r', label='Average Loss')
    axs[2].set_xlabel('Index (Epoch)')
    axs[2].set_ylabel('Value (Loss)')
    axs[2].set_title('Average Loss per Epoch')
    axs[2].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{args.folder_name}/plots/{args.prefix}output_eps_{args.epsilon}_lr_W_{args.lr_W}_lr_theta_{args.lr_theta_initial}_dp_{args.demographic_parity}_C_{args.C}_lip_theta_{args.lipschitz_theta}.png")
    plt.close()
    
    return

def calculate_final_mean_metrics(lambda_model_num_dict, lambda_values, eval_index_start=0):
    for lambda_value in lambda_values:
        if lambda_value in lambda_model_num_dict:
            models = lambda_model_num_dict[lambda_value]

            # Initialize accumulators for the required values
            total_average_misclassification_loss = 0
            total_average_demographic_parity = 0
            total_average_equalized_odds = 0

            model_count = len(models)

            # Iterate over all models for the specified lambda
            for model_number, model_data in models.items():
                total_average_misclassification_loss += statistics.mean(model_data["misclassification_error_list"][eval_index_start:])
                total_average_demographic_parity += statistics.mean(model_data["demographic_parity_list"][eval_index_start:])
                total_average_equalized_odds += statistics.mean(model_data["equalized_odds_list"][eval_index_start:])

            # Calculate means
            mean_average_misclassification_loss = total_average_misclassification_loss / model_count
            mean_average_demographic_parity = total_average_demographic_parity / model_count
            mean_average_equalized_odds = total_average_equalized_odds / model_count

            # Print results
            print(f"Lambda value: {lambda_value}", flush=True)
            print(f"  Mean misclassification loss: {mean_average_misclassification_loss}", flush=True)
            print(f"  Mean demographic parity: {mean_average_demographic_parity}", flush=True)
            print(f"  Mean equalized odds: {mean_average_equalized_odds}", flush=True)
            print("")
        else:
            print(f"No data available for lambda value: {lambda_value}", flush=True)
            
            

def calculate_final_median_metrics(lambda_model_num_dict, lambda_values, fairness_metric, both = True, eval_index_start = 0):
    
    if both:
        calculate_final_mean_metrics(lambda_model_num_dict, lambda_values)
    
    for lambda_value in lambda_values:
        if lambda_value in lambda_model_num_dict:
            models = lambda_model_num_dict[lambda_value]

            # Lists to hold the values that meet the criteria
            filtered_misclassification_loss = []
            filtered_fairness_metric = []

            # Iterate over all models for the specified lambda
            for model_number, model_data in models.items():
                demographic_parity_list = model_data["demographic_parity_list"]
                equalized_odds_list = model_data["equalized_odds_list"]
                misclassification_error_list = model_data["misclassification_error_list"]

                for index, (misclassification_loss, demographic_parity, equalized_odds) in enumerate(zip(misclassification_error_list, demographic_parity_list, equalized_odds_list)):
                    if index >= eval_index_start:
                        if fairness_metric == "demographic_parity":
                            filtered_misclassification_loss.append(misclassification_loss)
                            filtered_fairness_metric.append(demographic_parity)
                        elif fairness_metric == "equalized_odds":
                            filtered_misclassification_loss.append(misclassification_loss)
                            filtered_fairness_metric.append(equalized_odds)

            # Calculate medians
            if filtered_misclassification_loss and filtered_fairness_metric:
                median_misclassification_loss = np.median(filtered_misclassification_loss)
                median_fairness_metric = np.median(filtered_fairness_metric)
                
                # Print results
                print(f"Lambda value: {lambda_value}", flush=True)
                print(f"  Median misclassification loss: {median_misclassification_loss}", flush=True)
                print(f"  Median {fairness_metric}: {median_fairness_metric}", flush=True)
                print("")
            else:
                print(f"No data meeting the criteria for lambda value: {lambda_value}", flush=True)
        else:
            print(f"No data available for lambda value: {lambda_value}", flush=True)


