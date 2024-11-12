import numpy as np
from dataloader_fed import *

class Args():
    def __init__(self):
        # General configuration flags
        self.debug = True               # Debug mode flag
        self.tuning = False             # Flag for enabling/disabling tuning
        
        # Model training settings
        self.num_models_train = 15      # Number of models for training
        self.demographic_parity = None  # True for demographic parity, False for equalized odds experiments
        self.epochs = 60                # Number of training epochs
        self.batch_size = 256           # Batch size for training
        
        # Device and file settings
        self.device = "5"                # Device identifier (e.g., GPU ID)
        self.folder_name = "json_dumps"  # Folder name for storing results/logs
        
        # Worker and model settings
        self.num_workers = 2            # Number of worker threads for data loading
        self.split = 0.75               # Train-test split ratio
        self.num_layers = 1             # Number of layers for the model architecture

        # Federated learning settings
        '''
        level_of_heterogenity: 0 for homogeneous case
        Higher level_of_heterogeneity values indicate more heterogeneity.
        '''
        self.num_silos = None               # Number of silos in federated learning
        self.level_of_heterogeneity = None  # Heterogeneity level, lower value means higher heterogeneity
        self.random_state = 100             # Random seed for reproducibility
        self.min_per_silo = 10              # Minimum samples per silo (depends on dataset and number of silos)
        
        # Evaluation and metric settings based on the number of silos
        if self.num_silos == 1:
            self.eval_epochs = 3            # Evaluation frequency for single silo
            self.epochs = 15                # Adjust epochs for single silo case
        else:
            self.eval_epochs = 10           # Evaluation frequency for multiple silos
        
        self.eval_start = 20              # Start evaluation after this epoch
        self.metric_type = "mean"         # Metric type, can be "mean" or "median"
        self.both = False                 # Whether to compute both mean and median

        # Privacy-related parameters
        self.epsilon = None               # Epsilon for differential privacy, set to infinity (no privacy constraint)
        self.epsilon_list = None          # List of epsilon values (if tuning is enabled)

        # Calculated parameters (do not require tuning)
        self.delta = None                   # Delta for differential privacy (to be computed)
        self.std_theta = None               # Standard deviation for theta (to be computed)
        self.std_W = None                   # Standard deviation for W (to be computed)

        # Parameters requiring tuning (for each pair of {lambda, epsilon})
        self.C = None                         # Constant for regularization 
        self.lipschitz_theta = None           # Lipschitz constant for theta

        # Learning rates and decay settings (Need tuning as per the dataset)
        self.lr_theta = None                # Learning rate for theta
        self.lr_theta_decay_rate = None     # Learning rate decay rate for theta
        self.lr_theta_decay_step = None     # Step size for learning rate decay
        self.lr_theta_list = None           # List of theta learning rates

        self.lr_W = None                    # Learning rate for W
        self.lr_W_list = None               # List of W learning rates

        # Lambda values (used for weighting the fairness regularizer)
        self.lambd_list = [] # values generally between 0 to 2
        
        # Prefix for output filenames based on the configuration
        self.prefix = f"{self.eval_epochs}_eval_{self.num_silos}_silo_{self.epochs}_ep_{self.level_of_heterogeneity}_hetero_decay_{self.lr_theta_decay_rate}_{self.lr_theta_decay_step}_"

        # Dataset-specific settings
        self.dataset = ""      # Dataset to be used, one of ["adult", "retired-adult", "parkinsons", "credit-card"]
        self.lr_theta_initial = self.lr_theta  # Initial learning rate for theta
    
    def assign(self):
        # Model selection based on the number of layers
        if self.num_layers:
            if self.num_layers == 1:
                self.model_type = "logistic-regression"  # Use logistic regression if there's only one layer
            else:
                self.model_type = "neural-network"       # Use neural network if more than one layer
        else:
            self.model_type = "cnn-classifier"           # Default to CNN classifier

        # Default assignments for various parameters if not explicitly set
        if not self.lambd_list:
            self.lambd_list = [self.lambd]
        if not self.epsilon_list:
            self.epsilon_list = [self.epsilon]
        if not self.lr_W_list:
            self.lr_W_list = [self.lr_W]
        if not self.lr_theta_list:
            self.lr_theta_list = [self.lr_theta]
        
        # Dataset-specific attributes, normalization columns, and paths
        if self.dataset == "adult":
            self.num_inp_attr = 102
            self.silo_attribute = "age"
        elif self.dataset == "retired-adult":
            self.num_inp_attr = 101
            self.silo_attribute = "age"
        elif self.dataset == "credit-card":
            self.num_inp_attr = 85
            self.silo_attribute = "AGE"
        elif self.dataset == "parkinsons":
            self.num_inp_attr = 19
            self.silo_attribute = "age"
        elif self.dataset == "UTKFace":
            self.num_inp_attr = 512
        
        # Output attribute for specific datasets
        if self.dataset == "UTKFace":
            self.out_attr = 9
        else:
            self.out_attr = 1
        
        # Dataset paths
        if self.dataset == "adult":
            self.path = "./Datasets/Adult/adult_original_purified.csv"
        elif self.dataset == "retired-adult":
            self.path = "./Datasets/Adult/Retired-Adult/adult_reconstruction_processed.csv"
        elif self.dataset == "credit-card":
            self.path = "./Datasets/CreditCard/credit-card-defaulters_processed.csv"
        elif self.dataset == "parkinsons":
            self.path = "./Datasets/Parkinsons/parkinsons_updrs_processed.csv"
        
        # Columns to normalize based on dataset
        if self.dataset == "adult":
            self.cols_to_norm = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        elif self.dataset == "retired-adult":
            self.cols_to_norm = ["age", "hours-per-week", "education-num", "capital-gain", "capital-loss"]
        elif self.dataset == "credit-card":
            self.cols_to_norm = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", 
                                 "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
        elif self.dataset == "parkinsons":
            self.cols_to_norm = ['age', 'test_time', 'motor_UPDRS', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 
                                 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 
                                 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
        
        # Sensitive attributes based on dataset
        if self.dataset == "adult":
            self.sensitive_attributes = ["sex"]
        elif self.dataset == "retired-adult":
            self.sensitive_attributes = ["gender"]
        elif self.dataset == "credit-card":
            self.sensitive_attributes = ["SEX"]
        elif self.dataset == "parkinsons":
            self.sensitive_attributes = ["sex"]

        # Output column name based on dataset
        if self.dataset == "adult":
            self.output_col_name = ">50K"
        elif self.dataset == "retired-adult":
            self.output_col_name = "income"
        elif self.dataset == "credit-card":
            self.output_col_name = "default payment next month"
        elif self.dataset == "parkinsons":
            self.output_col_name = "total_UPDRS"

    def calculate_noise(self):
        # Load the dataset based on whether it is a federated or non-federated dataset
        if self.dataset in ["adult", "retired-adult", "credit-card", "parkinsons"]:
            full_data = GeneralData(path=self.path, random_state=self.random_state, num_silos=self.num_silos, silo_attribute=self.silo_attribute,
                                    level_of_heterogeneity=self.level_of_heterogeneity, sensitive_attributes=self.sensitive_attributes, 
                                    cols_to_norm=self.cols_to_norm, output_col_name=self.output_col_name, split=self.split, min_per_silo=self.min_per_silo)
            dataset_train_full = full_data.getTraindata_P_s()
        else:
            dataset_train_full = UTKFaceDataset()
        
        # Prepare the dataloader for training
        dataloader_train = Data.DataLoader(dataset_train_full, batch_size=self.batch_size, shuffle=True, num_workers=2)
        
        # Calculate privacy-related parameters
        n = full_data.n               # Total number of samples
        n_tilda = full_data.n_tilda   # Adjusted number of samples for differential privacy
        
        ''' Unchanged calculations (rho, delta) '''
        sensitive_index_all = []
        for non_sensitive, sensitive, label, sensitive_index in dataloader_train:
            sensitive_index_all.extend(sensitive_index.squeeze().tolist())
        
        count_dict = {}
        for attr in sensitive_index_all:
            try:
                count_dict[attr] += 1
            except KeyError:
                count_dict[attr] = 1
        
        # Calculate rho based on the smallest count of sensitive attributes
        min_count = n + 1
        for _, coun in count_dict.items():
            min_count = min(min_count, coun)
        
        rho = (min_count - 1) / n
        self.delta = 1 / n**2  # Differential privacy parameter delta
        
        ''' Changed calculations with n replaced by n_tilda '''
        # T = total number of iterations (epochs * batches per epoch)
        T = self.epochs * self.number_of_batches_per_epoch
        self.std_theta = ((16 * self.lipschitz_theta**2 * self.C**2 * np.log(1/self.delta) * T) / 
                          (self.epsilon**2 * n_tilda**2 * rho))**0.5
        self.std_W = ((16 * T * np.log(1/self.delta)) / 
                      (self.epsilon**2 * n_tilda**2 * rho))**0.5

