import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import transforms

import pandas as pd
import random
import numpy as np
from PIL import Image
import os
from copy import deepcopy

# Handling cases for the General Dataset:
#   - Format should be in CSV files with a single output column that is binary ("Yes" or "No").
#   - Categorical columns should be represented as strings.

class GeneralData():
    def __init__(self, path, random_state=100, num_silos=None, silo_attribute=None, 
                 level_of_heterogeneity=None, sensitive_attributes=None, cols_to_norm=None, split=0.75, output_col_name=None, 
                 min_per_silo=None):
        
        # Ensure necessary attributes are provided
        if not sensitive_attributes:
            raise Exception("No Sensitive Attributes Provided. Please provide one or more.")
        if not output_col_name:
            raise Exception("No output column name entered. Please provide one.")
        
        # Set random seeds for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)

        # Initialize base attributes
        self.output_col_name = output_col_name
        self.sensitive_attributes = sorted(sensitive_attributes)
        self.cols_to_norm = cols_to_norm
        
        # Initialize federated learning-specific attributes
        self.silo_attribute = silo_attribute
        self.num_silos = num_silos
        self.level_of_heterogeneity = level_of_heterogeneity
        self.min_per_silo = min_per_silo

        # Load and preprocess the dataset
        self.path = path
        df = pd.read_csv(self.path)
        
        # Convert output column ("Yes"/"No") to binary (1/0)
        df[output_col_name] = df[output_col_name].apply(lambda x: 1 if x.lower() == "yes" else 0)

        # Determine non-sensitive attributes by excluding sensitive attributes and output column
        if self.sensitive_attributes:
            non_sens_attr = sorted(list(set(df.columns).difference(set(self.sensitive_attributes + [output_col_name]))))
        else:
            non_sens_attr = sorted(list(df.columns).difference(set([output_col_name])))

        # One-hot encode categorical non-sensitive columns
        one_hot_cols = list(set(non_sens_attr).difference(self.cols_to_norm))
        df = pd.get_dummies(df, columns=one_hot_cols)
        
        # Convert boolean columns to float for compatibility
        bool_cols = df.select_dtypes(include=['bool']).columns
        df[bool_cols] = df[bool_cols].astype(float)
        
        # Store non-sensitive attributes for later use
        self.non_sens_attr = list(set(df.columns).difference(set(self.sensitive_attributes + [output_col_name])))

        # Split dataset into training and testing sets
        self.df = df
        self.df_train = df.sample(frac=split, random_state=random_state)
        self.n = len(self.df_train[self.silo_attribute])
        self.n_tilda = self.n // self.num_silos
        
        # Create copies of train and test datasets
        self.df_train_full = self.df_train.copy(deep=True)
        self.df_test = df.drop(self.df_train_full.index)
        
        # Normalize columns if specified
        if cols_to_norm:
            self.mean_train = self.df_train_full[cols_to_norm].mean()
            self.std_train = self.df_train_full[cols_to_norm].std()

            for col in cols_to_norm:
                self.df_train_full[col] = self.df_train_full[col].apply(lambda x: (x - self.mean_train[col]) / self.std_train[col])
                self.df_test[col] = self.df_test[col].apply(lambda x: (x - self.mean_train[col]) / self.std_train[col])

    def assign_partitions(self, df, column, n):
        """
        Assign partitions (silos) to the data based on the specified column.
        """
        # Sort the DataFrame by the specified column
        sorted_df = df.sort_values(by=column).reset_index()

        # Calculate partition size and create a new column to hold partition labels
        part_size = len(df) // n
        sorted_df['prior_silos'] = 0

        # Assign partition labels
        for i in range(n):
            start_idx = i * part_size
            end_idx = (i + 1) * part_size if i < n - 1 else len(df)
            sorted_df.loc[start_idx:end_idx, 'prior_silos'] = i + 1
        
        # Map partition labels back to the original DataFrame
        df_with_partitions = df.copy()
        df_with_partitions['prior_silos'] = sorted_df.set_index('index').loc[df.index, 'prior_silos']
        df_with_partitions['prior_silos'] = df_with_partitions['prior_silos'].apply(lambda x: x-1)

        self.df_with_partitions = df_with_partitions

    def get_train_silos_second_method(self):
        """
        Method to assign data points to silos in a heterogeneous fashion
        """
        # Assign initial partitions (prior silos)
        self.assign_partitions(column=self.silo_attribute, df=self.df_train, n=self.num_silos)
        
        # Retrieve prior assignments and initialize posterior assignments
        prior = np.array(self.df_with_partitions["prior_silos"])
        self.prior = np.copy(prior)
        
        silo_wise_points = []
        
        # Sample points for each silo based on the level of heterogeneity
        for silo_id in range(self.num_silos):
            silo_prior_points = np.argwhere(prior == silo_id)[:, 0]  # Get points for current silo
            sampled_prior = np.random.choice(silo_prior_points, replace=False, 
                                             size=int(int(self.n/self.num_silos) * self.level_of_heterogeneity))  
            prior[sampled_prior] = self.num_silos  # Mark points as sampled

            sampled_prior = list(sampled_prior)
            silo_wise_points.append(sampled_prior)
        
        # Assign remaining points to silos (excluding those already selected)
        for silo_id in range(self.num_silos):
            complement_silo_prior_points = np.argwhere(prior != self.num_silos)[:, 0]  # Points not yet assigned
            try:
                sampled_complement_silo = np.random.choice(complement_silo_prior_points, replace=False, 
                                                           size=int(self.n/self.num_silos) - len(silo_wise_points[silo_id]))
            except:
                sampled_complement_silo = np.copy(complement_silo_prior_points)  # Edge case: not enough points remaining
            prior[sampled_complement_silo] = self.num_silos  # Assign points

            sampled_complement_silo = list(sampled_complement_silo)
            silo_wise_points[silo_id].extend(sampled_complement_silo)
        
        # Finalize posterior silo assignments
        self.silo_wise_points = silo_wise_points
        self.posterior = np.copy(prior)
        for silo_id, silo_points in enumerate(silo_wise_points):
            self.posterior[silo_points] = silo_id

        # Calculate support for each silo
        self.silo_support = torch.tensor(np.bincount(np.int64(self.posterior)), dtype=torch.float)
        self.silo_support_fraction = self.silo_support / self.silo_support.sum()

        # Store the final partitions and normalize the data
        self.df_with_partitions["posterior_silos"] = self.posterior
        self.df_with_partitions = self.df_with_partitions.drop(columns=['prior_silos'])

        # Normalize all the silo data
        for col in self.cols_to_norm:
            self.df_with_partitions[col] = self.df_with_partitions[col].apply(lambda x: (x - self.mean_train[col]) / self.std_train[col])

        # Group by silos and store train data for each silo
        grouped = self.df_with_partitions.groupby('posterior_silos')
        self.silo_train_dfs = {int(posterior_silos): group.reset_index(drop=True) for posterior_silos, group in grouped}

    def getTraindata_P_s(self):
        """Retrieve the full training data as a `TabularDataset`."""
        return TabularDataset(self.df_train_full, self.non_sens_attr, self.sensitive_attributes, output_col_name=self.output_col_name)

    def getTrain(self):
        """
        Fetch the silo-specific training data """
        self.get_train_silos_second_method()
        return [TabularDataset(silo_df, self.non_sens_attr, self.sensitive_attributes, output_col_name=self.output_col_name) 
                for silo_id, silo_df in self.silo_train_dfs.items()]
    
    def getTest(self):
        """Retrieve the test dataset as a `TabularDataset`."""
        return TabularDataset(self.df_test, self.non_sens_attr, self.sensitive_attributes, output_col_name=self.output_col_name)

    def calculateP_s(self, demographic_parity=True):
        """
        Calculate the fairness transformation matrix P_s based on demographic parity.
        """
        if demographic_parity:
            dataset = self.getTraindata_P_s()
            sens = torch.zeros(dataset.count_attr[0])
            for i in range(dataset.__len__()):
                _, u, _, _ = dataset.__getitem__(i)
                sens += u
            sens /= dataset.__len__()
            return torch.diag(1 / (sens)**0.5)
        else:
            dataset = self.getTraindata_P_s()
            diff_matrices = [torch.zeros(dataset.count_attr[0]), torch.zeros(dataset.count_attr[0])]
            lengths = [0, 0]
            for i in range(dataset.__len__()):
                _, u, lab, _ = dataset.__getitem__(i)
                diff_matrices[lab] += u
                lengths[lab] += 1
            diff_matrices[0] /= lengths[0]
            diff_matrices[1] /= lengths[1]
            return [torch.diag(1 / (diff_matrices[0])**0.5), torch.diag(1 / (diff_matrices[1])**0.5)]


class TabularDataset(Data.Dataset):
    """Custom dataset class for handling tabular data."""
    def __init__(self, df, non_sens_attr, sensitive_attributes, output_col_name):
        self.df = df
        self.sensitive_attributes = sensitive_attributes

        # Prepare the data for non-sensitive and sensitive attributes
        self.one_hot_non_senstive = self.df[non_sens_attr]
        self.sensitive_table = self.df[self.sensitive_attributes]
        self.output = self.df[output_col_name]

        # Mapping for sensitive attribute counts and one-hot encoding
        self.count_attr = []
        self.attr_no = {}

        for col_name in self.sensitive_attributes:
            self.attr_no[col_name] = {}
            count = 0
            for col_nam_attr in list(self.df[col_name].unique()):
                self.attr_no[col_name][col_nam_attr] = count
                count += 1
            self.count_attr.append(count)

        for i in range(len(self.count_attr) - 2, -1, -1):
            self.count_attr[i] = self.count_attr[i] * self.count_attr[i + 1]
        self.count_attr.append(1)

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.one_hot_non_senstive.index)

    def __getitem__(self, idx):
        """
        Fetches a sample from the dataset including non-sensitive attributes, sensitive attributes, and the label.
        """
        non_sensitive_attributes = np.array(self.one_hot_non_senstive.iloc[idx])
        sensitive_one_hot, sens_ind = self.onehotlookup(self.sensitive_table.iloc[idx])
        label = self.output.iloc[idx]
                
        return torch.from_numpy(non_sensitive_attributes), sensitive_one_hot, label, sens_ind

    def onehotlookup(self, df):
        """
        Converts sensitive attributes into a one-hot encoded vector and returns the index.
        """
        one_hot_vector = torch.zeros(self.count_attr[0])
        index = 0
        for i, attr in enumerate(self.sensitive_attributes):
            index += self.count_attr[i + 1] * self.attr_no[attr][df[attr]]
        one_hot_vector[index] = 1
        return one_hot_vector, index


class UTKFaceDataset(Data.Dataset):
    """Dataset class for handling UTKFace dataset images."""
    def __init__(self, sensitive_attribute="race", split=0.75, train=True, seed=100):
        # Map sensitive attributes to their index and count
        self.sens_count = 0
        if sensitive_attribute == "gender":
            self.sens_count = 2
            self.sens_index = 1
        if sensitive_attribute == "race":
            self.sens_count = 5
            self.sens_index = 2
        self.age_ranges = [(0,10), (10,15), (15,20), (20,25), (25,30), (30,40), (40,50), (50,60), (60,120)]
        if sensitive_attribute == "age":
            self.sens_count = 9
            self.sens_index = 0
        
        # Image transformations and path setup
        self.transforms = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
        self.path = "./Datasets/UTKFace/UTKFace/"
        self.image_names = os.listdir(self.path)
        if seed:
            self.image_names = random.Random(seed).shuffle(self.image_names)
        
        # Split dataset into train and test
        if train:
            self.dataset = self.image_names[:int(split * len(self.image_names))]
        else:
            self.dataset = self.image_names[int(split * len(self.image_names)):]

    def __getitem__(self, idx):
        """
        Retrieves an image, its corresponding sensitive attribute, and its label.
        """
        image_retr = self.dataset[idx]
        attributes = image_retr.split("_")[:-1]

        sensitive_one_hot = torch.zeros(self.sens_count)
        sens_ind = int(attributes[self.sens_index])
        sensitive_one_hot[sens_ind] = 1

        # Assign age-based labels
        label = 0
        for i, (a, b) in enumerate(self.age_ranges):
            if a < int(attributes[0]) and int(attributes[0]) <= b:
                label = i
                break
        
        image_vec = Image.open(self.path + image_retr)
        u = self.transforms(image_vec)

        return u, sensitive_one_hot, label, sens_ind

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.dataset)
    
    def calculateP_s(self):
        """
        Calculate the fairness transformation matrix P_s for the UTKFace dataset.
        """
        sens = torch.zeros(self.sens_count)
        for i in range(self.__len__()):
            _, u, _, _ = self.__getitem__(i)
            sens += u
        sens /= self.__len__()
        return torch.diag(1 / (sens)**0.5)
