import torch
import copy

def compute_grad_per_silo(args, device, non_sensitive, sensitive, label, P_s_negative_half, classification_loss_fn, model, W, W_):
    # Initialize total loss and gradient dictionaries for the model and W
    total_loss = 0
    model_grad = {}  # Stores gradients for the model parameters
    W_grad = {}      # Stores gradients for W (and W_ if applicable)

    with torch.no_grad():
        # Zero out the actual gradients for W and W_ (if demographic parity is not used)
        if W.grad is not None:
            W.grad.zero_()
        if not args.demographic_parity and W_.grad is not None:
            W_.grad.zero_()

        # Initialize W gradient dictionary with zero tensors
        W_grad["W"] = torch.zeros(W.shape).to(device)
        if not args.demographic_parity:
            W_grad["W_"] = torch.zeros(W_.shape).to(device)

        # Initialize gradient dictionary for the model's parameters
        for name, param in model.named_parameters():
            model_grad[name] = torch.zeros(param.shape).to(device)

    # Zero out actual gradients for the model's parameters
    model.zero_grad()

    # Move non-sensitive, sensitive attributes, and labels to the appropriate device (GPU/CPU)
    non_sensitive = non_sensitive.to(device)
    sensitive = sensitive.to(device)
    label = label.to(device)

    '''Non-sensitive part of the silo'''
    # Forward pass through the model with non-sensitive data
    y_logit, y_hat = model(non_sensitive.float())

    # Compute classification loss based on the model type (logistic-regression or other)
    if args.model_type == "logistic-regression":
        classification_loss = classification_loss_fn(y_logit, label.unsqueeze(1).float())
    else:
        classification_loss = classification_loss_fn(y_logit, label)
    
    # Mean classification loss and backpropagation
    classification_loss = torch.mean(classification_loss)
    classification_loss.backward()

    # Store gradients for non-sensitive data
    with torch.no_grad():
        total_loss += classification_loss.item()
        batch_theta_non_sensitive_gradient = {name: param.grad.clone().to(device) for name, param in model.named_parameters()}

    # Zero out the gradients for the model after processing non-sensitive data
    model.zero_grad()

    '''Sensitive part of the silo'''
    # Forward pass through the model with non-sensitive data again for consistency
    y_logit, y_hat = model(non_sensitive.float())

    # Compute Fermi loss based on demographic parity or equalized odds
    if args.demographic_parity:
        # Compute loss for demographic parity using einsum operations
        p_hat_yhat = torch.diag_embed(y_hat)  # b x m x m
        p_hat_yhat_s = torch.einsum("bm, bk -> bmk", [y_hat, sensitive])
        fermi_loss = (-1 * torch.einsum("km, bmn, nj -> bkj", [W, p_hat_yhat, W.T]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) + 
                      2 * torch.einsum("km, bmn, nl -> bkl", [W, p_hat_yhat_s, P_s_negative_half]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1)
    else:
        # Separate sensitive data and predictions based on label values (0 or 1)
        y_hat_given_1 = []
        sensitive_given_1 = []
        y_hat_given_0 = []
        sensitive_given_0 = []
        
        for i in range(label.size(0)):
            if label[i] == 1:
                y_hat_given_1.append(y_hat[i].unsqueeze(0))
                sensitive_given_1.append(sensitive[i].unsqueeze(0))
            else:
                y_hat_given_0.append(y_hat[i].unsqueeze(0))
                sensitive_given_0.append(sensitive[i].unsqueeze(0))

        # Concatenate the data from the separated lists
        y_hat_given_1 = torch.cat(y_hat_given_1, axis=0)
        y_hat_given_0 = torch.cat(y_hat_given_0, axis=0)
        sensitive_given_1 = torch.cat(sensitive_given_1, axis=0)
        sensitive_given_0 = torch.cat(sensitive_given_0, axis=0)
        
        # Compute the conditional parts of the Fermi loss for each label group
        p_hat_yhat_part_1 = torch.diag_embed(y_hat_given_1)
        p_hat_yhat_part_0 = torch.diag_embed(y_hat_given_0)
        p_hat_yhat_s_given_1 = torch.einsum("bm, bk -> bmk", [y_hat_given_1, sensitive_given_1])
        p_hat_yhat_s_given_0 = torch.einsum("bm, bk -> bmk", [y_hat_given_0, sensitive_given_0])
        
        # Fermi loss for label 1
        fermi_loss_1 = (-1 * torch.einsum("km, bmn, nj -> bkj", [W, p_hat_yhat_part_1, W.T]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) + 
                         2 * torch.einsum("km, bmn, nl -> bkl", [W, p_hat_yhat_s_given_1, P_s_negative_half[1]]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1)
        
        # Fermi loss for label 0
        fermi_loss_0 = (-1 * torch.einsum("km, bmn, nj -> bkj", [W_, p_hat_yhat_part_0, W_.T]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) + 
                         2 * torch.einsum("km, bmn, nl -> bkl", [W_, p_hat_yhat_s_given_0, P_s_negative_half[0]]).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1)

        # Combine the Fermi loss for both label groups
        fermi_loss = torch.zeros_like(label, dtype=torch.float)
        fermi_loss[label == 1] = fermi_loss_1
        fermi_loss[label == 0] = fermi_loss_0

    # Multiply Fermi loss by lambda
    fermi_loss = args.lambd * fermi_loss
    fermi_loss = torch.mean(fermi_loss)
    fermi_loss.backward()

    # Compute gradients for sensitive data
    with torch.no_grad():
        total_loss += fermi_loss.item()
        batch_size = non_sensitive.shape[0]
        
        '''Clipping and noisy sensitive theta gradient'''
        # Calculate the norm of gradients for each sample
        grad_norms = torch.zeros(batch_size, device=device)
        for name, param in model.named_parameters():
            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                grad_norms += torch.norm(param.grad_sample.view(batch_size, -1), dim=1) ** 2
        grad_norms = grad_norms ** 0.5
        
        # Compute clipping factor for each sample based on the Lipschitz constant
        divide_by = grad_norms / args.lipschitz_theta
        divide_by = torch.where(divide_by > 1, divide_by, torch.ones_like(divide_by)).to(device)

        # Compute clipped gradients for theta
        batch_theta_sensitive_gradient = {}
        for name, param in model.named_parameters():
            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                divide_by_unsqueezed_new_shape = (batch_size,) + (1,) * (len(param.grad_sample.shape) - 1)
                divide_by = divide_by.view(divide_by_unsqueezed_new_shape)
                batch_theta_sensitive_gradient[name] = torch.sum(param.grad_sample / divide_by, dim=0)
        
        # Add noise to theta gradients if necessary
        for name, param in model.named_parameters():
            if args.std_theta != 0:
                u_t = torch.normal(mean=0, std=args.std_theta, size=param.shape).to(device)
            else:
                u_t = torch.zeros_like(param).to(device)
            batch_theta_sensitive_gradient[name] = batch_theta_sensitive_gradient[name] + u_t

        '''Noisy sensitive W gradient'''
        batch_W_sensitive_gradient = {}
        if args.std_W != 0:
            v_t = torch.normal(mean=0, std=args.std_W, size=W.shape).to(device)
        else:
            v_t = torch.zeros_like(W).to(device)
        batch_W_sensitive_gradient["W"] = W.grad + v_t

        # Add noise to W_ if demographic parity is not used
        if not args.demographic_parity:
            if args.std_W != 0:
                v_t = torch.normal(mean=0, std=args.std_W, size=W_.shape).to(device)
            else:
                v_t = torch.zeros_like(W_).to(device)
            batch_W_sensitive_gradient["W_"] = W_.grad + v_t

        '''Final gradients'''
        # Combine non-sensitive and sensitive gradients for the model
        for name, param in model.named_parameters():
            model_grad[name] = batch_theta_non_sensitive_gradient[name] + batch_theta_sensitive_gradient[name]

        # Store the gradients for W and W_
        W_grad["W"] = batch_W_sensitive_gradient["W"]
        if not args.demographic_parity:
            W_grad["W_"] = batch_W_sensitive_gradient["W_"]

    # Zero out the actual gradients for the model and W to avoid accumulation
    model.zero_grad()

    with torch.no_grad():
        # Zero out the actual gradients for W and W_ again to ensure no accumulation
        if W.grad is not None:
            W.grad.zero_()
        if not args.demographic_parity and W_.grad is not None:
            W_.grad.zero_()

    # Return the computed gradients and losses for the current silo
    return model_grad, W_grad, fermi_loss.item(), classification_loss.item()
