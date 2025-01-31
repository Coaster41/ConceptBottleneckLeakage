import torch

def leakage_loss(c_hat, c_tilde):
    batch_size = c_hat.shape[0]
    c_hat_mean = torch.mean(c_hat, axis=0) # (n_concepts, 1)
    c_tilde_mean = torch.mean(c_tilde, axis=0) # (n_latent, 1)
    c_tilde_deviation = c_tilde-c_tilde_mean # (batch_size, n_latent, 1)
    c_tilde_deviation_T = torch.moveaxis(c_tilde_deviation,-1,-2) # (batch_size, 1, n_latent)
    sigma_tilde = 1/batch_size * torch.sum(torch.matmul(c_tilde_deviation, c_tilde_deviation_T),axis=0)  # (n_latent, n_latent)
    log_det_sigma_tilde = torch.log(torch.linalg.det(sigma_tilde)) # scalar

    c_stack_T = torch.concatenate((c_hat,c_tilde), axis=1) # (batch_size, n_concepts+n_latent, 1)
    c_hat_mean_resize = torch.repeat(c_hat_mean.reshape(1,c_hat.shape[1],1), batch_size, axis=0) # (batch_size, n_concepts, 1)
    c_tilde_mean_resize = torch.repeat(c_tilde_mean.reshape(1,c_tilde.shape[1],1), batch_size, axis=0) # (batch_size, n_latent, 1)
    c_stack_mean_T = torch.concatenate((c_hat_mean_resize,c_tilde_mean_resize), axis=1)  # (batch_size, n_concepts+n_latent, 1)
    c_stack_deviation = c_stack_T-c_stack_mean_T  # (batch_size, n_concepts+n_latent, 1)
    c_stack_deviation_T = torch.moveaxis(c_stack_deviation,-1,-2)  # (batch_size, 1, n_concepts+n_latent)
    sigma = 1/batch_size * torch.sum(torch.matmul(c_stack_deviation,c_stack_deviation_T),axis=0)  # (batch_size, n_concepts+n_latent, n_concepts+n_latent)
    log_det_sigma = torch.log(torch.linalg.det(sigma)) # scalar

    return (log_det_sigma - log_det_sigma_tilde) / 2

def leakage_loss_simple(c_hat, c_tilde):
    c_hat = c_hat
    c_tilde = c_tilde
    c = torch.concatenate((c_hat, c_tilde), axis=1).squeeze() # 64, 256+128
    sigma = torch.cov(c.T)
    # print(sigma)
    # print(torch.logdet(sigma[-c_tilde.shape[1]:, -c_tilde.shape[1]:]), torch.logdet(sigma))
    return 0.5 * (torch.logdet(sigma[-c_tilde.shape[1]:, -c_tilde.shape[1]:]) - torch.logdet(sigma))

