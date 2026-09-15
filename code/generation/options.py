import torch

class Options:
    def __init__(self, model_name, isTrain):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_epochs = 300
        self.level = "power"  # "power"
        self.seq_len = 96  # power: 96 (24 hours data, 15min intervals)
        self.cond_flag = "unconditional"  # "unconditional" (non-conditional generation) 
        if isTrain:
            self.batch_size = 4  # station: 4, driver: 8
            self.shuffle = True
        else:
            self.batch_size = 1
            self.shuffle = False
        if model_name == "diffusion":
            self.init_lr = 1e-3
            self.network = "attention"  # "attention" or "cnn"
            self.input_dim = 1
            self.hidden_dim = 48
            self.cond_dim = 2
            self.nhead = 4
            self.beta_start = 1e-4
            self.beta_end = 0.5
            self.n_steps = 140
            self.schedule = "linear"  # "linear" / quadratic
        elif model_name == "timegan":
            self.input_size = 1
            self.latent_size = 64
            self.hidden_G = 128
            self.hidden_D = 128
            self.condition_size = 3
            self.lr_G = 1e-4
            self.lr_D = 1e-4
        elif model_name == "gan":
            self.input_size = 1
            self.latent_size = 64
            self.hidden_G = 128
            self.hidden_D = 128
            self.lr_G = 1e-4
            self.lr_D = 1e-4

        elif model_name == "aae":
            self.latent_size = 64
            self.lr_AE = 1e-4
            self.lr_D = 1e-4
            self.alpha = 0.5
            self.beta = 0.1
