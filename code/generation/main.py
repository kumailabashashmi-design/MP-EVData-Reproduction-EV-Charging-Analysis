"""
Charging Station Power Curve Generation - Main Training Script
Just modify model_name to train different models
"""

import torch
import os
from options import Options
from dataset import create_power_dataloader

def main():
    # ===== Model Configuration =====
    # Change here to select the model to train:
    # "diffusion" - DDPM Diffusion Model
    # "gan" - GAN Model (original TimeGAN changed to regular GAN)
    # "aae" - VAE/AAE Model
    # "lhs" - LHS Sampling (no training required)
    # "gmm" - GMM Model (no training required)

    # "all" - Run all models in sequence
    
    model_name = "all"  # <-- Change model name here
    
    # ===== Basic Settings =====
    # Run mode:
    # "train_only" - Only train all models
    # "generate_only" - Only generate scenarios and analyze (requires trained models)
    # "train_and_generate" - Train and generate scenarios immediately (original mode)
    mode = "generate_only"  # <-- Change run mode here
    
    # Set parameters based on mode
    if mode == "train_only":
        isTrain = True
        generate_scenarios = False
    elif mode == "generate_only":
        isTrain = False
        generate_scenarios = True
    else:  # train_and_generate
        isTrain = True
        generate_scenarios = True
    
    print(f"Running model: {model_name}")
    print(f"Run mode: {mode}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    if model_name == "all":
        # Run all models in sequence
        all_models = ["diffusion", "gan", "aae", "gmm"]
        
        if mode == "train_only":
            # Only train all models
            for single_model in all_models:
                print(f"\n{'='*60}")
                print(f"Start training model: {single_model}")
                print(f"{'='*60}")
                
                opt = Options(single_model, isTrain)
                os.makedirs(f"weights/{single_model}/{opt.level}", exist_ok=True)
                os.makedirs(f"generation/{single_model}", exist_ok=True)
                
                # Only train the model
                run_single_model(single_model, opt, isTrain)
                print(f"\nModel {single_model} training completed!")
        
        elif mode == "generate_only":
            # Only generate scenarios and analyze
            print(f"\n{'='*60}")
            print("Starting to generate scenarios for all models...")
            print(f"{'='*60}")
            
            for single_model in all_models:
                print(f"\nStarting to generate 1000 scenarios for {single_model}...")
                opt = Options(single_model, False)  # Generation mode
                os.makedirs(f"generation/{single_model}", exist_ok=True)
                generate_1000_scenarios(single_model, opt)
                print(f"\nModel {single_model} scenario generation completed!")
            

        
        else:  # train_and_generate
            # Train and immediately generate scenarios (original mode)
            for single_model in all_models:
                print(f"\n{'='*60}")
                print(f"Starting to run model: {single_model}")
                print(f"{'='*60}")
                
                opt = Options(single_model, isTrain)
                os.makedirs(f"weights/{single_model}/{opt.level}", exist_ok=True)
                os.makedirs(f"generation/{single_model}", exist_ok=True)
                
                # Train model
                run_single_model(single_model, opt, isTrain)
                
                # Generate 1000 scenarios
                if generate_scenarios:
                    print(f"\nStarting to generate 1000 scenarios for {single_model}...")
                    generate_1000_scenarios(single_model, opt)
                
                print(f"\nModel {single_model} completed!")
            

    else:
        # Run single model
        opt = Options(model_name, isTrain)
        os.makedirs(f"weights/{model_name}/{opt.level}", exist_ok=True)
        os.makedirs(f"generation/{model_name}", exist_ok=True)
        
        # Train model
        run_single_model(model_name, opt, isTrain)
        
        # Generate 1000 scenarios
        if generate_scenarios:
            print(f"\nStarting to generate 1000 scenarios for {model_name}...")
            generate_1000_scenarios(model_name, opt)
            

    
    print("\nAll tasks completed!")

def run_single_model(model_name, opt, isTrain):
    """Run a single model"""
    if model_name == "diffusion":
        run_ddpm(opt, isTrain)
    elif model_name == "gan":
        run_gan(opt, isTrain)
    elif model_name == "aae":
        run_aae(opt, isTrain)
    elif model_name == "lhs":
        run_lhs()
    elif model_name == "gmm":
        run_gmm()
    else:
        print(f"Unknown model: {model_name}")

def generate_1000_scenarios(model_name, opt):
    """Generate 500 scenarios for the specified model"""
    if model_name in ["lhs", "gmm"]:
        # LHS and GMM already generate samples during training, need to expand to 500
        if model_name == "lhs":
            from lhs import sample_power_lhs
            sample_power_lhs("A1.csv", 500, "generation/LHS/power")
        else:  # gmm
            from gmm import sample_power_gmm
            sample_power_gmm("A1.csv", 500, "generation/gmm/power")
    else:
        # Deep learning models need to load weights for generation
        if model_name == "diffusion":
            generate_ddpm_scenarios(opt)
        elif model_name == "gan":
            generate_gan_scenarios(opt)
        elif model_name == "aae":
            generate_aae_scenarios(opt)




def generate_ddpm_scenarios(opt):
    """Generate 1000 DDPM scenarios"""
    from diffusion import DDPM
    data_loader, dataset = create_power_dataloader("A1.csv", opt.batch_size, opt.shuffle)
    model = DDPM(opt, data_loader)
    
    weight_dir = f"weights/{opt.model_name}/{opt.network}/{opt.level}/{opt.cond_flag}"
    if os.path.exists(weight_dir):
        weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pt')]
        if weight_files:
            latest_epoch = max([int(f.split('epoch')[1].split('.pt')[0]) for f in weight_files])
            weight_path = f"{weight_dir}/epoch{latest_epoch}.pt"
            model.sample(weight_path, 500, condition=None, output_dir="generation/diffusion")
        else:
            print("DDPM weight files not found")
    else:
        print("DDPM weight directory not found")

def generate_gan_scenarios(opt):
    """Generate 1000 GAN scenarios"""
    from timegan import GAN
    data_loader, dataset = create_power_dataloader("A1.csv", opt.batch_size, opt.shuffle)
    model = GAN(opt, data_loader)
    
    weight_dir = f"weights/{opt.model_name}/{opt.level}"
    if os.path.exists(weight_dir):
        weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pt')]
        if weight_files:
            latest_epoch = max([int(f.split('epoch')[1].split('.pt')[0]) for f in weight_files])
            weight_path = f"{weight_dir}/epoch{latest_epoch}.pt"
            model.test(weight_path, 500, output_dir="generation/gan")
        else:
            print("GAN weight files not found")
    else:
        print("GAN weight directory not found")

def generate_aae_scenarios(opt):
    """Generate 1000 AAE scenarios"""
    from vae import AAE
    data_loader, dataset = create_power_dataloader("A1.csv", opt.batch_size, opt.shuffle)
    model = AAE(opt, data_loader)
    
    weight_dir = f"weights/{opt.model_name}/{opt.level}"
    if os.path.exists(weight_dir):
        weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pt')]
        if weight_files:
            latest_epoch = max([int(f.split('epoch')[1].split('.pt')[0]) for f in weight_files])
            weight_path = f"{weight_dir}/epoch{latest_epoch}.pt"
            model.test(weight_path, 500, output_dir="generation/aae")
        else:
            print("AAE weight files not found")
    else:
        print("AAE weight directory not found")
def run_ddpm(opt, isTrain):
    """Run DDPM model"""
    from diffusion import DDPM
    
    data_loader, dataset = create_power_dataloader("A1.csv", opt.batch_size, opt.shuffle)
    model = DDPM(opt, data_loader)
    
    if isTrain:
        print("Starting DDPM training...")
        model.train()
        print("DDPM training completed")
    else:
        print("Starting DDPM sample generation...")
        # Find latest weights
        weight_dir = f"weights/{opt.model_name}/{opt.network}/{opt.level}/{opt.cond_flag}"
        if os.path.exists(weight_dir):
            weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pt')]
            if weight_files:
                latest_epoch = max([int(f.split('epoch')[1].split('.pt')[0]) for f in weight_files])
                weight_path = f"{weight_dir}/epoch{latest_epoch}.pt"
                model.sample(weight_path, 100, condition=None)
                print("DDPM sample generation completed")
            else:
                print("DDPM weight files not found")
        else:
            print("DDPM weight directory not found")

def run_gan(opt, isTrain):
    """Run GAN model"""
    from timegan import GAN
    
    data_loader, dataset = create_power_dataloader("A1.csv", opt.batch_size, opt.shuffle)
    model = GAN(opt, data_loader)
    
    if isTrain:
        print("Starting GAN training...")
        model.train()
        print("GAN training completed")
    else:
        print("Starting GAN sample generation...")
        weight_dir = f"weights/{opt.model_name}/{opt.level}"
        if os.path.exists(weight_dir):
            weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pt')]
            if weight_files:
                latest_epoch = max([int(f.split('epoch')[1].split('.pt')[0]) for f in weight_files])
                weight_path = f"{weight_dir}/epoch{latest_epoch}.pt"
                model.test(weight_path, 100)
                print("GAN sample generation completed")
            else:
                print("GAN weight files not found")
        else:
            print("GAN weight directory not found")

def run_aae(opt, isTrain):
    """Run AAE model"""
    from vae import AAE
    
    data_loader, dataset = create_power_dataloader("A1.csv", opt.batch_size, opt.shuffle)
    model = AAE(opt, data_loader)
    
    if isTrain:
        print("Starting AAE training...")
        model.train()
        print("AAE training completed")
    else:
        print("Starting AAE sample generation...")
        weight_dir = f"weights/{opt.model_name}/{opt.level}"
        if os.path.exists(weight_dir):
            weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pt')]
            if weight_files:
                latest_epoch = max([int(f.split('epoch')[1].split('.pt')[0]) for f in weight_files])
                weight_path = f"{weight_dir}/epoch{latest_epoch}.pt"
                model.test(weight_path, 100)
                print("AAE sample generation completed")
            else:
                print("AAE weight files not found")
        else:
            print("AAE weight directory not found")

def run_lhs():
    """Run LHS sampling"""
    from lhs import sample_power_lhs
    
    print("Starting LHS sampling...")
    sample_power_lhs("A1.csv", 100, "generation/LHS/power")
    print("LHS sampling completed")

def run_gmm():
    """Run GMM model"""
    from gmm import sample_power_gmm
    
    print("Starting GMM sampling...")
    sample_power_gmm("A1.csv", 100, "generation/gmm/power")
    print("GMM sampling completed")



if __name__ == "__main__":
    main()