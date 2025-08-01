def print_config(config):
    attrs = {}
    attrs.update(config.__dict__)

    for key, value in config.__class__.__dict__.items():
        if not key.startswith('__') and key not in attrs:
            attrs[key] = value
            
    for key, value in attrs.items():
        print(f"{key}: {value}")
        
        
class Config:
    # Model
    model_name = "cv2.tiny"
    input_dim = 6
    target_dim = 1

    # Device & reproducibility
    device = "cuda"
    seed = 10
    benchmark = False
    deterministic = True
    deterministic_algo = False
    torch_compile = True

    # Optimization
    optimizer_name = "AdamW"
    lr = 4e-4
    weight_decay = 1e-4

    # Gradient accumulation
    grad_accum = False
    grad_accum_steps = 1
    batch_size = 32  # // grad_accum_steps
    
    #EMA
    ema_decay = 0.999
    warmup_gamma = 1.0
    warmup_power = 0.75

    # Training schedule
    num_epochs = 100

    # Scheduler
    scheduler_name = "default"
    scheduler_on_step = True
    scheduler_on_epoch = False
    log_lr_on_step = True
    log_lr_on_ep = False

    # Regularization
    dropout = 0.1
    drop_path_rate = 0.0
    label_smoothing = 0.1

    # Metrics & losses
    loss_names = ["mse"] # forgot why is this for, maybe cross checking 
    metric_names = ["rmse"] # if metric is reverse put it in loss_names

    # Evaluation & logging
    evaluation = True
    save = True
    save_ema = True
    save_every = True
    save_last = False
    saving_stages = ["eval"]

    # Checkpointing
    metric_names_for_saving = ["mse", "rmse"]
    ema_metrics = ["mse", "rmse"]
    save_last_n = None
    save_top_n = None
    ckpt_paths = []
    top_n_ckpt_paths = []
    resume = False
    with_id = "SPEED-313"

    # Experiment tracking
    neptune_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOGE2YjNiZS1mZGUyLTRjYjItYTg5Yy1mZWJkZTIzNzE1NmIifQ=="

config = Config()