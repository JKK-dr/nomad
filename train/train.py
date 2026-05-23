import os
import argparse
import numpy as np
import yaml
import time
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.models.future_prediction import FuturePredictionHead
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_nomad,
    load_model,
)
from vint_train.wandb_utils import wandb

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))


def _normalize_future_prediction_config(config):
    future_config = {
        "enabled": False,
        "horizon": 1,
        "spacing": None,
        "encoding_size": None,
        "loss_weight": 0.1,
        "predict_deltas": True,
        "detach_anchor": True,
    }
    future_config.update(config.get("future_prediction") or {})
    future_config["enabled"] = bool(future_config["enabled"])
    future_config["horizon"] = int(future_config["horizon"])
    if future_config["spacing"] is not None:
        future_config["spacing"] = int(future_config["spacing"])
    if future_config["encoding_size"] is not None:
        future_config["encoding_size"] = int(future_config["encoding_size"])
    future_config["loss_weight"] = float(future_config["loss_weight"])
    future_config["predict_deltas"] = bool(future_config["predict_deltas"])
    future_config["detach_anchor"] = bool(future_config["detach_anchor"])
    if future_config["enabled"]:
        assert future_config["horizon"] > 0
        assert future_config["loss_weight"] >= 0
        if future_config["spacing"] is not None:
            assert future_config["spacing"] > 0
        if future_config["encoding_size"] is not None:
            assert future_config["encoding_size"] > 0
    config["future_prediction"] = future_config
    return future_config


def _expand_path(path):
    if path is None:
        return None
    path = os.path.expandvars(os.path.expanduser(str(path)))
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def _processed_trajectory_names(data_folder):
    data_folder = _expand_path(data_folder)
    if not os.path.isdir(data_folder):
        return []
    return sorted(
        entry
        for entry in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, entry))
        and os.path.isfile(os.path.join(data_folder, entry, "traj_data.pkl"))
    )


def _has_recon_hdf5_files(data_folder):
    data_folder = _expand_path(data_folder)
    candidate_folders = [data_folder, os.path.join(data_folder, "recon_release")]
    for candidate_folder in candidate_folders:
        if not os.path.isdir(candidate_folder):
            continue
        if any(
            os.path.isfile(os.path.join(candidate_folder, entry))
            and entry.lower().endswith((".h5", ".hdf5"))
            for entry in os.listdir(candidate_folder)
        ):
            return True
    return False


def _maybe_process_recon_dataset(dataset_name, data_config):
    if _processed_trajectory_names(data_config["data_folder"]):
        return
    if dataset_name != "recon" or not data_config.get("auto_process", False):
        return

    raw_data_folder = _expand_path(data_config.get("raw_data_folder", data_config["data_folder"]))
    if not _has_recon_hdf5_files(raw_data_folder):
        return

    processed_data_folder = _expand_path(
        data_config.get("processed_data_folder")
        or os.path.join(os.path.dirname(raw_data_folder), "recon_processed")
    )
    if not _processed_trajectory_names(processed_data_folder):
        print(
            f"Processing raw RECON HDF5 data from {raw_data_folder} "
            f"to {processed_data_folder}"
        )
        from process_recon import process_recon_directory

        process_recon_directory(
            raw_data_folder,
            processed_data_folder,
            num_trajs=int(data_config.get("num_trajs", -1)),
        )

    data_config["raw_data_folder"] = raw_data_folder
    data_config["data_folder"] = processed_data_folder


def _ensure_split_files(dataset_name, data_config, seed):
    train_split_folder = _expand_path(data_config.get("train"))
    test_split_folder = _expand_path(data_config.get("test"))
    if train_split_folder is None or test_split_folder is None:
        return

    data_config["train"] = train_split_folder
    data_config["test"] = test_split_folder
    train_names_path = os.path.join(train_split_folder, "traj_names.txt")
    test_names_path = os.path.join(test_split_folder, "traj_names.txt")
    if os.path.isfile(train_names_path) and os.path.isfile(test_names_path):
        return

    if not data_config.get("auto_split", False):
        missing = [
            path
            for path in [train_names_path, test_names_path]
            if not os.path.isfile(path)
        ]
        raise FileNotFoundError(
            "Missing data split file(s): " + ", ".join(missing)
        )

    traj_names = _processed_trajectory_names(data_config["data_folder"])
    if len(traj_names) == 0:
        raise FileNotFoundError(
            f"No processed trajectories found in {data_config['data_folder']}. "
            "Expected trajectory folders containing traj_data.pkl."
        )

    split_ratio = float(data_config.get("split", 0.8))
    split_seed = int(data_config.get("split_seed", seed))
    rng = np.random.default_rng(split_seed)
    traj_names = list(traj_names)
    rng.shuffle(traj_names)

    if len(traj_names) == 1:
        train_names = traj_names
        test_names = traj_names
    else:
        split_index = int(split_ratio * len(traj_names))
        split_index = min(max(split_index, 1), len(traj_names) - 1)
        train_names = traj_names[:split_index]
        test_names = traj_names[split_index:]

    os.makedirs(train_split_folder, exist_ok=True)
    os.makedirs(test_split_folder, exist_ok=True)
    with open(train_names_path, "w") as f:
        f.write("\n".join(train_names) + "\n")
    with open(test_names_path, "w") as f:
        f.write("\n".join(test_names) + "\n")
    print(
        f"Created {dataset_name} split: "
        f"{len(train_names)} train / {len(test_names)} test trajectories"
    )


def _prepare_dataset_configs(config):
    seed = int(config.get("seed", 0))
    for dataset_name, data_config in config["datasets"].items():
        for path_key in ["data_folder", "raw_data_folder", "processed_data_folder"]:
            if path_key in data_config:
                data_config[path_key] = _expand_path(data_config[path_key])
        _maybe_process_recon_dataset(dataset_name, data_config)
        _ensure_split_files(dataset_name, data_config, seed)


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]
    future_prediction_config = _normalize_future_prediction_config(config)
    _prepare_dataset_configs(config)

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                        future_prediction=future_prediction_config,
                        max_trajs=data_config.get(f"max_{data_split_type}_trajs"),
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}
                        test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=config["num_workers"] > 0,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
            future_prediction=future_prediction_config,
        )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            future_prediction=future_prediction_config,
        )
    elif config["model_type"] == "nomad":
        if future_prediction_config["enabled"] and config["vision_encoder"] != "nomad_vint":
            raise NotImplementedError(
                "future_prediction is currently implemented for NoMaD with vision_encoder=nomad_vint"
            )
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
                future_prediction=future_prediction_config,
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vib": 
            vision_encoder = ViB(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
            
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        future_pred_network = None
        if future_prediction_config["enabled"]:
            future_pred_network = FuturePredictionHead(
                input_size=config["encoding_size"],
                horizon=future_prediction_config["horizon"],
                encoding_size=(
                    future_prediction_config["encoding_size"]
                    or config["encoding_size"]
                ),
                predict_deltas=future_prediction_config["predict_deltas"],
                detach_anchor=future_prediction_config["detach_anchor"],
            )
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
            future_pred_net=future_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    else:
        raise ValueError(f"Model {config['model']} not supported")

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, config["model_type"], latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    if config["model_type"] == "vint" or config["model_type"] == "gnm": 
        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            future_loss_weight=future_prediction_config["loss_weight"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
    else:
        train_eval_loop_nomad(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            future_loss_weight=future_prediction_config["loss_weight"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )

    print("FINISHED TRAINING")


if __name__ == "__main__":
    os.chdir(TRAIN_DIR)
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/nomad.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()
    config_path = _expand_path(args.config)

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)
    config["use_wandb"] = bool(config.get("use_wandb", True))

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            entity="gnmv2", # TODO: change this to your wandb entity
        )
        wandb.save(config_path, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)
    else:
        os.environ.setdefault("WANDB_MODE", "disabled")

    print(config)
    main(config)
