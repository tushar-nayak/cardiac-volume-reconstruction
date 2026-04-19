#!/usr/bin/env python3
"""
Mixed mode sweep agent with exhaustive wandb logging.

This script:
1. Trains global INR on train split (logs per-step train losses, per-epoch val metrics).
2. Evaluates global INR on test split (logs test_global metrics per-scan and aggregated).
3. Refines each test scan from global prior (logs per-step inner losses, per-scan refinement metrics).
4. Logs hyperparameter config, timing, dataset sizes, and all intermediate results.

Run with:
    wandb agent your_entity/your_project/sweep_id
"""

import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

# === Import your main module (adjust filename as needed) ===
import FINAL_2_gpu_optmized as core

BASE_CONFIG = core.CONFIG


def train_global_inr_with_logging(
    train_scans: List,
    val_scans: List,
    config: Dict,
) -> nn.Module:
    """
    Train shared INR on train split with exhaustive wandb logging.

    Per-step logs (every step):
      - global/train/step, epoch, lr
      - global/train/loss_total
      - global/train/loss_projection
      - global/train/loss_vol_bce
      - global/train/loss_smooth
      - global/train/loss_entropy
      - global/train/loss_area

    Per-epoch logs (after each epoch on full val set):
      - global/val/epoch
      - global/val/dice_2d_{mean, std, min, max}
      - global/val/iou_2d_{mean, std, min, max}
      - global/val/dice_3d_{mean, std, min, max}
      - global/val/iou_3d_{mean, std, min, max}
      - global/val/dice_3d_central_{mean, std, min, max}
      - global/val/iou_3d_central_{mean, std, min, max}
    """
    device = torch.device(config["device"])
    model = core.ImplicitNeuralRepresentation(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_inr_layers"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    total_steps = config["num_epochs"] * config["steps_per_epoch"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config["learning_rate"] / 10,
    )

    vol_weight = float(config.get("vol_supervision_weight", 0.0))
    vol_samples = int(config.get("vol_supervision_samples", 0))
    reg_every = int(config.get("reg_every", 1))
    proj_batch_size = int(config.get("proj_batch_size", 65536))
    grid_res = int(config.get("grid_resolution", 64))

    global_step = 0
    training_start = time.time()

    print(f"\n{'='*70}")
    print(f"TRAINING GLOBAL INR")
    print(f"{'='*70}")
    print(f"Epochs: {config['num_epochs']}, Steps/epoch: {config['steps_per_epoch']}")
    print(f"Total steps: {total_steps}")
    print(f"Train scans: {len(train_scans)}, Val scans: {len(val_scans)}")
    print(f"Hidden dim: {config['hidden_dim']}, Layers: {config['num_inr_layers']}")
    print(f"Vol supervision weight: {vol_weight}")

    for epoch in range(config["num_epochs"]):
        epoch_start = time.time()
        model.train()

        for step_in_epoch in range(config["steps_per_epoch"]):
            # Sample random scan
            scan = random.choice(train_scans)

            # Prepare data
            seg_device = scan.seg_c.to(device)
            D_seg, H_seg, W_seg = seg_device.shape
            contours_device = scan.contours_2d.to(device)

            # Identity pose (no learning)
            pose_layer = core.PoseParameters(config["num_views"])
            with torch.no_grad():
                pose_layer.pose.zero_()
            pose_layer = pose_layer.to(device)

            extrinsics = pose_layer.get_matrices(device=device).detach()

            # Project
            preds = core.project_slices_from_inr_batch(
                model,
                extrinsics,
                scan.chosen_z,
                D_seg,
                config["proj_resolution"],
                device,
                batch_size=proj_batch_size,
            )

            loss_projection = core.contour_bce_loss(preds, contours_device)

            # Volumetric losses
            vol_loss = torch.tensor(0.0, device=device)
            loss_smooth = torch.tensor(0.0, device=device)
            loss_entropy = torch.tensor(0.0, device=device)
            loss_area = torch.tensor(0.0, device=device)

            do_reg = (reg_every <= 1) or (global_step % reg_every == 0)
            if do_reg:
                # Vol BCE
                if vol_weight > 0.0 and vol_samples > 0:
                    coords3d = torch.rand(vol_samples, 3, device=device) * 2.0 - 1.0
                    pred_vol = model(coords3d).squeeze(-1)

                    x_norm = (coords3d[:, 0] + 1.0) * 0.5
                    y_norm = (coords3d[:, 1] + 1.0) * 0.5
                    z_norm = (coords3d[:, 2] + 1.0) * 0.5

                    x_idx = torch.clamp((x_norm * (H_seg - 1)).long(), 0, H_seg - 1)
                    y_idx = torch.clamp((y_norm * (W_seg - 1)).long(), 0, W_seg - 1)
                    z_idx = torch.clamp((z_norm * (D_seg - 1)).long(), 0, D_seg - 1)

                    gt_vol = seg_device[z_idx, x_idx, y_idx].view(-1)
                    vol_loss = F.binary_cross_entropy(pred_vol, gt_vol)

                # Regularizers
                occ_grid = model.sample_grid(
                    resolution=grid_res,
                    device=device,
                    requires_grad=True,
                )
                loss_smooth = core.laplacian_smoothness_loss(occ_grid, weight=0.02)
                loss_entropy = core.volume_entropy_loss(occ_grid, weight=0.01)
                loss_area = core.surface_area_loss(occ_grid, weight=0.005)

            total_loss = (
                loss_projection
                + loss_smooth
                + loss_entropy
                + loss_area
                + vol_weight * vol_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Log step
            current_lr = optimizer.param_groups[0]["lr"]
            log_dict = {
                "global/train/step": global_step,
                "global/train/epoch": epoch,
                "global/train/lr": current_lr,
                "global/train/loss_total": total_loss.item(),
                "global/train/loss_projection": loss_projection.item(),
                "global/train/loss_vol_bce": vol_loss.item(),
                "global/train/loss_smooth": loss_smooth.item(),
                "global/train/loss_entropy": loss_entropy.item(),
                "global/train/loss_area": loss_area.item(),
            }
            wandb.log(log_dict)

            if global_step % config.get("print_every", 50) == 0:
                print(
                    f"[Epoch {epoch}/{config['num_epochs']}] "
                    f"Step {global_step}/{total_steps}: "
                    f"loss={total_loss.item():.6f}, "
                    f"proj={loss_projection.item():.6f}, "
                    f"vol={vol_loss.item():.6f}, "
                    f"smooth={loss_smooth.item():.6f}"
                )

            global_step += 1

        # === Validation at end of epoch ===
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} took {epoch_time:.1f}s, running validation...")

        val_metrics = evaluate_split_with_logging(
            model, val_scans, config, split_name="global/val", device=device
        )

        val_log = {f"global/val/epoch": epoch}
        val_log.update(val_metrics)
        wandb.log(val_log)

        print(f"Val Dice_2D: {val_metrics.get('global/val/dice_2d_mean', 0):.4f} ± "
              f"{val_metrics.get('global/val/dice_2d_std', 0):.4f}")

    training_time = time.time() - training_start
    wandb.summary["global/training_time_seconds"] = training_time
    print(f"\nGlobal training completed in {training_time:.1f}s")

    return model


def evaluate_split_with_logging(
    model: nn.Module,
    scans: List,
    config: Dict,
    split_name: str = "test_global",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate model on a split, log per-scan and aggregate metrics.

    Logs:
      - {split_name}/scan/dice_2d, iou_2d, dice_3d, iou_3d, dice_3d_central, iou_3d_central (per scan)
      - {split_name}/dice_2d_{mean, std, min, max}
      - {split_name}/iou_2d_{mean, std, min, max}
      - {split_name}/dice_3d_{mean, std, min, max}
      - {split_name}/iou_3d_{mean, std, min, max}
      - {split_name}/dice_3d_central_{mean, std, min, max}
      - {split_name}/iou_3d_central_{mean, std, min, max}
    """
    if device is None:
        device = torch.device(config["device"])

    model = model.to(device)
    model.eval()

    dice_2d_list, iou_2d_list = [], []
    dice_3d_list, iou_3d_list = [], []
    dice_3d_c_list, iou_3d_c_list = [], []

    with torch.no_grad():
        for scan_idx, scan in enumerate(scans):
            print(f"  [{scan_idx+1}/{len(scans)}] Evaluating {scan.scan_id}...", end=" ")

            # Zero pose
            pose_layer = core.PoseParameters(config["num_views"])
            with torch.no_grad():
                pose_layer.pose.zero_()
            pose_layer = pose_layer.to(device)

            # 2D metrics
            metrics_2d = core.evaluate_subject_2d(
                model,
                scan.contours_2d,
                pose_layer,
                scan.chosen_z,
                scan.D,
                config,
            )

            # 3D metrics
            metrics_3d = core.evaluate_subject_3d_and_mesh(
                model,
                scan.seg_c,
                scan.scan_id + f"_{split_name.replace('/', '_')}",
                config,
                scan.chosen_z,
            )

            # Log per-scan
            scan_log = {
                f"{split_name}/scan_id": scan.scan_id,
                f"{split_name}/scan/dice_2d": metrics_2d["dice_2d"],
                f"{split_name}/scan/iou_2d": metrics_2d["iou_2d"],
                f"{split_name}/scan/dice_3d": metrics_3d["dice_3d"],
                f"{split_name}/scan/iou_3d": metrics_3d["iou_3d"],
                f"{split_name}/scan/dice_3d_central": metrics_3d["dice_3d_central"],
                f"{split_name}/scan/iou_3d_central": metrics_3d["iou_3d_central"],
            }
            wandb.log(scan_log)

            # Append
            dice_2d_list.append(metrics_2d["dice_2d"])
            iou_2d_list.append(metrics_2d["iou_2d"])
            dice_3d_list.append(metrics_3d["dice_3d"])
            iou_3d_list.append(metrics_3d["iou_3d"])
            dice_3d_c_list.append(metrics_3d["dice_3d_central"])
            iou_3d_c_list.append(metrics_3d["iou_3d_central"])

            print(
                f"2D Dice={metrics_2d['dice_2d']:.4f}, "
                f"3D Dice={metrics_3d['dice_3d']:.4f}"
            )

    # Aggregate
    def agg_stats(lst):
        return {
            "mean": float(np.mean(lst)),
            "std": float(np.std(lst)),
            "min": float(np.min(lst)),
            "max": float(np.max(lst)),
        }

    summary = {}
    for key, lst in [
        ("dice_2d", dice_2d_list),
        ("iou_2d", iou_2d_list),
        ("dice_3d", dice_3d_list),
        ("iou_3d", iou_3d_list),
        ("dice_3d_central", dice_3d_c_list),
        ("iou_3d_central", iou_3d_c_list),
    ]:
        stats = agg_stats(lst)
        for stat_name, stat_val in stats.items():
            summary[f"{split_name}/{key}_{stat_name}"] = stat_val

    return summary


def refine_test_scans_with_logging(
    model_global: nn.Module,
    test_scans: List,
    config: Dict,
) -> Dict[str, float]:
    """
    Refine each test scan from global prior, log all inner losses and per-scan metrics.

    Per-step logs (for each refinement step):
      - mixed/inner/step
      - mixed/inner/scan_id
      - mixed/inner/loss_total
      - mixed/inner/loss_projection
      - mixed/inner/loss_vol_bce
      - mixed/inner/loss_smooth
      - mixed/inner/loss_entropy
      - mixed/inner/loss_area
      - mixed/inner/loss_pose_reg
      - mixed/inner/is_shape_step

    Per-scan logs (after refinement):
      - mixed/refined_scan/scan_id
      - mixed/refined_scan/dice_2d
      - mixed/refined_scan/iou_2d
      - mixed/refined_scan/dice_3d
      - mixed/refined_scan/iou_3d
      - mixed/refined_scan/dice_3d_central
      - mixed/refined_scan/iou_3d_central
      - mixed/refined_scan/inner_steps

    Aggregate logs:
      - mixed/dice_2d_{mean, std, min, max}
      - mixed/iou_2d_{mean, std, min, max}
      - mixed/dice_3d_{mean, std, min, max}
      - mixed/iou_3d_{mean, std, min, max}
      - mixed/dice_3d_central_{mean, std, min, max}
      - mixed/iou_3d_central_{mean, std, min, max}
    """
    device = torch.device(config["device"])
    refine_steps = int(config.get("mixed_refine_steps", 200))

    dice_2d_list, iou_2d_list = [], []
    dice_3d_list, iou_3d_list = [], []
    dice_3d_c_list, iou_3d_c_list = [], []

    print(f"\n{'='*70}")
    print(f"MIXED MODE REFINEMENT (per-scan from global prior)")
    print(f"{'='*70}")
    print(f"Test scans: {len(test_scans)}, Refine steps: {refine_steps}")

    for scan_idx, scan in enumerate(test_scans):
        print(f"\n[{scan_idx+1}/{len(test_scans)}] Refining {scan.scan_id}...")

        # Clone global weights
        model_local = core.ImplicitNeuralRepresentation(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_inr_layers"],
        ).to(device)
        model_local.load_state_dict(model_global.state_dict())

        # Fresh pose layer
        pose_layer = core.PoseParameters(config["num_views"]).to(device)
        if not config.get("learn_pose", False):
            with torch.no_grad():
                pose_layer.pose.zero_()

        # Optimization with logging
        inner_hist = optimize_subject_refined_with_logging(
            model_local,
            scan.slices_2d,
            scan.contours_2d,
            pose_layer,
            scan.chosen_z,
            config,
            num_steps=refine_steps,
            D=scan.D,
            seg_vol=scan.seg_c,
            scan_id=scan.scan_id,
        )

        # Evaluate refined
        metrics_2d = core.evaluate_subject_2d(
            model_local,
            scan.contours_2d,
            pose_layer,
            scan.chosen_z,
            scan.D,
            config,
        )
        metrics_3d = core.evaluate_subject_3d_and_mesh(
            model_local,
            scan.seg_c,
            scan.scan_id + "_mixed_refined",
            config,
            scan.chosen_z,
        )

        # Log refined scan
        refined_log = {
            "mixed/refined_scan/scan_id": scan.scan_id,
            "mixed/refined_scan/dice_2d": metrics_2d["dice_2d"],
            "mixed/refined_scan/iou_2d": metrics_2d["iou_2d"],
            "mixed/refined_scan/dice_3d": metrics_3d["dice_3d"],
            "mixed/refined_scan/iou_3d": metrics_3d["iou_3d"],
            "mixed/refined_scan/dice_3d_central": metrics_3d["dice_3d_central"],
            "mixed/refined_scan/iou_3d_central": metrics_3d["iou_3d_central"],
            "mixed/refined_scan/inner_steps": len(inner_hist),
        }
        wandb.log(refined_log)

        dice_2d_list.append(metrics_2d["dice_2d"])
        iou_2d_list.append(metrics_2d["iou_2d"])
        dice_3d_list.append(metrics_3d["dice_3d"])
        iou_3d_list.append(metrics_3d["iou_3d"])
        dice_3d_c_list.append(metrics_3d["dice_3d_central"])
        iou_3d_c_list.append(metrics_3d["iou_3d_central"])

        print(
            f"  Refined: 2D Dice={metrics_2d['dice_2d']:.4f}, "
            f"3D Dice={metrics_3d['dice_3d']:.4f}, "
            f"3D Dice(c)={metrics_3d['dice_3d_central']:.4f}"
        )

    # Aggregate
    def agg_stats(lst):
        return {
            "mean": float(np.mean(lst)),
            "std": float(np.std(lst)),
            "min": float(np.min(lst)),
            "max": float(np.max(lst)),
        }

    summary = {}
    for key, lst in [
        ("dice_2d", dice_2d_list),
        ("iou_2d", iou_2d_list),
        ("dice_3d", dice_3d_list),
        ("iou_3d", iou_3d_list),
        ("dice_3d_central", dice_3d_c_list),
        ("iou_3d_central", iou_3d_c_list),
    ]:
        stats = agg_stats(lst)
        for stat_name, stat_val in stats.items():
            summary[f"mixed/{key}_{stat_name}"] = stat_val

    # Log aggregate
    wandb.log(summary)

    return summary


def optimize_subject_refined_with_logging(
    model: nn.Module,
    slices_2d: torch.Tensor,
    contours_2d: torch.Tensor,
    pose_layer: core.PoseParameters,
    chosen: List[int],
    config: Dict,
    num_steps: int,
    D: int,
    seg_vol: torch.Tensor,
    scan_id: str,
) -> List[Dict]:
    """
    Refine a single subject from global prior, logging all losses per step.

    Returns:
      history: list of dicts, one per step
    """
    device = torch.device(config["device"])
    learn_pose = bool(config.get("learn_pose", False))
    pose_reg_weight = float(config.get("pose_reg_weight", 0.0))
    vol_weight = float(config.get("vol_supervision_weight", 0.0))
    vol_samples = int(config.get("vol_supervision_samples", 0))
    reg_every = int(config.get("reg_every", 1))
    proj_batch_size = int(config.get("proj_batch_size", 65536))
    grid_res = int(config.get("grid_resolution", 64))
    alternate_every = int(config.get("alternate_every", 20))

    model = model.to(device)
    pose_layer = pose_layer.to(device)
    seg_device = seg_vol.to(device)
    contours_device = contours_2d.to(device)

    D_seg, H_seg, W_seg = seg_device.shape
    if D_seg != D:
        D = D_seg

    optimizer_shape = optim.Adam(model.parameters(), lr=config["learning_rate"])
    if learn_pose:
        optimizer_pose = optim.Adam(
            pose_layer.parameters(), lr=config["pose_learning_rate"]
        )
        scheduler_pose = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_pose, T_max=num_steps, eta_min=config["pose_learning_rate"] / 10
        )
    else:
        optimizer_pose = None
        scheduler_pose = None

    scheduler_shape = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_shape, T_max=num_steps, eta_min=config["learning_rate"] / 10
    )

    history = []

    for step in range(num_steps):
        if learn_pose:
            shape_step = ((step // alternate_every) % 2 == 0)
        else:
            shape_step = True

        if shape_step:
            extrinsics = pose_layer.get_matrices(device=device).detach()
        else:
            extrinsics = pose_layer.get_matrices(device=device)

        preds = core.project_slices_from_inr_batch(
            model,
            extrinsics,
            chosen,
            D,
            config["proj_resolution"],
            device,
            batch_size=proj_batch_size,
        )

        loss_projection = core.contour_bce_loss(preds, contours_device)

        vol_loss = torch.tensor(0.0, device=device)
        loss_smooth = torch.tensor(0.0, device=device)
        loss_entropy = torch.tensor(0.0, device=device)
        loss_area = torch.tensor(0.0, device=device)
        loss_pose_reg = torch.tensor(0.0, device=device)

        if shape_step:
            do_reg = (reg_every <= 1) or (step % reg_every == 0)
            if do_reg:
                if vol_weight > 0.0 and vol_samples > 0:
                    coords3d = torch.rand(vol_samples, 3, device=device) * 2.0 - 1.0
                    pred_vol = model(coords3d).squeeze(-1)

                    x_norm = (coords3d[:, 0] + 1.0) * 0.5
                    y_norm = (coords3d[:, 1] + 1.0) * 0.5
                    z_norm = (coords3d[:, 2] + 1.0) * 0.5

                    x_idx = torch.clamp((x_norm * (H_seg - 1)).long(), 0, H_seg - 1)
                    y_idx = torch.clamp((y_norm * (W_seg - 1)).long(), 0, W_seg - 1)
                    z_idx = torch.clamp((z_norm * (D_seg - 1)).long(), 0, D_seg - 1)

                    gt_vol = seg_device[z_idx, x_idx, y_idx].view(-1)
                    vol_loss = F.binary_cross_entropy(pred_vol, gt_vol)

                occ_grid = model.sample_grid(
                    resolution=grid_res,
                    device=device,
                    requires_grad=True,
                )
                loss_smooth = core.laplacian_smoothness_loss(occ_grid, weight=0.02)
                loss_entropy = core.volume_entropy_loss(occ_grid, weight=0.01)
                loss_area = core.surface_area_loss(occ_grid, weight=0.005)

            loss = (
                loss_projection
                + loss_smooth
                + loss_entropy
                + loss_area
                + vol_weight * vol_loss
            )
        else:
            loss = loss_projection
            if pose_reg_weight > 0.0:
                loss_pose_reg = pose_reg_weight * (pose_layer.pose ** 2).mean()
                loss = loss + loss_pose_reg

        optimizer_shape.zero_grad()
        if learn_pose and optimizer_pose is not None:
            optimizer_pose.zero_grad()

        loss.backward()

        if shape_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_shape.step()
            scheduler_shape.step()
        else:
            if learn_pose and optimizer_pose is not None:
                torch.nn.utils.clip_grad_norm_(pose_layer.parameters(), max_norm=1.0)
                optimizer_pose.step()
                scheduler_pose.step()

        # Log step
        log_dict = {
            "mixed/inner/scan_id": scan_id,
            "mixed/inner/step": step,
            "mixed/inner/loss_total": loss.item(),
            "mixed/inner/loss_projection": loss_projection.item(),
            "mixed/inner/loss_vol_bce": vol_loss.item(),
            "mixed/inner/loss_smooth": loss_smooth.item(),
            "mixed/inner/loss_entropy": loss_entropy.item(),
            "mixed/inner/loss_area": loss_area.item(),
            "mixed/inner/loss_pose_reg": loss_pose_reg.item(),
            "mixed/inner/is_shape_step": float(shape_step),
        }
        wandb.log(log_dict)
        history.append(log_dict)

        if step % config.get("print_every", 100) == 0 or step == num_steps - 1:
            print(
                f"    step {step}/{num_steps}: "
                f"loss={loss.item():.6f}, proj={loss_projection.item():.6f}, "
                f"shape_step={shape_step}"
            )

    return history


def run_mixed_experiment():
    """
    Main entry point for a single sweep run.
    Pulls hyperparams from wandb.config and logs everything.
    """
    # Merge base + sweep config
    config = copy.deepcopy(BASE_CONFIG)
    sweep_cfg = dict(wandb.config)
    for k, v in sweep_cfg.items():
        config[k] = v
    config["mode"] = "mixed"

    core.set_global_seed(int(config.get("random_seed", 42)))
    device = torch.device(config["device"])

    print(f"\n{'='*70}")
    print(f"STARTING MIXED MODE SWEEP RUN")
    print(f"{'='*70}")

    # Build dataset
    dataset = core.build_scan_dataset(config)
    train_scans = dataset["train"]
    val_scans = dataset["val"]
    test_scans = dataset["test"]

    if not train_scans:
        print("ERROR: No training scans")
        wandb.log({"error/no_train_scans": 1})
        return

    # Log dataset info
    wandb.log({
        "dataset/n_train": len(train_scans),
        "dataset/n_val": len(val_scans),
        "dataset/n_test": len(test_scans),
    })
    wandb.summary["dataset/n_train"] = len(train_scans)
    wandb.summary["dataset/n_val"] = len(val_scans)
    wandb.summary["dataset/n_test"] = len(test_scans)

    run_start = time.time()

    # 1) Train global INR
    try:
        model_global = train_global_inr_with_logging(train_scans, val_scans, config)
    except Exception as e:
        print(f"ERROR in global training: {e}")
        import traceback
        traceback.print_exc()
        wandb.log({"error/training": str(e)})
        return

    # 2) Test global (no refinement)
    print(f"\n{'='*70}")
    print(f"TEST: GLOBAL INR (NO REFINEMENT)")
    print(f"{'='*70}")
    try:
        test_global_metrics = evaluate_split_with_logging(
            model_global, test_scans, config, split_name="test_global", device=device
        )
        wandb.log(test_global_metrics)
        for k, v in test_global_metrics.items():
            wandb.summary[k] = v
    except Exception as e:
        print(f"ERROR in test_global evaluation: {e}")
        import traceback
        traceback.print_exc()
        wandb.log({"error/test_global": str(e)})

    # 3) Mixed refinement
    try:
        mixed_metrics = refine_test_scans_with_logging(
            model_global, test_scans, config
        )
        wandb.log(mixed_metrics)
        for k, v in mixed_metrics.items():
            wandb.summary[k] = v
    except Exception as e:
        print(f"ERROR in mixed refinement: {e}")
        import traceback
        traceback.print_exc()
        wandb.log({"error/mixed_refinement": str(e)})

    run_time = time.time() - run_start
    wandb.summary["run_time_seconds"] = run_time
    print(f"\nRun completed in {run_time:.1f}s")


if __name__ == "__main__":
    wandb.init(project="echo3d-mixed-sweep")
    run_mixed_experiment()
