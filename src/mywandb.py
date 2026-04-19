#!/usr/bin/env python3
"""
Mixed mode sweep agent with EXHAUSTIVE logging to stdout.
All outputs visible in terminal + wandb logs.
"""

import copy
import json
import random
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

import FINAL_2_gpu_optmized as core

BASE_CONFIG = core.CONFIG


def log_print(msg: str, level: str = "INFO"):
    """Print to both stdout and flush immediately."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] [{level}] {msg}"
    print(full_msg, flush=True)
    sys.stdout.flush()


def train_global_inr_with_logging(
    train_scans: List,
    val_scans: List,
    config: Dict,
) -> nn.Module:
    """Train shared INR with comprehensive logging."""
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

    log_print("="*80)
    log_print("TRAINING GLOBAL INR (BASELINE B)")
    log_print("="*80)
    log_print(f"Config: hidden_dim={config['hidden_dim']}, layers={config['num_inr_layers']}")
    log_print(f"Learning: lr={config['learning_rate']}, epochs={config['num_epochs']}, steps/epoch={config['steps_per_epoch']}")
    log_print(f"Volumetric: weight={vol_weight}, samples={vol_samples}")
    log_print(f"Train scans: {len(train_scans)}, Val scans: {len(val_scans)}")

    for epoch in range(config["num_epochs"]):
        epoch_start = time.time()
        model.train()
        epoch_losses = []

        for step_in_epoch in range(config["steps_per_epoch"]):
            scan = random.choice(train_scans)
            seg_device = scan.seg_c.to(device)
            D_seg, H_seg, W_seg = seg_device.shape
            contours_device = scan.contours_2d.to(device)

            pose_layer = core.PoseParameters(config["num_views"])
            with torch.no_grad():
                pose_layer.pose.zero_()
            pose_layer = pose_layer.to(device)

            extrinsics = pose_layer.get_matrices(device=device).detach()

            preds = core.project_slices_from_inr_batch(
                model,
                extrinsics,
                scan.chosen_z,
                D_seg,
                config["proj_resolution"],
                device,
                batch_size=proj_batch_size,
            )
            if contours_device.shape[-1] != config["proj_resolution"]:
                contours_target = F.interpolate(
                    contours_device.unsqueeze(1),  # (V, 1, H, W)
                    size=(config["proj_resolution"], config["proj_resolution"]),
                    mode="nearest"
                ).squeeze(1)  # (V, H, W)
            else:
                contours_target = contours_device

            loss_projection = core.contour_bce_loss(preds, contours_target)

            vol_loss = torch.tensor(0.0, device=device)
            loss_smooth = torch.tensor(0.0, device=device)
            loss_entropy = torch.tensor(0.0, device=device)
            loss_area = torch.tensor(0.0, device=device)

            do_reg = (reg_every <= 1) or (global_step % reg_every == 0)
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
            epoch_losses.append(total_loss.item())

            if global_step % 50 == 0 or step_in_epoch == config["steps_per_epoch"] - 1:
                log_print(
                    f"[TRAIN] Epoch {epoch}/{config['num_epochs']-1} "
                    f"Step {step_in_epoch}/{config['steps_per_epoch']-1} "
                    f"(global {global_step}/{total_steps-1}): "
                    f"loss={total_loss.item():.6f} "
                    f"proj={loss_projection.item():.6f} "
                    f"vol={vol_loss.item():.6f} "
                    f"smooth={loss_smooth.item():.6f} "
                    f"entropy={loss_entropy.item():.6f} "
                    f"area={loss_area.item():.6f} "
                    f"lr={current_lr:.2e}"
                )

            global_step += 1

        # Validation at end of epoch
        epoch_time = time.time() - epoch_start
        log_print(f"\n[EPOCH {epoch} COMPLETE] Time: {epoch_time:.1f}s, Mean loss: {np.mean(epoch_losses):.6f}")
        log_print(f"[VALIDATING] Running validation on {len(val_scans)} scans...")

        val_metrics = evaluate_split_with_logging(
            model, val_scans, config, split_name="global/val", device=device
        )

        val_log = {f"global/val/epoch": epoch}
        val_log.update(val_metrics)
        wandb.log(val_log)

        log_print(
            f"[VAL RESULTS - Epoch {epoch}] "
            f"Dice_2D: {val_metrics.get('global/val/dice_2d_mean', 0):.4f} ± {val_metrics.get('global/val/dice_2d_std', 0):.4f} | "
            f"Dice_3D: {val_metrics.get('global/val/dice_3d_mean', 0):.4f} ± {val_metrics.get('global/val/dice_3d_std', 0):.4f} | "
            f"Dice_3D_C: {val_metrics.get('global/val/dice_3d_central_mean', 0):.4f} ± {val_metrics.get('global/val/dice_3d_central_std', 0):.4f}\n"
        )

    training_time = time.time() - training_start
    log_print(f"\n[TRAINING COMPLETE] Total time: {training_time:.1f}s ({training_time/60:.1f} min)")
    wandb.summary["global/training_time_seconds"] = training_time

    return model

def evaluate_subject_2d_safe(
    model: nn.Module,
    contours_2d: torch.Tensor,
    pose_layer: nn.Module,
    chosen: List[int],
    D: int,
    config: Dict,
) -> Dict[str, float]:
    """
    Robust 2D evaluation that handles resolution mismatches dynamically.
    """
    device = torch.device(config["device"])
    model = model.to(device)
    contours = contours_2d.to(device)
    extrinsics = pose_layer.get_matrices(device=device)
    
    # 1. Project at config resolution
    preds = core.project_slices_from_inr_batch(
        model,
        extrinsics,
        chosen,
        D,
        config["proj_resolution"],
        device,
        batch_size=config.get("proj_batch_size", 65536),
    )  # (V, H_new, W_new)

    # 2. Resize GT contours if needed
    if contours.shape[-1] != config["proj_resolution"]:
        contours_target = F.interpolate(
            contours.unsqueeze(1),
            size=(config["proj_resolution"], config["proj_resolution"]),
            mode="nearest"
        ).squeeze(1)
    else:
        contours_target = contours

    dices: List[float] = []
    ious: List[float] = []

    for v in range(contours_target.shape[0]):
        pred = preds[v]
        pred_binary = (pred > 0.5).float()
        target_binary = (contours_target[v] > 0.5).float()

        intersection = torch.sum(pred_binary * target_binary)
        union = torch.sum(pred_binary) + torch.sum(target_binary)
        dice = (2 * intersection) / (union + 1e-6)
        iou = intersection / (union - intersection + 1e-6)

        dices.append(dice.item())
        ious.append(iou.item())

    return {"dice_2d": float(np.mean(dices)), "iou_2d": float(np.mean(ious))}


def evaluate_split_with_logging(
    model: nn.Module,
    scans: List,
    config: Dict,
    split_name: str = "test_global",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate with detailed logging."""
    if device is None:
        device = torch.device(config["device"])

    model = model.to(device)
    model.eval()

    dice_2d_list, iou_2d_list = [], []
    dice_3d_list, iou_3d_list = [], []
    dice_3d_c_list, iou_3d_c_list = [], []

    with torch.no_grad():
        for scan_idx, scan in enumerate(scans):
            log_print(f"  [{scan_idx+1}/{len(scans)}] Evaluating {scan.scan_id}...", level="EVAL")

            pose_layer = core.PoseParameters(config["num_views"])
            with torch.no_grad():
                pose_layer.pose.zero_()
            pose_layer = pose_layer.to(device)

            metrics_2d = evaluate_subject_2d_safe(
                model,
                scan.contours_2d,
                pose_layer,
                scan.chosen_z,
                scan.D,
                config,
            )

            metrics_3d = core.evaluate_subject_3d_and_mesh(
                model,
                scan.seg_c,
                scan.scan_id + f"_{split_name.replace('/', '_')}",
                config,
                scan.chosen_z,
            )

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

            dice_2d_list.append(metrics_2d["dice_2d"])
            iou_2d_list.append(metrics_2d["iou_2d"])
            dice_3d_list.append(metrics_3d["dice_3d"])
            iou_3d_list.append(metrics_3d["iou_3d"])
            dice_3d_c_list.append(metrics_3d["dice_3d_central"])
            iou_3d_c_list.append(metrics_3d["iou_3d_central"])

            log_print(
                f"    {scan.scan_id}: "
                f"2D Dice={metrics_2d['dice_2d']:.4f} | "
                f"3D Dice={metrics_3d['dice_3d']:.4f} | "
                f"3D Dice_C={metrics_3d['dice_3d_central']:.4f}",
                level="EVAL"
            )

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

    wandb.log(summary)
    
    log_print(
        f"\n[{split_name.upper()} SUMMARY] "
        f"Dice_2D: {summary.get(f'{split_name}/dice_2d_mean', 0):.4f} ± {summary.get(f'{split_name}/dice_2d_std', 0):.4f} | "
        f"Dice_3D: {summary.get(f'{split_name}/dice_3d_mean', 0):.4f} ± {summary.get(f'{split_name}/dice_3d_std', 0):.4f} | "
        f"Dice_3D_C: {summary.get(f'{split_name}/dice_3d_central_mean', 0):.4f} ± {summary.get(f'{split_name}/dice_3d_central_std', 0):.4f}\n",
        level="SUMMARY"
    )

    return summary


def refine_test_scans_with_logging(
    model_global: nn.Module,
    test_scans: List,
    config: Dict,
) -> Dict[str, float]:
    """Refine test scans with detailed logging."""
    device = torch.device(config["device"])
    refine_steps = int(config.get("mixed_refine_steps", 200))

    dice_2d_list, iou_2d_list = [], []
    dice_3d_list, iou_3d_list = [], []
    dice_3d_c_list, iou_3d_c_list = [], []

    log_print("\n" + "="*80)
    log_print("MIXED MODE REFINEMENT (Global Prior + Per-Scan Refinement)")
    log_print("="*80)
    log_print(f"Test scans: {len(test_scans)}, Refine steps: {refine_steps}\n")

    for scan_idx, scan in enumerate(test_scans):
        log_print(f"\n[REFINING {scan_idx+1}/{len(test_scans)}] {scan.scan_id}", level="REFINE")

        model_local = core.ImplicitNeuralRepresentation(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_inr_layers"],
        ).to(device)
        model_local.load_state_dict(model_global.state_dict())

        pose_layer = core.PoseParameters(config["num_views"]).to(device)
        if not config.get("learn_pose", False):
            with torch.no_grad():
                pose_layer.pose.zero_()

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

        metrics_2d = evaluate_subject_2d_safe(
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

        log_print(
            f"  REFINED: "
            f"2D Dice={metrics_2d['dice_2d']:.4f} | "
            f"3D Dice={metrics_3d['dice_3d']:.4f} | "
            f"3D Dice_C={metrics_3d['dice_3d_central']:.4f}",
            level="REFINE"
        )

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

    wandb.log(summary)
    
    log_print(
        f"\n[MIXED MODE SUMMARY] "
        f"Dice_2D: {summary.get('mixed/dice_2d_mean', 0):.4f} ± {summary.get('mixed/dice_2d_std', 0):.4f} | "
        f"Dice_3D: {summary.get('mixed/dice_3d_mean', 0):.4f} ± {summary.get('mixed/dice_3d_std', 0):.4f} | "
        f"Dice_3D_C: {summary.get('mixed/dice_3d_central_mean', 0):.4f} ± {summary.get('mixed/dice_3d_central_std', 0):.4f}\n",
        level="SUMMARY"
    )

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
    """Refine with detailed per-step logging."""
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
    log_print(f"  Starting {num_steps} refinement steps...", level="REFINE")

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

        # --- FIX: Resize GT contours ---
        if contours_device.shape[-1] != config["proj_resolution"]:
            contours_target = F.interpolate(
                contours_device.unsqueeze(1),
                size=(config["proj_resolution"], config["proj_resolution"]),
                mode="nearest"
            ).squeeze(1)
        else:
            contours_target = contours_device

        loss_projection = core.contour_bce_loss(preds, contours_target)

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

        if step % max(100, num_steps // 10) == 0 or step == num_steps - 1:
            log_print(
                f"    step {step}/{num_steps-1}: "
                f"loss={loss.item():.6f} "
                f"proj={loss_projection.item():.6f} "
                f"vol={vol_loss.item():.6f} "
                f"smooth={loss_smooth.item():.6f} "
                f"shape={shape_step}",
                level="INNER"
            )

    return history


def run_mixed_experiment():
    """Main entry point."""
    config = copy.deepcopy(BASE_CONFIG)
    sweep_cfg = dict(wandb.config)
    for k, v in sweep_cfg.items():
        config[k] = v
    config["mode"] = "mixed"

    core.set_global_seed(int(config.get("random_seed", 42)))
    device = torch.device(config["device"])

    run_start = time.time()

    log_print("\n" + "="*80)
    log_print("STARTING MIXED MODE SWEEP RUN")
    log_print("="*80)
    log_print(f"Device: {device}")
    log_print(f"Random seed: {config.get('random_seed', 42)}")
    log_print(f"Config: {json.dumps({k: v for k, v in config.items() if not isinstance(v, (dict, list, Path))}, indent=2, default=str)}\n")

    # Build dataset
    log_print("Building dataset...")
    dataset = core.build_scan_dataset(config)
    train_scans = dataset["train"]
    val_scans = dataset["val"]
    test_scans = dataset["test"]

    if not train_scans:
        log_print("ERROR: No training scans!", level="ERROR")
        wandb.log({"error/no_train_scans": 1})
        return

    log_print(f"Dataset loaded: train={len(train_scans)}, val={len(val_scans)}, test={len(test_scans)}\n")

    wandb.log({
        "dataset/n_train": len(train_scans),
        "dataset/n_val": len(val_scans),
        "dataset/n_test": len(test_scans),
    })

    # 1) Train global INR
    try:
        model_global = train_global_inr_with_logging(train_scans, val_scans, config)
    except Exception as e:
        log_print(f"ERROR in global training: {e}", level="ERROR")
        import traceback
        log_print(traceback.format_exc(), level="ERROR")
        wandb.log({"error/training": str(e)})
        return

    # 2) Test global
    log_print("\n" + "="*80)
    log_print("TEST: GLOBAL INR (NO REFINEMENT)")
    log_print("="*80)
    try:
        test_global_metrics = evaluate_split_with_logging(
            model_global, test_scans, config, split_name="test_global", device=device
        )
        wandb.log(test_global_metrics)
        for k, v in test_global_metrics.items():
            wandb.summary[k] = v
    except Exception as e:
        log_print(f"ERROR in test_global evaluation: {e}", level="ERROR")
        import traceback
        log_print(traceback.format_exc(), level="ERROR")
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
        log_print(f"ERROR in mixed refinement: {e}", level="ERROR")
        import traceback
        log_print(traceback.format_exc(), level="ERROR")
        wandb.log({"error/mixed_refinement": str(e)})

    run_time = time.time() - run_start
    wandb.summary["run_time_seconds"] = run_time
    log_print(f"\n{'='*80}")
    log_print(f"RUN COMPLETE in {run_time:.1f}s ({run_time/60:.1f} min)")
    log_print(f"{'='*80}\n")


if __name__ == "__main__":
    wandb.init(project="echo3d-mixed-sweep2")
    run_mixed_experiment()
