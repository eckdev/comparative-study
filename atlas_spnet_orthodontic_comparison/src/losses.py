import torch
import torch.nn.functional as F


SYMMETRY_PAIRS = [(13, 16), (14, 15), (17, 18), (19, 20), (21, 22)]
MIDLINE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def pairwise_structure_loss(pred, target):
    pd = torch.cdist(pred, pred)
    td = torch.cdist(target, target)
    mask = torch.triu(torch.ones(pd.shape[1], pd.shape[2], dtype=torch.bool, device=pd.device), diagonal=1)
    return F.smooth_l1_loss(pd[:, mask], td[:, mask])


def symmetry_loss(pred):
    mid_x = pred[:, MIDLINE, 0].mean(dim=1)
    losses = []
    for left, right in SYMMETRY_PAIRS:
        lp = pred[:, left]
        rp = pred[:, right]
        losses.append(F.smooth_l1_loss((lp[:, 0] + rp[:, 0]) * 0.5, mid_x) + 0.25 * F.smooth_l1_loss(lp[:, 1:], rp[:, 1:]))
    return torch.stack(losses).mean()


def clinical_loss(pred, target, scale, threshold_mm=2.0, margin_mm=0.5):
    err = torch.linalg.norm(pred - target, dim=-1) * scale.view(-1, 1)
    return F.softplus((err - float(threshold_mm)) / max(float(margin_mm), 1e-6)).mean()


def confidence_nll(pred, target, log_vars):
    sq = torch.linalg.norm(pred - target, dim=-1).pow(2)
    log_vars = log_vars.clamp(-6.0, 6.0)
    return (0.5 * torch.exp(-log_vars) * sq + 0.5 * log_vars).mean()


def atlas_loss(outputs, landmarks_norm, scale, args):
    pred = outputs["pred_norm"]
    coord_per_lm = F.smooth_l1_loss(pred, landmarks_norm, reduction="none").mean(dim=-1)
    coord = coord_per_lm.mean()
    coarse = F.smooth_l1_loss(outputs["coarse_norm"], landmarks_norm)
    nll = confidence_nll(pred, landmarks_norm, outputs["log_vars"])
    structure = pairwise_structure_loss(pred, landmarks_norm)
    sym = symmetry_loss(pred)
    clinical = clinical_loss(pred, landmarks_norm, scale, args.clinical_threshold_mm, args.clinical_margin_mm)
    loss = (
        float(args.coord_weight) * coord
        + float(args.coarse_weight) * coarse
        + float(args.confidence_weight) * nll
        + float(args.structure_weight) * structure
        + float(args.symmetry_weight) * sym
        + float(args.clinical_weight) * clinical
    )
    return loss, {
        "coord_loss": float(coord.detach().cpu()),
        "coarse_loss": float(coarse.detach().cpu()),
        "confidence_loss": float(nll.detach().cpu()),
        "structure_loss": float(structure.detach().cpu()),
        "symmetry_loss": float(sym.detach().cpu()),
        "clinical_loss": float(clinical.detach().cpu()),
        "per_landmark_coord_loss": coord_per_lm.detach().mean(dim=0).cpu().numpy().astype(float).tolist(),
    }
