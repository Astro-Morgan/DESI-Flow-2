"""
The fiducial instrument + sky template set (frozen 2026-11-17): a single file with the templates and the calibrated priors, and a loader.

Contents of artifacts/fiducial_templates_v1.npz
  wavelength_A (7781,)                       the brz grid, 3600 + 0.8 i
  inst_templates_flux (23, 7781)             per-camera instrument templates (b K=5, r K=6, z K=12; long-trained, patience-annealed, whitened-orthonormal on SKY-fibre residuals)
                                             on the merged grid, already multiplied by the camera ramps (partition of unity across the two overlaps)
  inst_camera (23,)                          camera letter of every column
  camera_weights (3, 7781)                   the ramps of b, r, z
  sky_templates_flux (64, 7781)              sky templates (K=64, DR1 VAC, long-trained)
  prior_{dark,bright,backup,all}_{mu,S,nu}   multivariate Student-t prior of the 23 instrument coefficients per observing program: location, scatter MATRIX (23 x 23, full covariance
                                             including the cross-camera blocks), degrees of freedom; fitted by EM to the coefficients of 30,000 SKY rows, estimation noise removed
  skyprior_{dark,bright,backup,all}_{S,nu}   Student-t prior (zero location) of the 64 sky coefficients per program, CALIBRATED 2026-11-17: scatter matrix eps_program^2 x the covariance of the VAC
                                             coefficients of that program (eps 0.0056 dark, 0.0032 bright, 0.032 backup, fitted on the SKY-row residuals), nu = 10; 'all' = the bright prior (fallback)
  skyprior_inflation (1.0 = the pure SKY-row calibration, the shipped default), skyprior_lineness (64,) the scale of template k can be multiplied by inflation ** lineness_k at load time (lineness = fraction of the template's energy in the sky-line pixels, top 10% of the
                                             sky-template energy): the SKY-row calibration describes sky-fibre residuals that are in-sample for the sky model and is far too tight for the LINE residuals of target
                                             fibres (SPS hold-out test, docs/sky_instrument_templates.md), while a uniform inflation lets the broad templates compete with the physical flux; the stored scatter matrices are the
                                             calibrated ones; the default is inflation = 1 (pure calibration), inflation = 30 is the SPS-tuned option (sky_prior(program, inflation=30))
  sky_prior_sigma (64,)                      the earlier assumed Gaussian sigma (1% of the VAC std), kept for reference
  provenance                                 JSON string
Shape is what is frozen; the scale of the instrument response can be changed after the fact through the prior (`rescale_prior`).

Use:  from DESIFlow.templates.fiducial import load_fiducial; fid = load_fiducial()
"""
import json
import os

import numpy as np

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts", "fiducial_templates_v1.npz")


def build(root, phys, legacy, out=PATH):
    from DESIFlow.templates import joint_fit_demo as jd
    fid = jd.Fiducial(root, phys, legacy); zt = np.load(os.path.join(root, "instrument_prior_t.npz")); W = jd.camera_weights(); d = {}
    for q in ("dark", "bright", "backup", "all"):
        d[f"prior_{q}_mu"] = zt[f"mu_{q}"]; d[f"prior_{q}_S"] = zt[f"S_{q}"]; d[f"prior_{q}_nu"] = np.array(float(zt[f"nu_{q}"]))
    prov = dict(date="2026-11-17", instrument="per-camera long-trained (patience annealing), K b5 r6 z12, spline K* 5.2/5.6/11.8, SKY rows split by tile, cache/sky_templates_v2/long/seed0",
                sky="K64, long-trained, DR1 VAC with assumed weights 1/(0.1^2+(0.03 f)^2), plateau from K=64", sky_prior="Student-t per program, scatter eps^2 x VAC covariance, eps fitted on SKY-row residuals (held-out log density gain 0.2-6.7 nats per row over the 1% diagonal), nu 10", prior="Student-t per program (EM, nu on a grid, lower edge 1.5), 30000 SKY rows, noise-deconvolved scatter, full 23x23 covariance",
                verified="fastspecfit-SPS fits on 300 clean galaxies: real vs reversed controls, hidden chi2/px -0.0217 vs -0.0036 (docs/sky_instrument_templates.md)")
    zs = np.load(os.path.join(root, "sky_prior.npz"))
    for q in ("dark", "bright", "backup"):
        d[f"skyprior_{q}_S"] = float(zs[f"eC_{q}"]) ** 2 * zs[f"Svac_{q}"]; d[f"skyprior_{q}_nu"] = np.array(float(zs[f"nuC_{q}"]))
    d["skyprior_all_S"] = d["skyprior_bright_S"]; d["skyprior_all_nu"] = d["skyprior_bright_nu"]; d["skyprior_inflation"] = np.array(1.0)
    e = np.sum(fid.sky ** 2, 0); d["skyprior_lineness"] = (fid.sky[:, e > np.percentile(e, 90)] ** 2).sum(1) / (fid.sky ** 2).sum(1)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, wavelength_A=jd.LAM, inst_templates_flux=fid.inst.astype(np.float32), inst_camera=fid.inst_kind, camera_weights=np.array([W[c] for c in "brz"], np.float32),
                        sky_templates_flux=fid.sky.astype(np.float32), sky_prior_sigma=fid.prior_sky, provenance=json.dumps(prov), **d)
    return out


class Fiducial:
    def __init__(self, path=PATH):
        z = np.load(path, allow_pickle=False); self.wavelength = z["wavelength_A"]; self.inst = z["inst_templates_flux"].astype(np.float64); self.camera = z["inst_camera"]; self.camera_weights = z["camera_weights"]
        self.sky_inflation = float(z["skyprior_inflation"]) if "skyprior_inflation" in z.files else 1.0; self.sky_lineness = z["skyprior_lineness"] if "skyprior_lineness" in z.files else np.ones(64)
        self.sky = z["sky_templates_flux"].astype(np.float64); self.sky_sigma = z["sky_prior_sigma"]; self.provenance = json.loads(str(z["provenance"])); self.priors = {}; self.sky_priors = {}
        for q in ("dark", "bright", "backup", "all"):
            self.priors[q] = dict(mu=z[f"prior_{q}_mu"], S=z[f"prior_{q}_S"], nu=float(z[f"prior_{q}_nu"]))
            if f"skyprior_{q}_S" in z.files:
                self.sky_priors[q] = dict(S=z[f"skyprior_{q}_S"], nu=float(z[f"skyprior_{q}_nu"]))

    def prior(self, program):
        """(mu, S^-1, nu) of the Student-t prior of a program (falls back to 'all')."""
        p = self.priors.get(program, self.priors["all"]); return p["mu"], np.linalg.inv(p["S"]), p["nu"]

    def sky_prior(self, program, inflation=None):
        """(mu = 0, S^-1, nu) of the Student-t prior of the sky coefficients of a program (falls back to 'all'); the scale is the calibrated one times `inflation` (default: the stored factor, 30; 1 = the pure SKY-row calibration)."""
        f = self.sky_inflation if inflation is None else inflation; p = self.sky_priors.get(program, self.sky_priors["all"]); D = f ** self.sky_lineness; return np.zeros(len(self.sky)), np.linalg.inv(p["S"] * np.outer(D, D)), p["nu"]

    def rescale_prior(self, program, scale=1.0, shift=0.0):
        """post-hoc change of the amplitude prior: scatter x scale^2 (scale > 1 lets the instrument response be larger), location + shift."""
        p = self.priors.get(program, self.priors["all"]); return p["mu"] + shift, np.linalg.inv(p["S"] * scale ** 2), p["nu"]


def load_fiducial(path=PATH):
    return Fiducial(path)
