"""
Validation metrics and plot inputs for the pretraining run.

Everything is evaluated on a fixed validation set (fixed hidden spans per object, so numbers are comparable across steps) held on the GPU.
Per object the model is run twice, both times decoding ALL 7781 native pixels:
    masked input   (the training condition): chi2 on the hidden pixels = MASKED RECONSTRUCTION; chi2 on every good pixel = TOTAL RECONSTRUCTION
    unmasked input (the deployment condition): chi2 on every good pixel, and chi2 of 8-pixel ivar-weighted bins (1 = noise-limited smooth model),
                    plus the redshift read from the content latents.
chi2 = mean of (flux - s * output)^2 * ivar over the pixels, so pure noise gives 1.
"""
from collections import defaultdict
import numpy as np
import torch
from DESIFlow.training.masking import SpanMasker, apply_hidden, N_PIX
from DESIFlow.training.loss import hidden_loss

Z_BINS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 100.0]
SNR_BINS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 1e9]


class Evaluator:
    def __init__(self, data, val_idx, device, seed=0, chunk=64, plot_z=(0.3, 1.0, 2.0, 3.0, 4.0), huber_c=5.0):
        self.device, self.chunk, self.huber_c = device, chunk, huber_c
        flux, ivar, mask = data.read(val_idx)
        self.n = len(val_idx)
        self.flux, self.ivar, self.mask = (torch.tensor(a, device=device) for a in (flux, ivar, mask))
        self.z = torch.tensor(data.z[val_idx], dtype=torch.float32, device=device)
        self.z_ok = torch.tensor(data.trusted(val_idx), device=device)
        good = (mask == 0) & (ivar > 0)
        masker = SpanMasker()
        self.hidden = torch.tensor(np.concatenate([masker.sample(good[i:i + 1], np.random.default_rng([seed, 31, i]))[0] for i in range(self.n)]), device=device)
        self.snr = np.array([np.median((flux[i] * np.sqrt(ivar[i]))[good[i]]) if good[i].any() else 0.0 for i in range(self.n)])
        self.plot_pos = self._pick_plot_objects(np.asarray(data.z[val_idx]), plot_z)
        pm = SpanMasker(frac=(0.25, 0.25), camera_prob=0.0)
        self.plot_hidden = torch.tensor(np.concatenate([pm.sample(good[p:p + 1], np.random.default_rng([seed, 41, int(p)]))[0] for p in self.plot_pos]), device=device)

    def _pick_plot_objects(self, z, targets):
        """For each target redshift: among validation objects within a z window, the one whose S/N is closest to that window's median."""
        picks = []
        for zt in targets:
            win = max(0.1, 0.05 * (1 + zt))
            cand = np.flatnonzero((np.abs(z - zt) < win) & (self.snr > 0) & ~np.isin(np.arange(len(z)), picks))
            if len(cand) == 0:                                               # nothing in the window: nearest unused object in z
                free = np.flatnonzero(~np.isin(np.arange(len(z)), picks))
                cand = np.array([int(free[np.argmin(np.abs(z[free] - zt))])])
            picks.append(int(cand[np.argmin(np.abs(self.snr[cand] - np.median(self.snr[cand])))]))
        return picks

    @torch.no_grad()
    def run(self, model, head, qv):
        was_training = model.training
        model.eval()
        per = defaultdict(list)
        zpred = []
        for a in range(0, self.n, self.chunk):
            sl = slice(a, a + self.chunk)
            flux, ivar, mask, hid = self.flux[sl], self.ivar[sl], self.mask[sl], self.hidden[sl]
            x = torch.stack([flux, ivar, mask], 1)
            good = (mask == 0) & (ivar > 0)
            lat = model.encode(apply_hidden(x, hid))
            s = 10 ** lat[:, 0, 0]
            pred = model.decoder(lat[:, 1:], qv)
            r2 = (flux - s.unsqueeze(-1) * pred) ** 2 * ivar
            per["hid_r2"].append((r2 * hid).sum(1)); per["hid_n"].append(hid.sum(1))
            per["good_r2"].append((r2 * good).sum(1)); per["good_n"].append(good.sum(1))
            per["huber"].append(hidden_loss(pred, s, flux, ivar, hid, self.huber_c).expand(len(x)) * hid.sum(1))
            lat_u = model.encode(x)
            s_u = 10 ** lat_u[:, 0, 0]
            res = flux - s_u.unsqueeze(-1) * model.decoder(lat_u[:, 1:], qv)
            per["unm_r2"].append((res ** 2 * ivar * good).sum(1))
            w8 = (ivar * good)[:, :7776].reshape(len(x), -1, 8)
            num, den = (w8 * res[:, :7776].reshape(len(x), -1, 8)).sum(-1), w8.sum(-1)
            ok = den > 0
            per["bin_chi"].append(((num ** 2 / den.clamp_min(1e-30)) * ok).sum(1)); per["bin_n"].append(ok.sum(1))
            if head is not None:
                zpred.append(head(lat_u[:, 1:]))
        cat = {k: torch.cat(v).double().cpu().numpy() for k, v in per.items()}
        z = self.z.cpu().numpy().astype(np.float64)
        zb = np.digitize(z, Z_BINS) - 1
        out = {"val_masked_chi2": cat["hid_r2"].sum() / cat["hid_n"].sum(), "val_masked_huber": cat["huber"].sum() / cat["hid_n"].sum(),
               "val_total_chi2": cat["good_r2"].sum() / cat["good_n"].sum(), "val_unmasked_chi2": cat["unm_r2"].sum() / cat["good_n"].sum(),
               "val_unmasked_bin8": cat["bin_chi"].sum() / cat["bin_n"].sum(), "val_hidden_fraction": cat["hid_n"].sum() / cat["good_n"].sum()}
        out["masked_chi2_by_z"] = {f"{Z_BINS[k]:g}-{Z_BINS[k + 1]:g}": float(cat["hid_r2"][zb == k].sum() / max(cat["hid_n"][zb == k].sum(), 1)) for k in range(len(Z_BINS) - 1) if (zb == k).any()}
        sb = np.digitize(self.snr, SNR_BINS) - 1
        out["masked_chi2_by_snr"] = {f"{SNR_BINS[k]:g}-{SNR_BINS[k + 1]:g}": float(cat["hid_r2"][sb == k].sum() / max(cat["hid_n"][sb == k].sum(), 1)) for k in range(len(SNR_BINS) - 1) if (sb == k).any()}
        out["unmasked_chi2_by_snr"] = {f"{SNR_BINS[k]:g}-{SNR_BINS[k + 1]:g}": float(cat["unm_r2"][sb == k].sum() / max(cat["good_n"][sb == k].sum(), 1)) for k in range(len(SNR_BINS) - 1) if (sb == k).any()}
        scatter = None
        if head is not None:
            lp = torch.cat(zpred).double().cpu().numpy()
            ok = self.z_ok.cpu().numpy()
            zp = np.expm1(lp)
            dz = (zp - z) / (1 + z)
            dzo = dz[ok]
            out.update({"z_rmse_log1p": float(np.sqrt(np.mean((lp - np.log1p(z))[ok] ** 2))), "z_bias": float(np.median(dzo)),
                        "z_sigma_nmad": float(1.4826 * np.median(np.abs(dzo - np.median(dzo)))), "z_cat_frac": float(np.mean(np.abs(dzo) > 0.15))})
            scatter = (z, zp)
        model.train(was_training)
        return {k: (float(v) if not isinstance(v, dict) else v) for k, v in out.items()}, scatter

    @torch.no_grad()
    def plot_items(self, model, head, qv):
        """Arrays for the 5 plotted spectra: data, unmasked-input output, masked-input reconstruction, redshift prediction."""
        was_training = model.training
        model.eval()
        items = []
        for j, p in enumerate(self.plot_pos):
            flux, ivar, mask, hid = self.flux[p:p + 1], self.ivar[p:p + 1], self.mask[p:p + 1], self.plot_hidden[j:j + 1]
            x = torch.stack([flux, ivar, mask], 1)
            lat_m = model.encode(apply_hidden(x, hid))
            rec = (10 ** lat_m[:, 0, 0, None] * model.decoder(lat_m[:, 1:], qv))[0].cpu().numpy()
            lat_u = model.encode(x)
            den = (10 ** lat_u[:, 0, 0, None] * model.decoder(lat_u[:, 1:], qv))[0].cpu().numpy()
            zp = float(torch.expm1(head(lat_u[:, 1:]))[0]) if head is not None else float("nan")
            items.append({"flux": flux[0].cpu().numpy(), "ivar": ivar[0].cpu().numpy(), "good": ((mask == 0) & (ivar > 0))[0].cpu().numpy(),
                          "hidden": hid[0].cpu().numpy(), "rec_masked": rec, "denoised": den, "z": float(self.z[p]), "z_pred": zp, "snr": float(self.snr[p]), "val_pos": int(p)})
        model.train(was_training)
        return items
