# Instrument and sky templates: DeepRock-style chi2 reconstruction, sweep over the number of templates

2026-11-06. Code: `src/DESIFlow/templates/deeprock_lite.py` (trainer), `sky_instrument_sweep.py` (data, sweep, figures); tests
`tests/test_deeprock_lite.py`. Figures: `docs/figures/template_chi2_vs_K`, `template_examples_instrument`,
`template_examples_sky` (each with a same-named `.npz` of the plotted arrays).

## Method

* DeepRock design of `deeprock_dr2.py`, observed frame, no redshift: K learnable templates `T`; per spectrum the coefficients are
  solved by weighted least squares `c = (T^T W T + ridge I)^-1 T^T W f` with gradient through the solve; AdamW (lr 3e-3, cosine,
  1500 steps, batch 512), initialised from a weighted SVD.
* Loss: masked-denoising chi2. 36 random spans (log-uniform 5-250 pixels, about 25% of the pixels) are hidden, the coefficients
  are fitted on the visible pixels and the chi2 per pixel is scored on the hidden ones. The reported "mean chi2" is this score on
  hold-out spectra (fixed masks), with the standard error of the mean over spectra, against the zero model (K = 0).
* Whitened units: `g = f s`, `omega = w / s^2`, `s_l = sqrt(mean ivar)`; identical chi2, better conditioned. Saved templates are
  an orthonormal basis of the learned span (in whitened units) ordered by the energy they carry, plus the same in flux units.
* Each K is trained independently (not nested); one seed.

## Data

| | instrument | sky |
|---|---|---|
| source | cached SKY rows (sky-fibre residuals after sky subtraction, calibrated) | DR1 sky VAC (9,176 per-exposure sky spectra) |
| weights | pipeline ivar and mask | assumed: ivar = 1 / (0.1^2 + (0.03 f)^2) (the VAC has none) |
| rows used | 150,000 train / 20,000 hold-out, TARGETID < 0 (tile derivable), >= 4000 good pixels | 8,372 train / 804 hold-out |
| split | by tile (tile % 20 == 0 held out: 705 of 13,577 tiles) | by night (night % 10 == 0: 32 nights) |

## Result: mean chi2 per pixel on hold-out spectra (hidden pixels)

| K | instrument (zero model 1.0143) | sky (zero model 288.9) |
|---|---|---|
| 1 | 0.9538 | 47.03 |
| 2 | 0.9469 | 20.41 |
| 3 | 0.9421 | 11.33 |
| 4 | 0.9404 | 6.63 |
| 6 | 0.9390 | 2.56 |
| 8 | 0.9380 | 1.58 |
| 12 | 0.9370 | 0.966 |
| 16 | 0.9364 | 0.756 |
| 20 | 0.9362 | (not run) |
| 24 | 0.9363 | 0.505 |
| 32 | 0.9371 | 0.431 |
| 48 | 0.9389 | 0.383 |
| 64 | 0.9412 | 0.367 |
| 80 | (not run) | 0.349 |
| 96 | (not run) | 0.353 |
| 128 | (not run) | 0.344 |

Standard errors: 0.002 (instrument), 0.02-1 (sky, larger at small K); the K ladder was extended on request (instrument K = 20, sky K = 80, 96, 128).

* Instrument: 6% of the 7.7% total gain comes from the first template; the curve is flat within the errors from K = 12 to 32 and
  rises beyond K = 32 while the in-sample (visible-pixel) chi2 keeps falling (0.9196 at K = 64): overfitting.
* Instrument K = 20 (added on request) lies on the plateau, 0.9362 +- 0.0022.
* Sky K = 80, 96 and 128 were added on request: the hold-out chi2 is flat from K = 48 (0.383) to 128 (0.344) within the standard error of
  about 0.03, while the in-sample chi2 keeps falling (0.187 at K = 128), so templates beyond about 48 are overfitting.
* Sky: the chi2 reaches 1 (the assumed 3% accuracy) at K = 12 and keeps improving slowly; the hold-out value separates from the
  in-sample one above K = 24 (0.505 / 0.404 at K = 24, 0.367 / 0.225 at K = 64).

## Files (not in the repository; `RepoClone/Claude/cache/sky_templates/`)

`instrument/K001.npz ... K064.npz` (13 sets, including K020) and `sky/K001.npz ... K128.npz` (15 sets), each with `templates_whitened` (K, 7781), `templates_flux`
(K, 7781), `pixel_scale_s`, `energy`, `wavelength_A`, the hold-out metrics, the per-row hold-out chi2 and the training-loss
history; `summary.json` per kind; the prepared training and hold-out arrays; `run.sh`.

## Caveats

* The instrument rows are the in-sample residuals of the sky fit (the sky model was fitted to these fibres): the structure the
  templates learn is what survives that fit, and may differ at target fibres.
* The sky chi2 values depend on the assumed weights; only the shape of the curve and the comparison with the zero model carry
  over. The recipe of the VAC is not documented.
* Differences between K = 12 and K = 32 (instrument) are within about one standard error; one seed.
