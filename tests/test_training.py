"""
Checks for the training pieces (src/DESIFlow/training): span/camera hiding, the hidden-pixel loss, the z head, PCGrad.
Run directly (python tests/test_training.py) or with pytest.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from DESIFlow.eidos.eidos import Eidos
from DESIFlow.eidos.reader import velocity
from DESIFlow.training.masking import SpanMasker, apply_hidden, run_lengths, CAMERA_RANGES, N_PIX
from DESIFlow.training.loss import hidden_loss, chi2_stats, normalized_residual, huber
from DESIFlow.training.zhead import ZHead
from DESIFlow.training.pcgrad import backward_with_surgery, project_out_conflict, _dot

TINY = dict(cnn_kwargs=dict(n_scales=6, branch_channels=4, mix_channels=8, down_channels=(8, 8, 8, 8), d_token=36),
            perceiver_kwargs=dict(n_latents=4, d_latent=36, n_heads=2, n_cross=1),
            decoder_kwargs=dict(d_query=16, n_heads=2))
NATIVE = torch.linspace(3600., 9824., N_PIX, dtype=torch.float64)


def real_batch(n, dtype=torch.float32):
    from astropy.io import fits
    d = fits.open(ROOT / "data" / "data.fits")[1].data
    x = np.stack([np.asarray(d[c], np.float32) for c in ("FLUX", "IVAR", "MASK")], 1)[:n]
    return torch.tensor(x, dtype=dtype), np.asarray(d["desi_z"], np.float64)[:n]


def test_span_masker():
    x, _ = real_batch(30)
    good = ((x[:, 2] == 0) & (x[:, 1] > 0)).numpy()
    m = SpanMasker()
    hid, span = m.sample(good, np.random.default_rng(0))
    assert not (hid & ~good).any(), "hidden pixels must be natively good"
    assert np.array_equal(hid, span & good)
    hid2, _ = m.sample(good, np.random.default_rng(0))
    assert np.array_equal(hid, hid2), "same seed, same masks"
    fr = np.concatenate([(m.sample(good, np.random.default_rng(s))[0].sum(1) / good.sum(1)) for s in range(20)])
    assert 0.04 < fr.min() and fr.max() < 0.45 and 0.15 < fr.mean() < 0.30, (fr.min(), fr.mean(), fr.max())
    lens = run_lengths(span)[span]
    assert lens.min() >= 3, lens.min()
    cam = SpanMasker(camera_prob=1.0)
    _, sp = cam.sample(good, np.random.default_rng(1))
    covered = [any(sp[b, a:e].all() for a, e in CAMERA_RANGES.values()) for b in range(len(good))]
    assert all(covered), "camera_prob=1 must hide a whole camera in every spectrum"
    return (f"hidden fraction min/mean/max {fr.min():.2f}/{fr.mean():.2f}/{fr.max():.2f}; span runs {lens.min()}..{lens.max()} px; "
            f"cameras b/r/z = " + ", ".join(f"{k}:{e - a}px" for k, (a, e) in CAMERA_RANGES.items()))


def test_hidden_pixels_are_invisible_to_the_model():
    torch.manual_seed(0)
    net = Eidos(**TINY).eval()
    x, _ = real_batch(3)
    hid = torch.from_numpy(SpanMasker().sample(((x[:, 2] == 0) & (x[:, 1] > 0)).numpy(), np.random.default_rng(2))[0])
    xa = apply_hidden(x, hid)
    assert (xa[:, 0][hid] == 0).all() and (xa[:, 1][hid] == 0).all() and (xa[:, 2][hid] != 0).all()
    assert torch.equal(xa[:, 0][~hid], x[:, 0][~hid]) and torch.equal(xa[:, 2][~hid], x[:, 2][~hid])
    xm = x.clone(); xm[:, 2][hid] = 1.0                      # mask bit only, flux and ivar left in place
    xr = x.clone(); xr[:, 0][hid] = torch.randn(int(hid.sum())) * 50; xr[:, 1][hid] = 1e3
    with torch.no_grad():
        la, lm, lr = net.encode(xa), net.encode(xm), net.encode(apply_hidden(xr, hid))
    assert torch.equal(la, lm), "setting only the mask bit must hide the pixel exactly like zeroing it"
    assert torch.equal(la, lr), "values under the hidden pixels must not reach the latents"
    return "latents identical whether hiding by mask bit only, zeroing, or after scrambling the hidden pixels' flux/ivar"


def test_loss_scores_only_hidden_pixels():
    torch.manual_seed(0)
    B, Q = 3, 500
    flux, ivar = torch.randn(B, Q), torch.rand(B, Q) * 20 + 1
    scale = torch.rand(B) + 0.5
    pred = torch.randn(B, Q, requires_grad=True)
    hid = torch.rand(B, Q) < 0.3
    loss = hidden_loss(pred, scale, flux, ivar, hid)
    r = (flux - scale[:, None] * pred) * ivar.sqrt()
    assert torch.allclose(loss, (r[hid] ** 2).mean())
    loss.backward()
    assert (pred.grad[~hid] == 0).all() and (pred.grad[hid] != 0).all()
    c = 2.0
    h = huber(r.detach(), c)
    small = r.detach().abs() <= c
    assert torch.allclose(h[small], r.detach()[small] ** 2) and (h[~small] < r.detach()[~small] ** 2).all()
    # chi2 of a perfect-model fit to noisy data is 1 on both hidden and visible pixels
    g = torch.Generator().manual_seed(1)
    truth = torch.randn(B, 20000, generator=g); iv = torch.rand(B, 20000, generator=g) * 50 + 5
    obs = truth + torch.randn(B, 20000, generator=g) / iv.sqrt()
    hid2 = torch.rand(B, 20000, generator=g) < 0.3
    st = chi2_stats(truth, torch.ones(B), obs, iv, hid2, torch.ones_like(hid2))
    assert abs(st["chi2_hidden"] - 1) < 0.03 and abs(st["chi2_visible"] - 1) < 0.03, st
    return f"loss = mean r^2 on hidden px, zero gradient on visible px; Huber matches r^2 inside c; chi2 of the true model on noisy data {st['chi2_hidden']:.3f}/{st['chi2_visible']:.3f}"


def test_zhead_trusted_mask():
    torch.manual_seed(0)
    head = ZHead(4, 8)
    c = torch.randn(5, 4, 8)
    z = torch.rand(5) * 2
    mask = torch.tensor([True, True, False, True, False])
    z2 = z.clone(); z2[~mask] = 99.0
    a, b = ZHead.loss(head(c), z, mask), ZHead.loss(head(c), z2, mask)
    assert torch.equal(a, b), "untrusted redshifts must not enter the loss"
    assert ZHead.loss(head(c), z, torch.zeros(5, dtype=torch.bool)) is None
    assert head(c).shape == (5,) and ZHead(4, 8, "pooled")(c).shape == (5,)
    return "untrusted z do not change the loss; no trusted z -> no loss; flat and pooled heads give (B,)"


def test_pcgrad_algebra():
    g = torch.Generator().manual_seed(0)
    rnd = lambda: [torch.randn(7, 3, generator=g, dtype=torch.float64), torch.randn(11, dtype=torch.float64)]
    for _ in range(20):
        gr, gz = rnd(), rnd()
        gz2, dot, nr2, nz2 = project_out_conflict(gz, gr)
        d2 = _dot(gz2, gr)
        if dot < 0:
            assert abs(d2) < 1e-10 * nr2.sqrt() * nz2.sqrt(), d2
            expect = nz2 - dot ** 2 / nr2
            assert abs(_dot(gz2, gz2) - expect) < 1e-9 * nz2
        else:
            assert all(torch.equal(a, b) for a, b in zip(gz, gz2)), "non-conflicting gradient must pass through unchanged"
    return "after surgery g_z' . g_rec = 0 (conflict) and |g_z'|^2 = |g_z|^2 - (g_z.g_rec)^2/|g_rec|^2; non-conflicting g_z unchanged"


def _tiny_setup(seed=0, hidden_seed=3):
    torch.manual_seed(seed)
    net = Eidos(**TINY).double()
    head = ZHead(net.n_latents, net.d_latent).double()
    x, zall = real_batch(4, torch.float64)
    good = ((x[:, 2] == 0) & (x[:, 1] > 0))
    hid = torch.from_numpy(SpanMasker().sample(good.numpy(), np.random.default_rng(hidden_seed))[0])
    return net, head, x, torch.tensor(zall), hid


def _losses(net, head, x, z, hid, qv, sign=1.0):
    lat = net.encode(apply_hidden(x, hid))
    pred = net.decoder(lat[:, 1:], qv)
    s = 10 ** lat[:, 0, 0]
    loss_rec = hidden_loss(pred, s, x[:, 0], x[:, 1], hid)
    loss_z = sign * ZHead.loss(head(lat[:, 1:]), z)
    return loss_rec, loss_z


def test_pcgrad_on_the_model():
    net, head, x, z, hid = _tiny_setup()
    qv = velocity(NATIVE, net.wave0)
    shared = [p for n, p in net.named_parameters() if n.startswith(("cnn", "perceiver"))]
    dec = list(net.decoder.parameters())
    hp = list(head.parameters())
    seen = {"conflict": 0, "free": 0}
    for sign in (1.0, -1.0):                                  # flipping the z loss flips the sign of g_rec . g_z
        rec0, z0 = _losses(net, head, x, z, hid, qv, sign)
        # references: each loss alone
        g_rec = torch.autograd.grad(rec0, shared + dec, retain_graph=True, allow_unused=True)
        g_z = torch.autograd.grad(z0, shared + hp, retain_graph=True, allow_unused=True)
        assert all(g is None for g in torch.autograd.grad(z0, dec, retain_graph=True, allow_unused=True)), \
            "the z loss must not reach the decoder"
        for surgery in (True, False):
            for p in shared + dec + hp:
                p.grad = None
            rec, lz = _losses(net, head, x, z, hid, qv, sign)
            st = backward_with_surgery(rec, lz, shared, dec, hp, surgery=surgery)
            n = len(shared)
            gr_sh = [torch.zeros_like(p) if g is None else g for p, g in zip(shared, g_rec[:n])]
            gz_sh = [torch.zeros_like(p) if g is None else g for p, g in zip(shared, g_z[:n])]
            final = [p.grad for p in shared]
            # decoder gets exactly g_rec, head exactly g_z
            for p, g in zip(dec, g_rec[n:]):
                assert torch.allclose(p.grad, torch.zeros_like(p) if g is None else g, atol=1e-12, rtol=1e-9)
            for p, g in zip(hp, g_z[n:]):
                assert torch.allclose(p.grad, g, atol=1e-12, rtol=1e-9)
            dot_rz = _dot(gr_sh, gz_sh); nr2 = _dot(gr_sh, gr_sh)
            along = _dot(final, gr_sh)
            if dot_rz < 0:
                seen["conflict"] += surgery
                if surgery:
                    assert abs(along - nr2) < 1e-9 * nr2, "after surgery the update has no net effect on g_rec beyond g_rec"
                    assert st["surgery"] == 1.0
                else:
                    assert along < nr2 * (1 - 1e-6), "the plain sum should lose reconstruction progress when they conflict"
            else:
                seen["free"] += surgery
                assert all(torch.allclose(f, a + b, atol=1e-12, rtol=1e-9) for f, a, b in zip(final, gr_sh, gz_sh))
                assert st["surgery"] == 0.0
            if surgery:
                assert along >= nr2 * (1 - 1e-9), "PCGrad guarantee: g_rec . g_update >= |g_rec|^2"
                # first-order check by finite differences of the reconstruction loss along the assembled update
                d = [p.grad.clone() for p in shared + dec]
                params = shared + dec
                pred_slope = _dot(g_rec_full := [torch.zeros_like(p) if g is None else g for p, g in zip(params, g_rec)], d)
                eta = 1e-6 * rec0.item() / pred_slope.item()
                with torch.no_grad():
                    for p, di in zip(params, d):
                        p.sub_(eta * di)
                    lat = net.encode(apply_hidden(x, hid)); s = 10 ** lat[:, 0, 0]
                    new = hidden_loss(net.decoder(lat[:, 1:], qv), s, x[:, 0], x[:, 1], hid)
                    for p, di in zip(params, d):
                        p.add_(eta * di)
                measured = (rec0 - new).item()
                assert abs(measured - eta * pred_slope.item()) < 2e-3 * abs(measured), (measured, eta * pred_slope.item())
    assert seen["conflict"] >= 1 and seen["free"] >= 1, seen
    return (f"decoder grad = g_rec, head grad = g_z, z loss never reaches the decoder; conflicting case: g_rec.update = |g_rec|^2 "
            f"(plain sum falls short), free case: update = g_rec + g_z; finite-difference slope of L_rec matches g_rec.update to 0.2%")


def test_default_eidos_unchanged_and_small_config_shapes():
    full = Eidos()
    n = {k: sum(p.numel() for p in m.parameters()) / 1e6 for k, m in (("cnn", full.cnn), ("perceiver", full.perceiver), ("decoder", full.decoder))}
    assert abs(n["cnn"] - 1.03) < 0.02 and abs(n["perceiver"] - 11.07) < 0.05 and abs(n["decoder"] - 0.33) < 0.02, n
    assert (full.n_latents, full.d_latent) == (64, 256)
    small = Eidos(**TINY).eval()
    x, _ = real_batch(2)
    with torch.no_grad():
        lat = small.encode(x)
        out = small.decode(lat, NATIVE)
    assert lat.shape == (2, 5, 36) and out.shape == (2, N_PIX) and torch.isfinite(out).all()
    return "Eidos() unchanged: params (M) " + ", ".join(f"{k} {v:.2f}" for k, v in n.items()) + f"; tiny config latents {tuple(lat.shape)}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
