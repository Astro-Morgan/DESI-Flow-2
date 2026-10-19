"""
Checks for the grid Eidos (src/DESIFlow/eidos/{tokens,hybrid,grid_eidos}.py, training/{zhead,pcgrad}.py additions).
Run directly (python tests/test_eidos_grid.py) or with pytest. The equivalence test against the benchmark implementation (scripts/narrow_bench/phys) is skipped when
those untracked scripts are not present.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from DESIFlow.eidos.grid_eidos import EidosGrid
from DESIFlow.eidos.hybrid import HybridDecoder, RotAttn
from DESIFlow.eidos.tokens import TokenBlock
from DESIFlow.eidos.perceiver import C_KMS
from DESIFlow.training.pcgrad import route_grads, acc_grads, route_apply
from DESIFlow.training.zhead import ZClassHead
from DESIFlow.training.pretrain import SMALL

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
_NET = {}


def small_net(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _NET:
        torch.manual_seed(0)
        _NET[key] = EidosGrid(cnn_kwargs=SMALL["cnn_kwargs"], d=64, n_layers=2, n_heads=4, bott=8, **kw).to(DEV).eval()
    return _NET[key]


def synthetic_batch(n=3, seed=1):
    g = np.random.default_rng(seed)
    wave = np.linspace(3600., 9824., 7781)
    flux = np.stack([1 + 0.5 * np.sin(wave / (300 + 40 * i)) + 4 * np.exp(-0.5 * ((wave - 6563 * (1 + 0.1 * i)) / 3.0) ** 2) + 0.05 * g.standard_normal(7781) for i in range(n)])
    ivar = np.full((n, 7781), 1 / 0.05 ** 2)
    mask = np.zeros((n, 7781))
    mask[:, 1000:1040] = 1
    return torch.tensor(np.stack([flux, ivar, mask], 1), dtype=torch.float32, device=DEV)


def test_token_block_rope_is_relative():
    """Identical weights, all token positions shifted by a constant -> identical output (rotary depends on position DIFFERENCES only); rope has no position parameters."""
    tv = 370.44 * torch.arange(50)
    a, b = TokenBlock(64, 4, 50, pos="rope", token_v=tv).eval(), TokenBlock(64, 4, 50, pos="rope", token_v=tv + 40000.0).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 50, 64)
    with torch.no_grad():
        d = float((a(x) - b(x)).abs().max())
    assert d < 1e-4, d
    assert not any("rb" in n for n, _ in a.named_parameters())
    bias = TokenBlock(64, 4, 50, pos="bias").eval()
    assert bias.rb.shape == (4, 99) and bias(x).shape == x.shape
    cb = TokenBlock(64, 4, 50, pos="cbias").eval()
    assert TokenBlock(64, 4, 50).pos == "cbias", "cbias is the default position scheme"
    cb.load_state_dict(bias.state_dict(), strict=False)
    with torch.no_grad():
        d_init = float((cb(x) - bias(x)).abs().max())
    assert d_init == 0.0, d_init                                                                  # the zero-initialised g makes cbias start as the plain table
    try:
        TokenBlock(64, 4, 50, pos="rope")
        raise AssertionError("rope without token_v must be rejected")
    except ValueError:
        pass
    return f"rope output changes by {d:.1e} when all token positions are shifted by 40000 km/s; bias block table {tuple(bias.rb.shape)}; cbias (the default) starts identical to the table ({d_init:.1e})"


def test_cbias_content_conditioned_offsets():
    """pos='cbias': u[b, h, i, j] = g(x_i) . R[h, :, i - j + L - 1] (strided view == explicit gather, values and gradients); asymmetric in the signed offset; depends on the ASKING token's content
    and on offsets only (translation equivariant); gradients reach g, R and the table."""
    torch.manual_seed(2)
    L = 41
    blk = TokenBlock(64, 4, L, pos="cbias")
    with torch.no_grad():
        blk.g[2].weight.normal_(0, 0.3); blk.g[2].bias.normal_(0, 0.1)
    h = torch.randn(3, L, 64, requires_grad=True)
    u = blk.content_bias(h)
    g = blk.g(h).view(3, L, 4, blk.rank)
    ref = torch.einsum("blhr,hrlj->bhlj", g, blk.R[:, :, blk.idx])                              # the definition: R indexed by the table offset i - j
    w = torch.randn_like(ref)
    gr = torch.autograd.grad((ref * w).sum(), [h, blk.R], retain_graph=True)
    gu = torch.autograd.grad((u * w).sum(), [h, blk.R])
    d_val, d_gh, d_gR = float((u - ref).abs().max()), float((gr[0] - gu[0]).abs().max()), float((gr[1] - gu[1]).abs().max())
    assert d_val < 1e-5 and d_gh < 1e-4 and d_gR < 1e-4, (d_val, d_gh, d_gR)
    asym = float((u[0, 0, 10, 25] - u[0, 0, 25, 10]).abs())
    askers = [round(float(u[0, 0, a, a + 15]), 3) for a in (5, 10, 20)]
    assert asym > 1e-3 and len(set(askers)) == 3, (asym, askers)                                 # asymmetric, and different askers want the same offset differently
    h2 = torch.roll(h.detach(), 1, dims=1)
    u2 = blk.content_bias(h2)
    d_tr = float((u2[:, :, 11:30, 11:30] - u.detach()[:, :, 10:29, 10:29]).abs().max())
    assert d_tr < 1e-6, d_tr                                                                     # contents shifted by one token -> the preference matrix shifts with them
    x = torch.randn(2, L, 64, requires_grad=True)
    blk(x).square().mean().backward()
    gs = [float(p.grad.abs().sum()) for p in (blk.g[0].weight, blk.g[2].weight, blk.R, blk.rb)]
    assert all(v > 0 for v in gs), gs
    return f"strided view == gather definition (values {d_val:.0e}, grads {max(d_gh, d_gR):.0e}); u(10->25) vs u(25->10) differ by {asym:.2f}; same offset, three askers: {askers}; shift-equivariant ({d_tr:.0e}); grads reach g, R, table"


def test_hybrid_decoder_shapes_and_queries():
    dec = HybridDecoder(8, 813, 23.1, 13000, factor=2).to(DEV).eval()
    z = torch.randn(2, 813, 8, device=DEV)
    v = torch.linspace(0, 2.9e5, 500, device=DEV)
    with torch.no_grad():
        y = dec.render(z)
        full = dec(z, v)
        sub = dec(z, v[::7])
        per = dec(z, v.expand(2, -1))
    assert y.shape == (2, 13000) and full.shape == (2, 500) and per.shape == (2, 500) and torch.isfinite(y).all()
    assert float((full[:, ::7] - sub).abs().max()) < 1e-6 and float((full - per).abs().max()) < 1e-6
    for f in (1, 2, 4):
        assert HybridDecoder(8, 813, 23.1, 13000, factor=f).to(DEV)(z[:1], v[:10]).shape == (1, 10)
    assert HybridDecoder(1, 813, 23.1, 13000).to(DEV)(torch.randn(1, 813, 1, device=DEV), v[:10]).shape == (1, 10)         # width-1 latents are readable (lift before the norms)
    return f"{sum(p.numel() for p in dec.parameters()) / 1e6:.3f}M params; render (B, 13000); queries independent (subset / per-sample positions identical); factors 1/2/4 and width 1 work"


def test_eidos_grid_encode_decode():
    net = small_net()
    x = synthetic_batch()
    with torch.no_grad():
        lat = net.encode(x)
        s = 10 ** lat[:, 0, 0]
        q = torch.tensor(np.linspace(3600., 9824., 7781), dtype=torch.float64, device=DEV)
        y = net.decode(lat, q)
        y_sub = net.decode(lat, q[[5, 900, 4000]])
    assert lat.shape == (3, 814, 8) and y.shape == (3, 7781) and torch.isfinite(y).all()
    with torch.no_grad():
        s_cnn = net.cnn(*net.preprocessor(x))[1]
    assert torch.allclose(s, s_cnn, rtol=1e-5) and bool((s > 0).all()), (s, s_cnn)          # the scale token carries log10 of the CNN's scale s exactly
    assert float((y[:, [5, 900, 4000]] - y_sub).abs().max()) < 1e-5
    return f"latents {tuple(lat.shape)} = scale token + 813 x 8; s = {s.tolist()}; decode (B, 7781); a query subset reproduces the full render"


def test_z_feed_is_zero_init_and_gated():
    net = small_net()
    x = synthetic_batch()
    zr, zc = torch.tensor([0.1, 0.2, 0.3], device=DEV), torch.tensor([0.11, 0.19, 0.31], device=DEV)
    q = torch.tensor(np.linspace(3600., 9824., 7781), dtype=torch.float64, device=DEV)
    with torch.no_grad():
        lat = net.encode(x)
        y0 = net.decode(lat, q)
        y1 = net.decode(lat, q, z=(zr, zc), gate=1.0)
        assert float((y0 - y1).abs().max()) == 0.0                                            # zero-init feed: opening the gate does not disturb the decoder
        for m in (net.zpos[2], net.cond[2]):
            torch.nn.init.normal_(m.weight, std=0.05)
        y2 = net.decode(lat, q, z=(zr, zc), gate=1.0)
        y3 = net.decode(lat, q, z=(zr, zc), gate=0.0)
        for m in (net.zpos[2], net.cond[2]):
            torch.nn.init.zeros_(m.weight); torch.nn.init.zeros_(m.bias)
    assert float((y2 - y1).abs().max()) > 1e-4 and float((y3 - y0).abs().max()) == 0.0
    return "z feed: gate 1 == no feed at initialisation; non-zero weights change the output; gate 0 switches it off exactly"


def test_param_partition_and_gradients():
    net = small_net().train()
    shared, rec = net.partition_params()
    ids = lambda L: {id(p) for p in L}
    all_ids = {id(p) for p in net.parameters()}
    assert ids(shared) | ids(rec) == all_ids and not (ids(shared) & ids(rec))
    x = synthetic_batch()
    lat = net.encode(x)
    q = torch.tensor(np.linspace(3600., 9824., 7781), dtype=torch.float64, device=DEV)
    zr, zc = torch.zeros(3, device=DEV), torch.zeros(3, device=DEV)
    y = net.decode_native(lat, z=(zr, zc), gate=1.0)
    y_gen = net.decode(lat, q, z=(zr, zc), gate=1.0)
    d_y = float((y - y_gen).detach().abs().max())
    assert d_y < 1e-3, d_y            # dense-matrix fast path == gather-based interpolation (float32 position rounding)
    y.square().mean().backward()
    n_dec = sum(1 for p in rec if p.grad is not None and float(p.grad.abs().sum()) > 0)
    n_enc = sum(1 for p in shared if p.grad is not None and float(p.grad.abs().sum()) > 0)
    assert n_enc > 0.9 * len(shared) and n_dec > 0.5 * len(rec), (n_enc, len(shared), n_dec, len(rec))
    net.eval()
    return f"{len(shared)} shared + {len(rec)} decoder-side tensors cover all {len(all_ids)}; reconstruction gradient reaches {n_enc}/{len(shared)} encoder and {n_dec}/{len(rec)} decoder tensors"


def test_class_head():
    head = ZClassHead(813, 8)
    c = torch.randn(4, 813, 8)
    z = torch.tensor([0.3, 1.0, 2.0, 3.0])
    logits = head(c)
    loss = head.loss(logits, z)
    loss.backward()
    assert logits.shape == (4, 801) and torch.isfinite(loss) and head.predict(logits).shape == (4,)
    with torch.no_grad():                                                                       # a head that puts all mass on the right bin predicts it
        lg = torch.full((4, 801), -30.0)
        k = torch.round(torch.log1p(z) / 0.002).long()
        lg[torch.arange(4), k] = 30.0
        pred = head.predict(lg)
    assert float((pred - torch.log1p(z)).abs().max()) < 1e-3
    return f"logits {tuple(logits.shape)}, loss {loss.item():.2f}; a one-hot head returns ln(1+z) to {float((pred - torch.log1p(z)).abs().max()):.1e}"


def test_accumulated_pcgrad_equals_whole_batch():
    """route_grads summed over micro-batches then route_apply == the same on the whole batch (PCGrad needs the WHOLE-batch gradients)."""
    torch.manual_seed(3)
    enc, dec, head = torch.nn.Linear(6, 5), torch.nn.Linear(5, 1), torch.nn.Linear(5, 1)
    shared, rec, both = list(enc.parameters()), list(dec.parameters()), list(head.parameters())
    x, t, tz = torch.randn(8, 6), torch.randn(8, 1), torch.randn(8, 1)

    def losses(sl):
        h = torch.relu(enc(x[sl]))
        return ((dec(h) - t[sl]) ** 2).sum() / 8, ((head(h) - tz[sl]) ** 2).sum() / 8

    lr, lz = losses(slice(0, 8))
    g_rec, g_z = route_grads(lr, lz, shared, rec, both)
    stats = route_apply(g_rec, g_z, shared, rec, both, surgery=True, z_cap=True, z_scale=0.5)
    ref = [p.grad.clone() for p in shared + rec + both]
    acc_r = acc_z = None
    for sl in (slice(0, 4), slice(4, 8)):
        lr, lz = losses(sl)
        gr, gz = route_grads(lr, lz, shared, rec, both)
        acc_r, acc_z = acc_grads(acc_r, gr), acc_grads(acc_z, gz)
    route_apply(acc_r, acc_z, shared, rec, both, surgery=True, z_cap=True, z_scale=0.5)
    d = max(float((a - p.grad).abs().max()) for a, p in zip(ref, shared + rec + both))
    assert d < 1e-6, d
    return f"2 micro-batches == whole batch (max |diff| {d:.1e}); cos(g_rec, g_z) {stats['cos']:+.2f}, z_cap {stats['z_cap']:.2f}"


def test_matches_benchmark_implementation():
    """Same weights -> same latents and same reconstruction as the benchmark GridCNN (scripts/narrow_bench/phys/physgrid.py), for cbias (the default), rope and the learned bias; the position parameters and the z feed are randomised so every path is exercised."""
    bench_dir = ROOT / "scripts" / "narrow_bench" / "phys"
    if not (bench_dir / "physgrid.py").exists():
        return "SKIPPED (benchmark scripts not present)"
    sys.path.insert(0, str(bench_dir)); sys.path.insert(0, str(bench_dir.parent))
    try:
        import physgrid
    except Exception as e:                                                                       # untracked scripts may not import on this machine
        return f"SKIPPED (benchmark import failed: {e!r})"
    msgs = []
    for pos in ("cbias", "rope", "bias"):
        torch.manual_seed(7)
        bench = physgrid.GridCNN(64, 2, 4, 8, "both", True, "hybrid", False, hyb=dict(factor=2, dim=96, conv_norm=False), pos=pos).to(DEV).eval()
        net = EidosGrid(cnn_kwargs=SMALL["cnn_kwargs"], d=64, n_layers=2, n_heads=4, bott=8, pos=pos).to(DEV).eval()
        for mod in (bench.zpos[2], bench.cond[2]):                                               # non-trivial z feed (it is zero-initialised)
            torch.nn.init.normal_(mod.weight, std=0.05); torch.nn.init.normal_(mod.bias, std=0.05)
        for blk in bench.blocks:                                                                 # non-trivial position parameters (table zero-init; cbias g last layer zero-init)
            if pos in ("bias", "cbias"):
                torch.nn.init.normal_(blk.rb, std=0.5)
            if pos == "cbias":
                torch.nn.init.normal_(blk.g[2].weight, std=0.2); torch.nn.init.normal_(blk.g[2].bias, std=0.2)
        sd = {k.replace("hyb.", "decoder.", 1) if k.startswith("hyb.") else k: v for k, v in bench.state_dict().items() if not k.startswith(("reg.", "cls."))}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("scale_token") for k in missing), missing                      # only the scale token (parameter-free) has no counterpart in the benchmark model
        x = synthetic_batch()
        zr, zc = torch.tensor([0.1, 0.2, 0.3], device=DEV), torch.tensor([0.11, 0.19, 0.31], device=DEV)
        with torch.no_grad():
            lat_b, scale_b = bench._encode(x[:, 0], x[:, 1], x[:, 2])
            lat = net.encode(x)
            bench.z_reg, bench.z_cls, bench.gate = zr, zc, 1.0
            y_b = bench._hybrid_decode(lat_b)
            y = net.decode_native(lat, z=(zr, zc), gate=1.0)
        dl = float((lat[:, 1:] - lat_b).abs().max())
        dy = float((y - y_b).abs().max())
        assert dl < 1e-5 and dy < 1e-4 * max(1.0, float(y_b.abs().max())), (pos, dl, dy)
        msgs.append(f"{pos}: latents differ by {dl:.1e}, reconstruction by {dy:.1e}")
    return "; ".join(msgs)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
