"""Educational re-derivation of a Widar3.0 BVP from raw Intel-5300 CSI.

This module exists to *verify understanding* of where BVP comes from — it is not
on the modelling path (chunk 12 uses the official pre-computed .mat volumes). It
reimplements, in Python, the CSI → Doppler → velocity chain that the official
MATLAB extractor (``BVPExtractionCode/Widar3.0Release-Matlab``) runs, for the one
sample shipped with that toolkit (``Data/userA-1-1-1-1-r{1..6}.dat`` →
``BVP/user-user-1-1-1-1-1-...mat``), and compares the result against that file.

The math, following Widar3.0 (Zheng et al., MobiSys '19) §4:

1.  **CSI → per-receiver Doppler (DFS).** For each of the 6 receivers we read the
    raw 802.11n CSI (Linux CSI Tool log format), denoise it the way the
    reference code does (antenna selection, amplitude adjustment, conjugate
    multiplication against a reference antenna to cancel the carrier-frequency
    offset, band-pass to keep 2–60 Hz of motion), reduce the subcarriers with
    PCA, and take an STFT (``scipy.signal.stft``) to get energy vs. Doppler
    frequency over time — a ``(F=121, T)`` map per receiver, F covering
    [-60, +60] Hz.

2.  **Forward physics velocity → Doppler.** A body part moving at velocity
    ``v = (v_x, v_y)`` produces on receiver *i* the Doppler shift
    ``f_i = (1/λ) · a_i · v`` (Eq. 3 in the paper), where ``λ = c / 5.825 GHz``
    and ``a_i`` is the bistatic geometry vector
    ``a_i = (p - p_tx)/‖p - p_tx‖ + (p - p_rx,i)/‖p - p_rx,i‖`` (``get_A_matrix``).
    Discretising the 20×20 velocity grid gives a sparse 0/1 operator
    ``G`` (the paper's ``A_{ji}`` assignment / the code's ``VDM``) that maps a
    velocity distribution to the Doppler each receiver would see.

3.  **Inverse problem Doppler → velocity (the BVP).** We seek the non-negative
    20×20 velocity distribution ``P`` whose six predicted Doppler spectra match
    the six measured ones. The official extractor minimises an Earth-Mover's
    Distance with an L0 sparsity term under non-negativity, via ``fmincon`` SQP.
    That objective is numerically delicate; here we solve the **same linear
    system** ``G·vec(P) = d`` in the non-negative least-squares sense
    (``scipy.optimize.nnls``), which is stable and captures the essential
    inversion while deliberately dropping the EMD/sparsity refinement. The
    discrepancy this introduces is the point of the comparison — see
    ``docs/chunk11_bvp_pipeline.md``.

4.  **Body-frame rotation.** The velocity grid is rotated by the torso
    orientation so the representation is in the person's own frame
    (``get_rotated_spectrum`` → here ``scipy.ndimage.rotate``).

Run the comparison (saves ``figures/csi_to_bvp_check.png`` and prints metrics)::

    conda activate wifisense
    python -m src.data.csi_to_bvp
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import rotate as _nd_rotate
from scipy.optimize import nnls

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TOOLKIT = (
    PROJECT_ROOT
    / "data" / "raw" / "widar3" / "BVPExtractionCode" / "Widar3.0Release-Matlab"
)
SAMPLE_DATA_DIR = _TOOLKIT / "Data"
SAMPLE_PREFIX = "userA-1-1-1-1"  # the 6 receiver .dat files share this prefix
SAMPLE_OFFICIAL_BVP = _TOOLKIT / "BVP" / (
    "user-user-1-1-1-1-1-1e-07-100-20-100000-L0.mat"
)

# --- Physical / acquisition constants (verbatim from DVM_main.m) ------------
SAMP_RATE = 1000               # CSI packet rate (Hz)
WAVE_LENGTH = 299792458 / 5.825e9  # ~5.15 cm at 5.825 GHz
SEG_LENGTH = 100               # packets averaged per BVP frame → 10 Hz output
DOPPLER_HALF_BAND = 60         # keep Doppler in [-60, +60] Hz
FREQ_AXIS = np.arange(-DOPPLER_HALF_BAND, DOPPLER_HALF_BAND + 1)  # 121 bins
RX_CNT = 6
RX_ACNT = 3                    # antennas per receiver

V_MAX, V_BINS = 2.0, 20
# velocity_bin = ((1:M) - M/2)/(M/2)*V_max  →  [-1.8, -1.6, ..., 2.0]
VELOCITY_BIN = ((np.arange(1, V_BINS + 1)) - V_BINS / 2) / (V_BINS / 2) * V_MAX

# Device deployment (metres) and torso orientations (deg) from DVM_main.m.
TX_POS = np.array([0.0, 0.0])
RX_POS = np.array([
    [0.455, -0.455], [1.365, -0.455], [2.0, 0.0],      # Rx1-3
    [-0.455, 0.455], [-0.455, 1.365], [0.0, 2.0],      # Rx4-6
])
TORSO_POS = np.array([
    [1.365, 0.455], [0.455, 0.455], [0.455, 1.365], [1.365, 1.365],
    [0.91, 0.91], [2.275, 1.365], [2.275, 2.275], [1.365, 2.275],
])
TORSO_ORI = np.array([-90, -45, 0, 45, 90])


# ===========================================================================
# 1a. Raw CSI reader (Linux 802.11n CSI Tool .dat → complex array)
# ===========================================================================


def _dbinv(x: float) -> float:
    """Inverse of the dB (power) scale: 10^(x/10)."""
    return 10.0 ** (x / 10.0)


def _total_rss(rssi_a: int, rssi_b: int, rssi_c: int, agc: int) -> float:
    """Total received signal strength in dB (port of get_total_rss.m)."""
    mag = 0.0
    for r in (rssi_a, rssi_b, rssi_c):
        if r != 0:
            mag += _dbinv(r)
    return 10.0 * np.log10(mag) - 44.0 - agc


def _scaled_csi(csi: np.ndarray, rssi, agc, noise, nrx, ntx) -> np.ndarray:
    """Port of get_scaled_csi.m: scale raw CSI to units of sqrt(SNR).

    csi is (nrx, 30) complex (one Tx). Returns the same shape, scaled.
    """
    csi_pwr = np.sum(np.abs(csi) ** 2)
    rssi_pwr = _dbinv(_total_rss(*rssi, agc))
    scale = rssi_pwr / (csi_pwr / 30.0) if csi_pwr > 0 else 0.0
    noise_db = -92.0 if noise == -127 else float(noise)
    thermal = _dbinv(noise_db)
    quant = scale * nrx * ntx
    total = thermal + quant
    ret = csi * np.sqrt(scale / total) if total > 0 else csi
    if ntx == 2:
        ret = ret * np.sqrt(2)
    elif ntx == 3:
        ret = ret * np.sqrt(_dbinv(4.5))
    return ret


def _parse_bfee(payload: np.ndarray) -> dict | None:
    """Decode one beamforming record (port of read_bfee.c).

    Returns a dict with the scaled CSI as (nrx, 30) complex, or None if the
    record's length field is inconsistent (corrupt/partial packet).
    """
    nrx = int(payload[8])
    ntx = int(payload[9])
    rssi = (int(payload[10]), int(payload[11]), int(payload[12]))
    noise = int(np.int8(payload[13]))
    agc = int(payload[14])
    clen = int(payload[16]) + (int(payload[17]) << 8)
    calc_len = (30 * (nrx * ntx * 8 * 2 + 3) + 7) // 8
    if clen != calc_len:
        return None
    body = payload[20:]

    csi = np.zeros((ntx, nrx, 30), dtype=np.complex128)
    index = 0
    for sc in range(30):
        index += 3
        rem = index % 8
        for k in range(nrx * ntx):
            b = index // 8
            tmp_r = (int(body[b]) >> rem) | (int(body[b + 1]) << (8 - rem))
            tmp_i = (int(body[b + 1]) >> rem) | (int(body[b + 2]) << (8 - rem))
            real = (tmp_r & 0xFF) - 256 if (tmp_r & 0xFF) >= 128 else (tmp_r & 0xFF)
            imag = (tmp_i & 0xFF) - 256 if (tmp_i & 0xFF) >= 128 else (tmp_i & 0xFF)
            # C fills [ntx, nrx, 30] column-major: k runs tx-fastest then rx.
            tx, rx = k % ntx, k // ntx
            csi[tx, rx, sc] = real + 1j * imag
            index += 16

    scaled = _scaled_csi(csi[0], rssi, agc, noise, nrx, ntx)  # (nrx, 30)
    return {"csi": scaled, "nrx": nrx}


def csi_get_all(path: Path | str) -> np.ndarray:
    """Read a .dat capture into a (n_packets, 90) complex CSI array.

    Columns are antenna-major: ``[ant0 sc0..29, ant1 sc0..29, ant2 sc0..29]``,
    matching csi_get_all.m. Records that fail the length check are skipped.
    """
    raw = np.fromfile(str(path), dtype=np.uint8)
    rows: list[np.ndarray] = []
    cur, n = 0, len(raw)
    while cur < n - 3:
        field_len = (int(raw[cur]) << 8) | int(raw[cur + 1])  # big-endian
        code = raw[cur + 2]
        cur += 3
        if code == 187:  # beamforming record
            body = raw[cur : cur + field_len - 1]
            cur += field_len - 1
            if len(body) != field_len - 1:
                break
            rec = _parse_bfee(body)
            if rec is None or rec["nrx"] != RX_ACNT:
                continue
            # (nrx, 30) → flat 90, antenna-major
            rows.append(rec["csi"].reshape(-1))
        else:
            cur += field_len - 1
    if not rows:
        raise ValueError(f"no valid CSI records in {path}")
    return np.asarray(rows)


# ===========================================================================
# 1b. CSI → per-receiver Doppler spectrum (port of get_doppler_spectrum.m)
# ===========================================================================


def _doppler_one_receiver(csi_data: np.ndarray) -> np.ndarray:
    """One receiver's (90-col) CSI → Doppler spectrum (121, n_frames).

    Faithfully follows get_doppler_spectrum.m but substitutes scipy.signal.stft
    for the toolkit's TFTB ``tfrsp``. Each output column is L1-normalized over
    frequency, and frames are produced at ~10 Hz (one per SEG_LENGTH packets).
    """
    n = csi_data.shape[0]

    # Antenna selection (WiDance): pick the antenna with the largest mean/std.
    amp = np.abs(csi_data)
    ratio = amp.mean(0) / (amp.std(0) + 1e-12)         # (90,)
    idx = int(np.argmax(ratio.reshape(RX_ACNT, 30).mean(1)))  # 0..2
    ref = np.tile(csi_data[:, idx * 30 : (idx + 1) * 30], (1, RX_ACNT))

    # Amplitude adjustment (IndoTrack).
    adj = np.empty_like(csi_data)
    alphas = np.zeros(90)
    for jj in range(90):
        a = amp[:, jj]
        nz = a[a != 0]
        alpha = nz.min() if nz.size else 0.0
        alphas[jj] = alpha
        adj[:, jj] = (a - alpha) * np.exp(1j * np.angle(csi_data[:, jj]))
    beta = 1000.0 * alphas.sum() / 90.0
    ref_adj = (np.abs(ref) + beta) * np.exp(1j * np.angle(ref))

    # Conjugate multiplication, then drop the reference antenna's 30 columns.
    conj_mult = adj * np.conj(ref_adj)
    conj_mult = np.concatenate(
        [conj_mult[:, : 30 * idx], conj_mult[:, 30 * (idx + 1) :]], axis=1
    )  # (n, 60)

    # Band-pass: low-pass 60 Hz (order 6) then high-pass 2 Hz (order 3),
    # causal (lfilter) to match MATLAB ``filter``.
    lu, ld = signal.butter(6, 60 / (SAMP_RATE / 2), "low")
    hu, hd = signal.butter(3, 2 / (SAMP_RATE / 2), "high")
    conj_mult = signal.lfilter(lu, ld, conj_mult, axis=0)
    conj_mult = signal.lfilter(hu, hd, conj_mult, axis=0)

    # PCA over the 60 conjugate-multiplied streams → 1-D complex series.
    xc = conj_mult - conj_mult.mean(0, keepdims=True)
    cov = xc.conj().T @ xc
    _, vecs = np.linalg.eigh(cov)
    series = xc @ vecs[:, -1]                            # first PC

    # STFT: 1 Hz frequency bins (nfft = fs), Gaussian window, hop = SEG_LENGTH
    # so each column is one ~10 Hz BVP frame.
    win = signal.windows.gaussian(round(SAMP_RATE / 4) | 1, std=round(SAMP_RATE / 4) / 6)
    f, _, z = signal.stft(
        series, fs=SAMP_RATE, window=win, nperseg=win.size,
        noverlap=win.size - SEG_LENGTH, nfft=SAMP_RATE,
        return_onesided=False, boundary=None, padded=False,
    )
    mag = np.abs(z)
    # Reorder/select the integer Hz bins in [-60, 60].
    fr = np.round(f).astype(int)
    rows = [int(np.where(fr == target)[0][0]) for target in FREQ_AXIS]
    spec = mag[rows, :]                                  # (121, n_frames)
    spec /= spec.sum(0, keepdims=True) + 1e-12           # per-frame L1 norm
    return spec


def doppler_spectrum(prefix_path: Path | str) -> np.ndarray:
    """Stack all 6 receivers' Doppler spectra: returns (6, 121, T).

    ``prefix_path`` is the shared path prefix; receiver files are
    ``<prefix>-r{1..6}.dat``. Receivers with differing frame counts are
    truncated to the common minimum T.
    """
    specs = []
    for ii in range(1, RX_CNT + 1):
        csi = csi_get_all(f"{prefix_path}-r{ii}.dat")
        specs.append(_doppler_one_receiver(csi))
    T = min(s.shape[1] for s in specs)
    return np.stack([s[:, :T] for s in specs])           # (6, 121, T)


# ===========================================================================
# 2. Forward operator (velocity → Doppler) — paper Eq. 3 / VDM
# ===========================================================================


def get_A_matrix(torso_pos: np.ndarray) -> np.ndarray:
    """Bistatic geometry vectors a_i for the 6 links (port of get_A_matrix.m).

    Returns (6, 2): a_i = unit(p - p_tx) + unit(p - p_rx,i).
    """
    A = np.zeros((RX_CNT, 2))
    d_tx = np.linalg.norm(torso_pos - TX_POS)
    for ii in range(RX_CNT):
        d_rx = np.linalg.norm(torso_pos - RX_POS[ii])
        A[ii] = (torso_pos - TX_POS) / d_tx + (torso_pos - RX_POS[ii]) / d_rx
    return A


def build_doppler_operator(A: np.ndarray) -> np.ndarray:
    """0/1 operator G (6, 121, 400) mapping vec(P) → per-receiver Doppler.

    For link ii and velocity cell (i, j), the Doppler is
    ``f = round(a_i · [v_i, v_j] / λ)`` (the code's ``VDM``). Cells whose
    Doppler falls outside [-60, 60] Hz get an all-zero column (no contribution),
    matching the toolkit's out-of-band handling.
    """
    M = V_BINS
    G = np.zeros((RX_CNT, FREQ_AXIS.size, M * M))
    for ii in range(RX_CNT):
        for i in range(M):
            for j in range(M):
                f = int(round(A[ii] @ np.array([VELOCITY_BIN[i], VELOCITY_BIN[j]])
                              / WAVE_LENGTH))
                if -DOPPLER_HALF_BAND <= f <= DOPPLER_HALF_BAND:
                    G[ii, f + DOPPLER_HALF_BAND, i * M + j] = 1.0
    return G


# ===========================================================================
# 3. Inverse problem (Doppler → velocity) via non-negative least squares
# ===========================================================================


def _solve_frame(d: np.ndarray, G_stack: np.ndarray) -> np.ndarray:
    """Solve G·vec(P) = d for one frame, P ≥ 0, returning a 20×20 grid.

    d is (6, 121); receivers are path-loss-normalized to receiver 0's mass
    (as in DVM_main) before stacking into one linear system.
    """
    d = d.copy()
    ref_mass = d[0].sum()
    for jj in range(1, RX_CNT):
        m = d[jj].sum()
        if m > 0:
            d[jj] *= ref_mass / m
    p, _ = nnls(G_stack, d.reshape(-1))
    return p.reshape(V_BINS, V_BINS)


def derive_bvp(prefix_path: Path | str, position: int, orientation: int) -> np.ndarray:
    """Full CSI → BVP derivation for one sample. Returns (T, 20, 20) float32.

    Args:
        prefix_path: shared path prefix of the 6 ``-r{i}.dat`` files.
        position: torso-location id (1-8).
        orientation: face-orientation id (1-5).
    """
    spec = doppler_spectrum(prefix_path)                 # (6, 121, T)
    A = get_A_matrix(TORSO_POS[position - 1])
    G = build_doppler_operator(A)
    G_stack = G.reshape(RX_CNT * FREQ_AXIS.size, V_BINS * V_BINS)

    T = spec.shape[2]
    vel = np.stack([_solve_frame(spec[:, :, t], G_stack) for t in range(T)])
    # Body-frame rotation by torso orientation (deg). order=0 ≈ imrotate nearest.
    angle = float(TORSO_ORI[orientation - 1])
    vel_ro = _nd_rotate(vel, angle, axes=(1, 2), reshape=False, order=0)
    vel_ro = np.clip(vel_ro, 0.0, None)
    return vel_ro.astype(np.float32)


# ===========================================================================
# Comparison against the official .mat
# ===========================================================================


def _frame_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened, non-negative frames."""
    fa, fb = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    return float(fa @ fb / (na * nb)) if na > 0 and nb > 0 else 0.0


def compare(derived: np.ndarray, official: np.ndarray) -> dict:
    """Quantify agreement between derived and official BVP volumes.

    Both are (T, 20, 20) (T may differ; compared over the common length after
    per-frame L1 renormalization so the EMD/sparsity scale difference is
    factored out). Returns a dict of summary metrics.
    """
    def l1norm(v):
        s = v.reshape(v.shape[0], -1).sum(1, keepdims=True)
        return v / (s[..., None] + 1e-12)

    T = min(derived.shape[0], official.shape[0])
    d, o = l1norm(derived[:T]), l1norm(official[:T])
    cos = [_frame_cosine(d[t], o[t]) for t in range(T)]
    # Peak-velocity location error (in m/s) per frame.
    peak_err = []
    for t in range(T):
        di = np.unravel_index(np.argmax(d[t]), d[t].shape)
        oi = np.unravel_index(np.argmax(o[t]), o[t].shape)
        dv = np.array([VELOCITY_BIN[di[0]], VELOCITY_BIN[di[1]]])
        ov = np.array([VELOCITY_BIN[oi[0]], VELOCITY_BIN[oi[1]]])
        peak_err.append(float(np.linalg.norm(dv - ov)))
    return {
        "frames_derived": int(derived.shape[0]),
        "frames_official": int(official.shape[0]),
        "frames_compared": int(T),
        "mean_cosine": float(np.mean(cos)),
        "median_cosine": float(np.median(cos)),
        "mean_peak_velocity_error_mps": float(np.mean(peak_err)),
        "per_frame_cosine": [round(c, 3) for c in cos],
    }


def main() -> None:
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.io as sio

    if not SAMPLE_DATA_DIR.exists():
        raise FileNotFoundError(
            f"sample raw CSI not found at {SAMPLE_DATA_DIR}. It ships inside "
            "BVPExtractionCode.zip (see README.md / docs/chunk11_bvp_pipeline.md)."
        )

    prefix = SAMPLE_DATA_DIR / SAMPLE_PREFIX
    print(f"deriving BVP from {SAMPLE_PREFIX}-r{{1..6}}.dat ...")
    derived = derive_bvp(prefix, position=1, orientation=1)

    official = np.transpose(
        sio.loadmat(str(SAMPLE_OFFICIAL_BVP))["velocity_spectrum_ro"], (2, 0, 1)
    ).astype(np.float32)

    metrics = compare(derived, official)
    print(json.dumps(metrics, indent=2))

    # Side-by-side: time-aggregated derived vs official + a few frames.
    vmin, vmax = -V_MAX, V_MAX
    T = metrics["frames_compared"]
    cols = min(4, T)
    steps = np.linspace(0, T - 1, cols).round().astype(int)
    fig, axes = plt.subplots(2, cols + 1, figsize=((cols + 1) * 2.6, 5.4))
    for row, (vol, name) in enumerate([(derived, "derived"), (official, "official")]):
        agg = vol[:T].sum(0)
        axes[row, 0].imshow(agg.T, origin="lower", cmap="inferno",
                            extent=[vmin, vmax, vmin, vmax], aspect="equal")
        axes[row, 0].set_ylabel(f"{name}\n$v_y$ (m/s)", fontsize=9)
        axes[row, 0].set_title("Σ over time" if row == 0 else "", fontsize=9)
        for c, t in enumerate(steps):
            ax = axes[row, c + 1]
            ax.imshow(vol[t].T, origin="lower", cmap="inferno",
                      extent=[vmin, vmax, vmin, vmax], aspect="equal")
            if row == 0:
                ax.set_title(f"t={t}", fontsize=9)
            ax.set_xticklabels([]); ax.set_yticklabels([])
    fig.suptitle(
        "CSI→BVP re-derivation vs official (userA-1-1-1-1)  ·  "
        f"mean cosine={metrics['mean_cosine']:.2f}, "
        f"peak-vel err={metrics['mean_peak_velocity_error_mps']:.2f} m/s",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = PROJECT_ROOT / "figures" / "csi_to_bvp_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
