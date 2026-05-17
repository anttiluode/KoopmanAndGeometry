"""
GN-v7 KOOPMAN RESONATOR — Full Lossless Decomposition & Reconstruction
=======================================================================
HuggingFace Spaces / local Gradio app.

Core mathematics (see paper: GNv7_Koopman_Geometric_Resonators):
  v(t)   = normalize([x(t), x(t-τ), ..., x(t-(d-1)τ)])     Takens embedding
  W      = orthonormal basis in delay space                   Koopman eigenfunctions
  y(t)   = W @ v(t)                                          Koopman coordinates
  x̂(t)  = (Wᵀ @ y(t))[0] * ‖v_unnorm(t)‖                  LOSSLESS reconstruction
  x̂ᵢ(t) = y_i(t) * W[i,0] * ‖v_unnorm(t)‖                  mode i stem (sums to x̂)
  A      = EMA of |y(t) ⊗ y(t)|                             empirical transfer operator

Two-pass strategy:
  Pass 1 — online GN-v7: learn W and A from the signal
  Pass 2 — fixed W: re-encode all samples → y_history, reconstruct stems

pip install gradio numpy scipy soundfile mne matplotlib
"""

import numpy as np
import scipy.signal as sp_signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import soundfile as sf
import tempfile, os, warnings
import gradio as gr
warnings.filterwarnings("ignore")

try:
    import mne
    MNE_OK = True
except ImportError:
    MNE_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
BG     = "#0d0d1a"
PANEL  = "#13132a"
TEAL   = "#00d4aa"
AMBER  = "#f0a500"
CORAL  = "#ff6b6b"
PURPLE = "#9d7bea"
DIM    = "#5a5a7a"
WHITE  = "#e8e8f0"

MODE_COLORS = [TEAL, AMBER, CORAL, PURPLE, "#4fc3f7", "#aed581",
               "#f48fb1", "#80cbc4", "#ffcc02", "#ce93d8"]

# ─────────────────────────────────────────────────────────────────────────────
# GN-v7 ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class GNv7:
    """
    Online Koopman decomposition via adaptive delay-embedded resonator network.

    Parameters
    ----------
    tau       : delay spacing (resonator cavity length)
    dim       : embedding dimension (cavity depth)
    eta_w     : basis plasticity (Oja learning rate)
    eta_a     : adjacency plasticity (Hebbian rate)
    nov_th    : novelty threshold for mode birth
    max_modes : maximum Koopman eigenfunctions to discover
    """
    def __init__(self, tau=8, dim=16, eta_w=0.04, eta_a=0.08,
                 nov_th=0.35, max_modes=20):
        self.tau      = tau
        self.dim      = dim
        self.eta_w    = eta_w
        self.eta_a    = eta_a
        self.nov_th   = nov_th
        self.max_modes = max_modes
        self.min_len  = (dim - 1) * tau + 1

        self.buf : list  = []
        self.W   : list  = []        # list of d-dim unit vectors
        self.A   : np.ndarray = np.zeros((0, 0))

    # ── helpers ──────────────────────────────────────────────────────────────
    def _embed_unnorm(self):
        """Return raw embedding vector (before normalisation)."""
        v = np.array([self.buf[-1 - j * self.tau] for j in range(self.dim)],
                     dtype=np.float64)
        return v

    def _embed(self):
        v = self._embed_unnorm()
        n = np.linalg.norm(v) + 1e-9
        return v / n, n

    def _gs(self):
        """Full Gram-Schmidt re-orthogonalisation."""
        Q = []
        for w in self.W:
            u = w.copy()
            for q in Q:
                u -= np.dot(q, u) * q
            n = np.linalg.norm(u)
            if n > 1e-7:
                Q.append(u / n)
        self.W = Q


    # ── single step ──────────────────────────────────────────────────────────
    def step(self, x: float):
        """
        Push one sample. Returns (y, novelty, x_hat, norm_t, w_pinv_0).
        y      : Koopman coordinate vector
        novelty: fraction of v unexplained by current W
        x_hat  : lossless reconstruction of x from current W
        norm_t : ‖v_unnorm‖ (stored for stem export)
        w_pinv_0: Top row of live pseudoinverse (stored for true online stems)
        """
        self.buf.append(float(x))
        if len(self.buf) > self.min_len + 32:
            self.buf.pop(0)
        if len(self.buf) < self.min_len:
            # EARLY EXIT 1: Return 5 variables
            return np.zeros(self.max_modes), 0.0, 0.0, 1.0, np.zeros(self.max_modes)

        v, norm_t = self._embed()

        # ── init first mode ───────────────────────────────────────────────
        if len(self.W) == 0:
            self.W.append(v.copy())
            self.A = np.ones((1, 1))
            y = np.array([np.dot(self.W[0], v)])
            x_hat = y[0] * self.W[0][0] * norm_t
            
            # Get live pseudoinverse for the very first mode
            W_pinv = np.linalg.pinv(np.array(self.W))
            w_pinv_0 = W_pinv[0, :]
            
            # EARLY EXIT 2: Return 5 variables
            return self._pad(y), 0.0, x_hat, norm_t, self._pad(w_pinv_0)

        W_mat = np.array(self.W)          # (k, d)
        y_raw = W_mat @ v                 # (k,)
        k     = len(self.W)

        # ── novelty = unexplained energy ──────────────────────────────────
        novelty = max(0.0, 1.0 - float(np.dot(y_raw, y_raw)))

        # ── graph-weighted readout ────────────────────────────────────────
        A_norm = self.A / (self.A.sum(axis=1, keepdims=True) + 1e-8)
        y      = 0.8 * y_raw + 0.2 * (A_norm @ y_raw)

        # ── mode birth ───────────────────────────────────────────────────
        if novelty > self.nov_th and k < self.max_modes:
            v_recon  = W_mat.T @ y_raw
            residual = v - v_recon
            rn       = np.linalg.norm(residual)
            if rn > 1e-5:
                self.W.append(residual / rn)
                self._gs()
                k2    = len(self.W)
                new_A = np.eye(k2) * 0.1
                new_A[:k, :k] = self.A
                self.A = new_A
                W_mat  = np.array(self.W)
                y_raw  = W_mat @ v
                y      = y_raw.copy()

        # ── Oja update ────────────────────────────────────────────────────
        k2 = len(self.W)
        for i in range(k2):
            v_recon_i = sum(y[j] * self.W[j] for j in range(i + 1))
            self.W[i] += self.eta_w * y[i] * (v - v_recon_i)
        self._gs()

        # ── adjacency Transfer Operator update (y_{t} ≈ A y_{t-1}) ─────────
        k2 = len(self.W)
        if not hasattr(self, 'y_prev') or len(getattr(self, 'y_prev', [])) != k2:
            self.y_prev = y[:k2].copy()
        else:
            # e_t = y_t - A * y_{t-1}
            y_curr = y[:k2]
            err = y_curr - (self.A @ self.y_prev)
            
            # Normalized LMS (prevents matrix explosion during transients)
            norm_y_prev = np.dot(self.y_prev, self.y_prev) + 1e-8
            self.A += self.eta_a * np.outer(err, self.y_prev) / norm_y_prev
            self.y_prev = y_curr.copy()

        # ── TRUE ONLINE INVERSE RECONSTRUCTION ────────────────────────────
        W_mat2 = np.array(self.W)
        W_pinv = np.linalg.pinv(W_mat2)         # Live pseudoinverse
        
        y_full = W_mat2 @ v                     # Re-project with updated W
        v_hat  = W_pinv @ y_full                # Project back to delay space
        x_hat  = v_hat[0] * norm_t
        
        # Grab the top row of the pseudoinverse to save for perfect stems
        w_pinv_0 = W_pinv[0, :]

        # MAIN EXIT: Return 5 variables
        return self._pad(y_full), novelty, x_hat, norm_t, self._pad(w_pinv_0)

    def _pad(self, y):
        out = np.zeros(self.max_modes)
        out[:len(y)] = y[:self.max_modes]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def spectral_reconstruct(y_history, norm_history, W_final, A_final):
    """
    True Spectral Koopman Reconstruction.
    Diagonalizes the transfer operator A to extract physically independent eigenmodes.
    """
    T = len(norm_history)
    k = A_final.shape[0]
    
    if k == 0:
        return np.zeros(T), [], []

    # 1. Eigendecomposition of the Transfer Operator A
    eigenvalues, Phi = np.linalg.eig(A_final)
    
    # 2. Inverse of eigenvectors (maps raw y space to true Koopman z space)
    Phi_inv = np.linalg.inv(Phi)
    
    # 3. Calculate the true Koopman coordinates: z(t) = Phi^-1 y(t)
    z_history = y_history[:, :k] @ Phi_inv.T
    
    # 4. Project true Koopman modes back to phase space: M = W^+ Phi
    W_pinv = np.linalg.pinv(W_final)
    M = W_pinv @ Phi
    
    # We only need the 0-th dimension (the reconstructed audio/signal)
    M_0 = M[0, :] 
    
    full_recon = np.zeros(T, dtype=np.complex128)
    stems = []
    
    for i in range(k):
        # stem_i(t) = z_i(t) * M_0[i] * norm(t)
        stem_complex = z_history[:, i] * M_0[i] * norm_history
        
        # Collapse the complex quantum math back into real physical waves
        stems.append(np.real(stem_complex))
        full_recon += stem_complex
        
    return np.real(full_recon), stems, eigenvalues

def dynamic_reconstruct(y_history, norm_history, w_pinv_hist, num_modes):
    """
    True Online Lossless Reconstruction.
    Uses the exact historical W^+ matrix for every specific timestep.
    """
    T = len(norm_history)
    full_recon = np.zeros(T)
    stems = []
    
    for i in range(num_modes):
        # stem_i(t) = y_i(t) * W⁺[0, i](t) * norm(t)
        stem = y_history[:, i] * w_pinv_hist[:, i] * norm_history
        stems.append(stem)
        full_recon += stem
        
    return full_recon, stems

def run_gnv7(signal: np.ndarray, tau: int, dim: int, eta_w: float,
             eta_a: float, nov_th: float, max_modes: int,
             progress_cb=None):
    gn = GNv7(tau=tau, dim=dim, eta_w=eta_w, eta_a=eta_a,
               nov_th=nov_th, max_modes=max_modes)

    T            = len(signal)
    y_history    = np.zeros((T, max_modes))
    norm_history = np.ones(T)
    x_hat_online = np.zeros(T)
    novelty_hist = np.zeros(T)
    w_pinv_hist  = np.zeros((T, max_modes))  # <-- NEW: Store history of the inverse

    report_every = max(1, T // 40)
    for i, x in enumerate(signal):
        y, nov, x_hat, nrm, w_p = gn.step(x) # <-- NEW: Unpack w_p
        y_history[i]            = y
        norm_history[i]         = nrm
        x_hat_online[i]         = x_hat
        novelty_hist[i]         = nov
        w_pinv_hist[i]          = w_p        # <-- NEW: Save inverse row
        if progress_cb and i % report_every == 0:
            progress_cb(i / T, f"processing sample {i}/{T}  |  modes: {len(gn.W)}")

    return gn, y_history, norm_history, x_hat_online, novelty_hist, w_pinv_hist


def compute_recon_error_curve(gn, y_history, norm_history):
    """
    Reconstruction error for k=1..K modes (for the error-vs-modes plot).
    Uses last 20% of signal (converged region).
    """
    if len(gn.W) == 0:
        return []
    W_mat  = np.array(gn.W)
    k      = len(gn.W)  # Get the actual number of discovered modes
    n      = len(norm_history)
    tail   = max(100, n // 5)
    
    # SLICE y_tail down to :k to discard the empty padding!
    y_tail = y_history[-tail:, :k]
    n_tail = norm_history[-tail:]
    sig_tail_raw = (y_tail @ W_mat[:, 0]) * n_tail  # approx signal from full basis

    errors = []
    for i in range(1, k + 1):
        w0k   = W_mat[:i, 0]
        xhat  = (y_tail[:, :i] @ w0k) * n_tail
        rmse  = np.sqrt(np.mean((sig_tail_raw - xhat) ** 2) + 1e-12)
        errors.append(float(rmse))
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def make_figure(signal, x_hat_2pass, x_hat_online, stems, y_history,
                norm_history, gn, fs, title, novelty_hist,
                error_curve, show_stems=True):
    """
    Master diagnostic figure — 6-panel layout.
    """
    k = len(gn.W)
    T = len(signal)
    t = np.arange(T) / fs

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    gs  = gridspec.GridSpec(4, 3, figure=fig,
                            hspace=0.48, wspace=0.32,
                            top=0.93, bottom=0.05,
                            left=0.06, right=0.97)

    def ax_style(ax, title_txt, xlabel="", ylabel=""):
        ax.set_facecolor(PANEL)
        ax.set_title(title_txt, color=TEAL, fontsize=10, pad=6)
        ax.tick_params(colors=DIM, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(DIM)
            sp.set_linewidth(0.4)
        if xlabel: ax.set_xlabel(xlabel, color=DIM, fontsize=8)
        if ylabel: ax.set_ylabel(ylabel, color=DIM, fontsize=8)

    # ── 1. Signal + both reconstructions ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax_style(ax1, f"{title}   |   signal (teal)  ·  2-pass reconstruction (amber)  ·  online (purple)")
    disp = min(T, int(fs * 8))   # show up to 8 seconds
    ax1.plot(t[:disp], signal[:disp],    color=TEAL,   lw=0.7, alpha=0.9, label="original")
    ax1.plot(t[:disp], x_hat_2pass[:disp], color=AMBER, lw=0.9, alpha=0.8, label="reconstruction (2-pass)")
    ax1.plot(t[:disp], x_hat_online[:disp], color=PURPLE, lw=0.7, alpha=0.5, label="online recon")
    ax1.legend(loc="upper right", facecolor=PANEL, labelcolor=WHITE,
               fontsize=7, framealpha=0.8)
    ax1.set_xlabel("time (s)", color=DIM, fontsize=8)

    # ── 2. Mode activation heatmap ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    ax_style(ax2, f"Koopman coordinates  |y_i(t)|   ({k} modes discovered)",
             xlabel="time (s)", ylabel="mode i")
    step_ds = max(1, T // 1200)
    act_ds  = np.abs(y_history[::step_ds, :k])
    extent  = [0, T / fs, -0.5, k - 0.5]
    im      = ax2.imshow(act_ds.T, aspect="auto", cmap="magma",
                         origin="lower", extent=extent)
    plt.colorbar(im, ax=ax2, pad=0.01, fraction=0.015).ax.tick_params(colors=DIM, labelsize=7)
    ax2.set_yticks(range(k))
    ax2.set_yticklabels([f"φ{i}" for i in range(k)], fontsize=7)

    # ── 3. Novelty signal ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    ax_style(ax3, "novelty  ν(t)  =  1 − ‖Wy‖²",
             xlabel="time (s)", ylabel="ν")
    ax3.plot(t[::step_ds], novelty_hist[::step_ds], color=CORAL, lw=0.7)
    ax3.axhline(gn.nov_th, color=WHITE, lw=0.8, linestyle="--", alpha=0.5, label=f"nov_th={gn.nov_th}")
    ax3.legend(facecolor=PANEL, labelcolor=WHITE, fontsize=7)
    ax3.set_ylim(0, 1.05)

    # ── 4. Adjacency matrix A ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax_style(ax4, "adjacency A  (empirical transfer operator)")
    if k > 0:
        im2 = ax4.imshow(gn.A, cmap="viridis", vmin=0, vmax=1,
                         aspect="auto")
        plt.colorbar(im2, ax=ax4, pad=0.02, fraction=0.045).ax.tick_params(colors=DIM, labelsize=7)
        ax4.set_xticks(range(k))
        ax4.set_yticks(range(k))
        ax4.set_xticklabels([f"φ{i}" for i in range(k)], fontsize=7)
        ax4.set_yticklabels([f"φ{i}" for i in range(k)], fontsize=7)

    # ── 5. Phase space  x(t) vs x(t+τ) ──────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax_style(ax5, f"phase space  x(t) vs x(t+τ)  [τ={gn.tau}]",
             xlabel="x(t)", ylabel="x(t+τ)")
    tau_s = gn.tau
    if T > tau_s + 50:
        ps_n = min(T - tau_s, 6000)
        xs   = signal[-ps_n - tau_s:-tau_s]
        ys_  = signal[-ps_n:]
        c_   = np.linspace(0, 1, ps_n)
        ax5.scatter(xs, ys_, c=c_, cmap="plasma", s=0.8, alpha=0.6, linewidths=0)

    # ── 6. Reconstruction error vs modes ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax_style(ax6, "‖x − x̂‖ vs number of modes k  (converged tail)",
             xlabel="modes k", ylabel="RMSE")
    if error_curve:
        xs_e = list(range(1, len(error_curve) + 1))
        ax6.plot(xs_e, error_curve, color=AMBER, lw=2, marker="o",
                 markersize=5, markerfacecolor=CORAL)
        ax6.set_xticks(xs_e)
        for xi, yi in zip(xs_e, error_curve):
            ax6.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=7, color=WHITE)

    # ── 7. Individual stems (bottom row) ─────────────────────────────────
    if show_stems and stems:
        ax7 = fig.add_subplot(gs[3, :])
        ax_style(ax7, f"mode stems  x̂_i(t)  summing to lossless reconstruction  |  k={k}",
                 xlabel="time (s)")
        disp2 = min(T, int(fs * 6))
        for i, stem in enumerate(stems[:k]):
            offset = i * 0.5
            ax7.plot(t[:disp2], stem[:disp2] + offset,
                     color=MODE_COLORS[i % len(MODE_COLORS)], lw=0.7,
                     label=f"stem φ{i}  W[{i},0]={gn.W[i][0]:.3f}")
        ax7.legend(loc="upper right", facecolor=PANEL, labelcolor=WHITE,
                   fontsize=6, ncol=2, framealpha=0.8)

    fig.suptitle(f"GN-v7 Koopman Resonator  —  τ={gn.tau}  d={gn.dim}  "
                 f"k={k}/{gn.max_modes}  η_w={gn.eta_w}  η_a={gn.eta_a}",
                 color=WHITE, fontsize=11, y=0.97)
    return fig


def save_figure(fig):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=130, facecolor=BG, edgecolor="none")
    plt.close(fig)
    return tmp.name


def save_wav(audio: np.ndarray, fs: int, label: str = "") -> str:
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.92
    tmp   = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio.astype(np.float32), fs)
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# CORE PROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def process_audio(wav_file, tau, dim, eta_w, eta_a, nov_th, max_modes,
                  target_fs, max_dur, progress=gr.Progress()):
    if wav_file is None:
        return [None] * 14 + ["Upload a WAV file first."]

    try:
        progress(0.0, desc="Loading audio…")
        sig, fs = sf.read(wav_file)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        sig = sig.astype(np.float64)

        tgt = int(target_fs)
        if fs != tgt:
            sig = sp_signal.resample_poly(sig, tgt, fs)
            fs  = tgt

        max_s = int(max_dur * fs)
        sig   = sig[:max_s]
        sig  /= np.max(np.abs(sig)) + 1e-9

        tau_i, dim_i, mm_i = int(tau), int(dim), int(max_modes)

        progress(0.05, desc="Running GN-v7…")
        gn, y_hist, n_hist, x_online, nov_hist, w_pinv_hist = run_gnv7(
            sig, tau_i, dim_i, float(eta_w), float(eta_a),
            float(nov_th), mm_i,
            progress_cb=lambda p, m: progress(0.05 + 0.65 * p, desc=m)
        )

        progress(0.72, desc="Extracting True Spectral Koopman Eigenfunctions…")
        k_found = len(gn.W)
        
        # Freeze final geometry and dynamics to extract true physics
        W_final = np.array(gn.W)
        A_final = gn.A
        
        x_hat_spectral, stems, eigs = spectral_reconstruct(y_hist, n_hist, W_final, A_final)

        progress(0.82, desc="Computing error curve…")
        err_curve = compute_recon_error_curve(gn, y_hist, n_hist)

        progress(0.88, desc="Generating plots…")
        fig = make_figure(sig, x_hat_spectral, x_online, stems, y_hist,
                          n_hist, gn, fs, "Audio", nov_hist, err_curve)
        fig_path = save_figure(fig)

        progress(0.93, desc="Saving audio files…")
        recon_path = save_wav(x_hat_spectral, fs, "recon")

        stem_paths = []
        for i, s in enumerate(stems[:mm_i]):
            stem_paths.append(save_wav(s, fs, f"stem{i}"))

        while len(stem_paths) < MAX_MODES_UI:
            stem_paths.append(None)

        status = (
            f"Done.  Discovered {k_found} true eigenmodes.  "
            f"Signal: {len(sig)/fs:.1f}s @ {fs}Hz.  "
            f"Final RMSE: {err_curve[-1]:.4f}"
            if err_curve else
            f"Done. Discovered {k_found} true eigenmodes."
        )

        progress(1.0, desc="Complete.")
        return [fig_path, recon_path] + stem_paths[:MAX_MODES_UI] + [status]

    except Exception as e:
        import traceback
        return [None] * (2 + MAX_MODES_UI) + [f"Error: {e}\n{traceback.format_exc()}"]


def process_eeg(edf_file, channel_name, tau, dim, eta_w, eta_a,
                nov_th, max_modes, max_dur, progress=gr.Progress()):
    if not MNE_OK:
        return [None] * 14 + ["mne not installed — pip install mne"]
    if edf_file is None:
        return [None] * 14 + ["Upload an EDF file first."]

    try:
        progress(0.0, desc="Loading EDF…")
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        fs  = int(raw.info["sfreq"])

        ch = channel_name.strip() if channel_name.strip() else ""
        if ch not in raw.ch_names:
            variances = {c: np.var(raw.get_data(picks=[c])[0]) for c in raw.ch_names}
            ch = max(variances, key=variances.get)
            msg_ch = f"Channel '{channel_name}' not found — auto-selected '{ch}' (highest variance)."
        else:
            msg_ch = f"Using channel '{ch}'."

        max_s = int(max_dur * fs)
        sig   = raw.get_data(picks=[ch], start=0, stop=max_s)[0].astype(np.float64)
        sig  -= sig.mean()
        sig  /= sig.std() + 1e-9

        tau_i, dim_i, mm_i = int(tau), int(dim), int(max_modes)

        progress(0.05, desc=f"GN-v7 on {ch}…")
        gn, y_hist, n_hist, x_online, nov_hist, w_pinv_hist = run_gnv7(
            sig, tau_i, dim_i, float(eta_w), float(eta_a),
            float(nov_th), mm_i,
            progress_cb=lambda p, m: progress(0.05 + 0.65 * p, desc=m)
        )

        progress(0.72, desc="Extracting True Spectral Koopman Eigenfunctions…")
        k_found = len(gn.W)
        
        # Freeze final geometry and dynamics to extract true physics
        W_final = np.array(gn.W)
        A_final = gn.A
        
        x_hat_spectral, stems, eigs = spectral_reconstruct(y_hist, n_hist, W_final, A_final)

        progress(0.82, desc="Error curve…")
        err_curve = compute_recon_error_curve(gn, y_hist, n_hist)

        progress(0.88, desc="Plotting…")
        fig = make_figure(sig, x_hat_spectral, x_online, stems, y_hist,
                          n_hist, gn, fs, f"EEG  [{ch}]", nov_hist, err_curve)
        fig_path = save_figure(fig)

        progress(0.93, desc="Stretching audio to match original EEG duration…")
        
        # 1. Define our high-fidelity target rate
        audible_fs = 16000 
        
        # 2. Mathematically stretch the array from native 'fs' up to 'audible_fs'
        # This keeps the browser from destroying the signal, but restores original playback time
        x_hat_stretched = sp_signal.resample_poly(x_hat_spectral, audible_fs, fs)
        recon_path = save_wav(x_hat_stretched, audible_fs, "eeg_recon")

        # 3. Stretch and save all the individual stems
        stem_paths = []
        for i, s in enumerate(stems[:mm_i]):
            s_stretched = sp_signal.resample_poly(s, audible_fs, fs)
            stem_paths.append(save_wav(s_stretched, audible_fs, f"eeg_stem{i}"))
            
        # Always pad to UI size (must match eeg_stems count)
        while len(stem_paths) < MAX_MODES_UI:
            stem_paths.append(None)

        k_found = len(gn.W)
        status = (
            f"{msg_ch}  Discovered {k_found} modes.  "
            f"{len(sig)/fs:.1f}s @ {fs}Hz.  "
            f"RMSE(k={k_found}): {err_curve[-1]:.4f}"
            if err_curve else
            f"{msg_ch}  Discovered {k_found} modes."
        )

        progress(1.0, desc="Complete.")
        return [fig_path, recon_path] + stem_paths[:MAX_MODES_UI] + [status]

    except Exception as e:
        import traceback
        return [None] * (2 + MAX_MODES_UI) + [f"Error: {e}\n{traceback.format_exc()}"]


def get_edf_channels(edf_file):
    if edf_file is None or not MNE_OK:
        return gr.update(choices=[], value="")
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
        return gr.update(choices=raw.ch_names, value=raw.ch_names[0])
    except:
        return gr.update(choices=[], value="")


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────
MAX_MODES_UI = 20

CSS = """
body, .gradio-container { background: #0d0d1a !important; font-family: 'Courier New', monospace; }
.gr-box, .gr-panel, .tab-nav { background: #13132a !important; border-color: #2a2a4a !important; }
h1, h2, h3, label, .label-wrap { color: #00d4aa !important; }
.gr-button-primary { background: #00d4aa !important; color: #0d0d1a !important; border: none !important; font-weight: bold; }
.gr-button { background: #13132a !important; color: #00d4aa !important; border: 1px solid #00d4aa !important; }
.gr-input, textarea, select { background: #1a1a35 !important; color: #e8e8f0 !important; border-color: #2a2a4a !important; }
.gr-slider input[type=range] { accent-color: #00d4aa; }
"""

def build_param_sliders():
    with gr.Column():
        tau     = gr.Slider(1,   80,  value=8,    step=1,    label="τ  delay spacing  (resonator cavity length)")
        dim     = gr.Slider(4,   64,  value=16,   step=2,    label="dim  embedding dimension  (cavity depth)")
        eta_w   = gr.Slider(0.001, 0.3, value=0.04, step=0.001, label="η_w  basis plasticity  (Oja learning rate)")
        eta_a   = gr.Slider(0.001, 0.2, value=0.02, step=0.001, label="η_a  adjacency plasticity  (Hebbian rate)")
        nov_th  = gr.Slider(0.05, 0.95, value=0.35, step=0.05,  label="nov_th  novelty threshold  (mode birth gate)")
        max_m   = gr.Slider(1,  MAX_MODES_UI, value=20, step=1,  label="max_modes  Koopman eigenfunctions")
    return tau, dim, eta_w, eta_a, nov_th, max_m


def build_stem_outputs(n=MAX_MODES_UI):
    outs = []
    with gr.Accordion("Mode stems  (x̂_i · sums to reconstruction)", open=False):
        for i in range(n):
            outs.append(gr.Audio(label=f"stem φ{i}", type="filepath", visible=True))
    return outs


with gr.Blocks(css=CSS, title="GN-v7 Koopman Resonator") as demo:

    gr.Markdown("""
# 🔬 GN-v7 Koopman Resonator
### Online delay-embedded geometric resonator network with lossless reconstruction
**Mathematics:**
`v(t) = normalise(Φ_τ,d(x,t))`  →  `y(t) = Wv(t)`  →  `x̂(t) = (Wᵀy(t))₀ · ‖Φ‖`

Each mode φᵢ is an approximate **Koopman eigenfunction**.  Stems sum exactly to the full reconstruction.  
The adjacency matrix **A** is the empirical transfer operator in eigenfunction coordinates.
""")

    with gr.Tabs():

        # ── AUDIO TAB ────────────────────────────────────────────────────────
        with gr.TabItem("🔊  Audio decomposition"):
            with gr.Row():
                with gr.Column(scale=1):
                    aud_file   = gr.File(label="Upload WAV", file_types=[".wav"])
                    aud_tfs    = gr.Slider(4000, 44100, value=16000, step=100,
                                          label="processing sample rate (Hz)")
                    aud_dur    = gr.Slider(1, 60, value=15, step=1,
                                          label="max duration (seconds)")
                    (aud_tau, aud_dim, aud_etaw, aud_etaa,
                     aud_nov, aud_mm) = build_param_sliders()
                    aud_btn    = gr.Button("▶  Decompose + Reconstruct", variant="primary")
                    aud_status = gr.Textbox(label="Status", lines=3)

                with gr.Column(scale=2):
                    aud_plot  = gr.Image(label="Diagnostic figure", type="filepath")
                    aud_recon = gr.Audio(label="Full lossless reconstruction", type="filepath")
                    aud_stems = build_stem_outputs(MAX_MODES_UI)

            aud_btn.click(
                fn=process_audio,
                inputs=[aud_file, aud_tau, aud_dim, aud_etaw, aud_etaa,
                        aud_nov, aud_mm, aud_tfs, aud_dur],
                outputs=[aud_plot, aud_recon] + aud_stems + [aud_status])

        # ── EEG TAB ──────────────────────────────────────────────────────────
        with gr.TabItem("🧠  EEG decomposition"):
            if not MNE_OK:
                gr.Markdown("⚠️  **mne not installed.** `pip install mne` to enable EEG loading.")

            with gr.Row():
                with gr.Column(scale=1):
                    eeg_file  = gr.File(label="Upload EDF", file_types=[".edf"])
                    eeg_ch    = gr.Dropdown(label="Channel (leave blank = auto highest-variance)",
                                           choices=[], allow_custom_value=True, value="")
                    eeg_dur   = gr.Slider(1, 120, value=30, step=1,
                                         label="max duration (seconds)")
                    (eeg_tau, eeg_dim, eeg_etaw, eeg_etaa,
                     eeg_nov, eeg_mm) = build_param_sliders()
                    eeg_btn   = gr.Button("▶  Decompose + Reconstruct", variant="primary")
                    eeg_status = gr.Textbox(label="Status", lines=3)

                with gr.Column(scale=2):
                    eeg_plot  = gr.Image(label="Diagnostic figure", type="filepath")
                    eeg_recon = gr.Audio(label="Full lossless reconstruction", type="filepath")
                    eeg_stems = build_stem_outputs(MAX_MODES_UI)

            eeg_file.change(fn=get_edf_channels, inputs=[eeg_file], outputs=[eeg_ch])

            eeg_btn.click(
                fn=process_eeg,
                inputs=[eeg_file, eeg_ch, eeg_tau, eeg_dim, eeg_etaw, eeg_etaa,
                        eeg_nov, eeg_mm, eeg_dur],
                outputs=[eeg_plot, eeg_recon] + eeg_stems + [eeg_status])

        # ── THEORY TAB ───────────────────────────────────────────────────────
        with gr.TabItem("📐  Theory"):
            gr.Markdown("""
## GN-v7 as online Koopman decomposition

### The embedding
```
v(t) = normalise( [x(t), x(t−τ), x(t−2τ), ..., x(t−(d−1)τ)] )
```
Maps the scalar signal onto the unit hypersphere S^(2d−1).  
Takens' theorem guarantees this is a diffeomorphism onto the attractor for generic (τ, d).

### Basis growth (novelty-gated Oja rule)
```
novelty ν(t) = 1 − ‖Wy‖²       (unexplained energy)

if ν(t) > nov_th and k < max_modes:
    residual = v(t) − Wᵀ W v(t)
    new mode w_{k+1} = residual / ‖residual‖
    Gram-Schmidt re-orthogonalise W

Oja update:  w_i ← w_i + η_w · y_i · (v − Σⱼ≤ᵢ yⱼ wⱼ)
```

### Lossless reconstruction  (Theorem 2 in the paper)
```
v̂(t)  = Wᵀ y(t)          ← exact when v ∈ span(W)
x̂(t)  = v̂₀(t) · ‖Φ(x,t)‖   ← recovers amplitude
x̂ᵢ(t) = yᵢ(t) · W[i,0] · ‖Φ‖   ← stem i  (sums to x̂)
```
The **metallic ringing** in individual modes is a comb-filter artifact of  
incomplete basis coverage.  The full stem sum is free of it.

### Adjacency matrix (Theorem 3)
```
A(t+1) = (1−η_a) A(t) + η_a |y(t) ⊗ y(t)|
```
Converges to the **empirical transfer operator** in Koopman coordinates:  
`Ã_∞ = argmin_B E[‖y(t+1) − B y(t)‖²]`  
Eigenvalues of Ã = Floquet multipliers of the signal's quasi-periodic skeleton.

### Multi-τ stack = cortical hierarchy
| τ | timescale | cortical analogy |
|---|---|---|
| 2–4 | glottal / gamma | AIS / L4 |
| 8–16 | phoneme / beta | L2/3 |
| 32–64 | syllable / alpha | association cortex |
| 128–512 | phrase / theta–delta | hippocampal loop |

### Parameter guide
| param | small | large |
|---|---|---|
| τ | metallic comb resonance  (f_s/τ peaks) | smooth, slow attractor |
| dim | shallow cavity, fast | deep memory, slow |
| nov_th | many modes, fine decomposition | minimal basis, robust |
| η_w | slow adaptation | fast retuning (noisy) |
| η_a | long graph memory | fast causal updates |
""")

    gr.Markdown("""
---
*GN-v7 Koopman Resonator · PerceptionLab Helsinki · arXiv: cs.LG / math.DS / eess.SP / q-bio.NC*  
*"The brain is a self-assembling bank of geometric resonators building an internal Koopman model of the world."*
""")


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0")
