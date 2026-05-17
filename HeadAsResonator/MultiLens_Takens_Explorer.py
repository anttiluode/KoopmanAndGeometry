#!/usr/bin/env python3
"""
MULTI-LENS TAKENS EXPLORER
==========================
Chain multiple 3D Takens views with controllable delay spacing (linear or log).

One electrode → many simultaneous "lenses" into different biological timescales.

Features:
- Choose number of lenses (1–12)
- Max delay (samples)
- Spacing: Linear or Log
- Auto-generates delay values
- Each lens shows its own 3D attractor + geometry label
- Shared playback + H(f) + EQ
- Safety limit + adaptive point count

This is the "multi-singularity" probe: one wire, many realities.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
from scipy import signal
from collections import deque
import os

# Optional imports
try:
    import mne
    MNE_OK = True
except ImportError:
    MNE_OK = False

try:
    import sounddevice as sd
    SD_OK = True
except ImportError:
    SD_OK = False

# Theme
BG      = "#0f0f23"
PANEL   = "#16213e"
ACCENT  = "#00ffcc"
WARN    = "#ffaa00"
DIM     = "#888888"
DARK2   = "#1a1a2e"

# =============================================================================
# GRAPHICAL EQ (same as before)
# =============================================================================
class GraphicalEQ(tk.Frame):
    BAND_LABELS = ["Heart","Breath","Theta","Alpha","Beta","Gamma",
                   "Voice","Hi1","Hi2","Hi3","Hi4","Hi5"]

    def __init__(self, parent, num_bands=12, width=310, height=55, callback=None):
        super().__init__(parent, bg=DARK2)
        self.width = width
        self.height = height
        self.num_bands = num_bands
        self.callback = callback
        self.gains = np.array([0.05,0.15,0.35,0.65,0.9,1.0,1.0,0.85,0.6,0.35,0.15,0.05], dtype=float)
        self.selected = None
        self.band_w = width / num_bands

        self.cv = tk.Canvas(self, width=width, height=height, bg=BG, highlightthickness=0)
        self.cv.pack(padx=3, pady=2)
        self.cv.bind("<ButtonPress-1>", self._click)
        self.cv.bind("<B1-Motion>", self._drag)
        self.cv.bind("<ButtonRelease-1>", self._release)
        self._draw()

    def _draw(self):
        self.cv.delete("all")
        for i in range(4):
            y = i * self.height // 3
            self.cv.create_line(0, y, self.width, y, fill=PANEL, width=1)
        pts = [0, self.height]
        xs = np.linspace(0, self.width, self.width)
        bc = (np.arange(self.num_bands) + .5) * self.band_w
        ig = np.interp(xs, bc, self.gains)
        for x, g in zip(xs, ig):
            pts.extend([x, (1-g)*self.height])
        pts.extend([self.width, self.height])
        self.cv.create_polygon(pts, fill="#1a3a6e", outline=ACCENT, width=1)
        for i, g in enumerate(self.gains):
            x = (i+.5)*self.band_w
            y = (1-g)*self.height
            col = WARN if g > .7 else ACCENT
            self.cv.create_oval(x-3, y-3, x+3, y+3, fill=col, outline="white", width=1)
        for i, lab in enumerate(self.BAND_LABELS[:self.num_bands]):
            x = (i+.5)*self.band_w
            self.cv.create_text(x, self.height-5, text=lab, fill=DIM, font=("Consolas",5))

    def _click(self, e):
        bi = int(e.x // self.band_w)
        if 0 <= bi < self.num_bands:
            self.selected = bi
            self._set(e.y)

    def _drag(self, e):
        if self.selected is not None:
            self._set(e.y)

    def _release(self, e):
        self.selected = None
        if self.callback: self.callback()

    def _set(self, y):
        self.gains[self.selected] = max(0.0, min(1.0, 1.0 - y/self.height))
        self._draw()
        if self.callback: self.callback()

    def get_curve(self, n=512):
        xs = np.linspace(0, 1, n)
        bc = np.linspace(0, 1, self.num_bands)
        return np.interp(xs, bc, self.gains).astype(np.float32)

    def set_gains(self, g):
        self.gains = np.array(g, dtype=float)
        self._draw()


# =============================================================================
# MAIN APP
# =============================================================================
class MultiLensTakens:
    REFRESH_MS = 90
    CHUNK_MS   = 60
    MAX_LENSES = 12

    def __init__(self, root):
        self.root = root
        self.root.title("MULTI-LENS TAKENS EXPLORER — One Electrode, Many Realities")
        self.root.geometry("1680x1020")
        self.root.configure(bg=BG)

        # Data
        self.eeg_data = None
        self.eeg_fs = None
        self.eeg_channels = []
        self.audio_data = None
        self.audio_fs = None
        self.npz = None
        self.H_mag = None

        # Playback
        self.playing = False
        self.current_samp = 0
        self._pb_thread = None
        self._buf = deque(maxlen=800_000)
        self._buf_lock = threading.Lock()
        self._audio_stream = None

        # Lenses
        self.lenses = []          # list of delay values
        self.ax_list = []         # list of 3D axes
        self.cv_list = []         # list of canvases
        self.fig = None

        # Tk vars
        self.v_num_lenses = tk.IntVar(value=6)
        self.v_max_delay = tk.IntVar(value=800)
        self.v_spacing = tk.StringVar(value="log")   # "linear" or "log"
        self.v_channel = tk.StringVar(value="")
        self.v_hf_mode = tk.StringVar(value="shape")
        self.v_speed = tk.DoubleVar(value=1.0)
        self.v_status = tk.StringVar(value="Load EEG + NPZ → Generate Lenses → Play")
        self.v_time = tk.StringVar(value="t = 0.00 s")

        self._build_ui()
        self._schedule_refresh()

    # -------------------------------------------------------------------------
    def _build_ui(self):
        # Top bar
        bar = tk.Frame(self.root, bg=PANEL, height=42)
        bar.pack(fill="x", side="top")
        tk.Label(bar, text="MULTI-LENS TAKENS EXPLORER",
                 font=("Consolas",13,"bold"), fg=ACCENT, bg=PANEL).pack(side="left", padx=16, pady=6)
        tk.Label(bar, text="One Electrode → Many Biological Timescales",
                 font=("Consolas",7), fg=DIM, bg=PANEL).pack(side="left", padx=6)

        # Body
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=BG, width=320)
        left.pack(side="left", fill="y", padx=5, pady=5)
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_left(left)
        self._build_right(right)

        # Status
        sb = tk.Frame(self.root, bg=PANEL, height=24)
        sb.pack(fill="x", side="bottom")
        tk.Label(sb, textvariable=self.v_status, fg=ACCENT, bg=PANEL,
                 font=("Consolas",6)).pack(side="left", padx=8, pady=2)
        tk.Label(sb, textvariable=self.v_time, fg=WARN, bg=PANEL,
                 font=("Consolas",6)).pack(side="right", padx=8)

    def _build_left(self, P):
        def sec(txt):
            tk.Frame(P, bg=ACCENT, height=1).pack(fill="x", pady=(8,1))
            tk.Label(P, text=txt, fg=ACCENT, bg=BG,
                     font=("Consolas",7,"bold")).pack(anchor="w")

        # File loading
        sec("LOAD FILES")
        self.lbl_eeg = tk.StringVar(value="— no EEG —")
        self.lbl_npz = tk.StringVar(value="— no NPZ —")

        ttk.Button(P, text="📂 Load EEG (.edf)", command=self.load_edf, width=28).pack(fill="x", pady=1)
        tk.Label(P, textvariable=self.lbl_eeg, fg=DIM, bg=BG, font=("Consolas",6), wraplength=300).pack(anchor="w")
        ttk.Button(P, text="📂 Load H(f) (.npz)", command=self.load_npz, width=28).pack(fill="x", pady=1)
        tk.Label(P, textvariable=self.lbl_npz, fg=DIM, bg=BG, font=("Consolas",6), wraplength=300).pack(anchor="w")

        # Channel
        sec("EEG CHANNEL")
        self.ch_combo = ttk.Combobox(P, textvariable=self.v_channel, state="readonly", width=26)
        self.ch_combo.pack(fill="x", pady=1)
        self.ch_combo.bind("<<ComboboxSelected>>", self._on_channel_change)

        # H(f) mode
        tk.Label(P, text="H(f) mode:", fg=DIM, bg=BG, font=("Consolas",6)).pack(anchor="w", pady=(4,0))
        for val, txt in [("none","Off (raw)"), ("shape","Apply H(f)"), ("inverse","Inverse (voice)")]:
            tk.Radiobutton(P, text=txt, variable=self.v_hf_mode, value=val,
                           fg=DIM, bg=BG, selectcolor=PANEL, font=("Consolas",6)).pack(anchor="w")

        # EQ
        sec("GRAPHICAL EQ")
        self.eq = GraphicalEQ(P, num_bands=12, width=280, height=48)
        self.eq.pack(pady=1)
        pf = tk.Frame(P, bg=BG)
        pf.pack(fill="x")
        for name, g in [("Flat",[1.0]*12), ("Voice",[0.01,0.05,0.2,0.6,1.0,1.0,0.8,0.5,0.3,0.15,0.08,0.03]),
                        ("Low",[1,0.9,0.7,0.4,0.2,0.1,0.05,0.02,0.01,0.01,0.01,0.01]), ("High",[0,0,0.05,0.2,0.5,0.9,1,1,0.8,0.6,0.4,0.2])]:
            ttk.Button(pf, text=name, width=6, command=lambda gg=g: self.eq.set_gains(gg)).pack(side="left", padx=1)

        # Multi-Lens Controls
        sec("MULTI-LENS CHAIN")
        lrow = lambda txt, w: (tk.Label(P, text=txt, fg=DIM, bg=BG, font=("Consolas",6)).pack(anchor="w"),
                               w.pack(fill="x", pady=1))

        tk.Label(P, text="Number of Lenses (1–12)", fg=DIM, bg=BG, font=("Consolas",6)).pack(anchor="w")
        tk.Scale(P, from_=1, to=self.MAX_LENSES, variable=self.v_num_lenses, orient="horizontal",
                 bg=PANEL, fg=ACCENT, length=260, highlightthickness=0).pack(fill="x", pady=1)

        tk.Label(P, text="Max Delay (samples)", fg=DIM, bg=BG, font=("Consolas",6)).pack(anchor="w")
        tk.Scale(P, from_=20, to=3000, variable=self.v_max_delay, orient="horizontal",
                 bg=PANEL, fg=ACCENT, length=260, highlightthickness=0).pack(fill="x", pady=1)

        tk.Label(P, text="Spacing Type", fg=DIM, bg=BG, font=("Consolas",6)).pack(anchor="w")
        ttk.Combobox(P, textvariable=self.v_spacing, values=["linear", "log"], state="readonly", width=12).pack(fill="x", pady=1)

        ttk.Button(P, text="🔄 REGENERATE LENSES", command=self._generate_lenses, width=26).pack(fill="x", pady=4)

        # Playback
        sec("PLAYBACK")
        tk.Label(P, text="Speed ×", fg=DIM, bg=BG, font=("Consolas",6)).pack(anchor="w")
        tk.Scale(P, from_=0.2, to=8.0, resolution=0.1, variable=self.v_speed, orient="horizontal",
                 bg=PANEL, fg=ACCENT, length=260, highlightthickness=0).pack(fill="x", pady=1)

        pb = tk.Frame(P, bg=BG)
        pb.pack(fill="x", pady=3)
        ttk.Button(pb, text="▶ PLAY", command=self.play, width=8).pack(side="left", padx=1)
        ttk.Button(pb, text="⏸ PAUSE", command=self.pause, width=8).pack(side="left", padx=1)
        ttk.Button(pb, text="⏹ STOP", command=self.stop, width=8).pack(side="left", padx=1)

        # Stats
        sec("STATUS")
        self.v_stats = tk.StringVar(value="Generate lenses first")
        tk.Label(P, textvariable=self.v_stats, fg=DIM, bg=BG, font=("Consolas",6), justify="left", wraplength=300).pack(anchor="w")

    def _build_right(self, P):
        self.fig = plt.Figure(figsize=(13, 8.5), facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=P)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    # -------------------------------------------------------------------------
    def _generate_lenses(self):
        n = min(self.v_num_lenses.get(), self.MAX_LENSES)
        max_d = self.v_max_delay.get()
        mode = self.v_spacing.get()

        if mode == "linear":
            delays = np.linspace(5, max_d, n).astype(int)
        else:  # log
            delays = np.logspace(np.log10(5), np.log10(max_d), n).astype(int)

        self.lenses = delays.tolist()
        self._build_lens_grid()
        self.v_status.set(f"Generated {n} lenses | {mode} spacing | max τ={max_d} samp")
        self.v_stats.set(f"Lenses: {self.lenses}\n\nEach lens shows a different biological timescale.\n"
                         "Short τ = larynx/glottal\nMedium τ = syllable/prosody\nLong τ = breathing/rhythm")

    def _build_lens_grid(self):
        self.fig.clear()
        self.ax_list = []
        self.cv_list = []

        n = len(self.lenses)
        if n == 0:
            return

        # Choose grid layout
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        for i, delay in enumerate(self.lenses):
            ax = self.fig.add_subplot(rows, cols, i+1, projection='3d')
            ax.set_facecolor(BG)
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False
                pane.set_edgecolor(PANEL)
            ax.tick_params(colors=DIM, labelsize=5)
            ax.set_title(f"τ = {delay} samp", color=ACCENT, fontsize=7, pad=2)
            self.ax_list.append(ax)

        self.fig.patch.set_facecolor(BG)
        self.canvas.draw()

    # -------------------------------------------------------------------------
    def load_edf(self):
        if not MNE_OK:
            messagebox.showerror("Missing", "pip install mne")
            return
        path = filedialog.askopenfilename(title="Load EDF", filetypes=[("EDF","*.edf")])
        if not path: return
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            self.eeg_fs = int(raw.info['sfreq'])
            self.eeg_channels = raw.ch_names
            self.eeg_data = raw.get_data().astype(np.float32)
            best = int(np.argmax(np.var(self.eeg_data, axis=1)))
            self.v_channel.set(self.eeg_channels[best])
            self.ch_combo['values'] = self.eeg_channels
            dur = self.eeg_data.shape[1] / self.eeg_fs
            self.lbl_eeg.set(f"✓ {os.path.basename(path)} | {len(self.eeg_channels)}ch @ {self.eeg_fs}Hz | {dur:.1f}s")
            self.v_status.set(f"EEG loaded — {dur:.1f}s @ {self.eeg_fs}Hz")
            self._rebuild_H()
        except Exception as ex:
            messagebox.showerror("EDF Error", str(ex))

    def load_npz(self):
        path = filedialog.askopenfilename(title="Load NPZ", filetypes=[("NPZ","*.npz")])
        if not path: return
        try:
            d = np.load(path)
            self.npz = {k: d[k] for k in d.files}
            n = len(self.npz['H_smooth'])
            fs_npz = float(self.npz.get('fs', 1200))
            self.lbl_npz.set(f"✓ {os.path.basename(path)} | {n} bins | fs={fs_npz:.0f}Hz")
            self.v_status.set(f"NPZ loaded — {n} frequency bins")
            self._rebuild_H()
        except Exception as ex:
            messagebox.showerror("NPZ Error", str(ex))

    def _rebuild_H(self):
        if self.npz is None or self.eeg_fs is None:
            return
        H_src = self.npz['H_smooth'].astype(np.float64)
        f_src = self.npz['f'].astype(np.float64)
        n_bins = self.eeg_fs // 2 + 1
        f_tgt = np.linspace(0, self.eeg_fs/2, n_bins)
        H_tgt = np.interp(f_tgt, f_src, H_src)
        H_tgt = H_tgt / (H_tgt.max() + 1e-12)
        self.H_mag = H_tgt.astype(np.float32)

    # -------------------------------------------------------------------------
    def play(self):
        if self.eeg_data is None or not self.lenses:
            messagebox.showinfo("Missing", "Load EEG and generate lenses first.")
            return
        if self.playing: return
        self.playing = True
        self._pb_thread = threading.Thread(target=self._pb_loop, daemon=True)
        self._pb_thread.start()
        self.v_status.set("▶ Playing multi-lens view...")

    def pause(self):
        self.playing = False
        self.v_status.set("⏸ Paused")

    def stop(self):
        self.playing = False
        self.current_samp = 0
        with self._buf_lock:
            self._buf.clear()
        self.v_status.set("⏹ Stopped")

    def _on_channel_change(self, *_):
        with self._buf_lock:
            self._buf.clear()

    # -------------------------------------------------------------------------
    def _pb_loop(self):
        if self.eeg_data is None: return
        ci = self.eeg_channels.index(self.v_channel.get()) if self.v_channel.get() in self.eeg_channels else 0
        sig = self.eeg_data[ci]
        N = len(sig)

        while self.playing and self.current_samp < N:
            t0 = time.perf_counter()
            speed = max(0.1, self.v_speed.get())
            n_push = int(self.eeg_fs * (self.CHUNK_MS / 1000.0) * speed)
            end = min(self.current_samp + n_push, N)
            chunk = sig[self.current_samp:end].copy()

            # H(f) shaping
            mode = self.v_hf_mode.get()
            if mode != "none" and self.H_mag is not None:
                chunk = self._spectral_shape(chunk, inverse=(mode == "inverse"))

            # EQ
            eq = self.eq.get_curve(n=max(8, len(chunk)//2 + 1))
            chunk = self._apply_eq(chunk, eq)

            with self._buf_lock:
                self._buf.extend(chunk.tolist())

            self.current_samp = end
            t_sec = self.current_samp / self.eeg_fs
            self.v_time.set(f"t = {t_sec:.2f} s")

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, self.CHUNK_MS/1000.0 - elapsed))

        if self.current_samp >= N:
            self.playing = False
            self.v_status.set("▶ Playback complete.")

    def _spectral_shape(self, chunk, inverse=False):
        if len(chunk) < 8: return chunk
        N = len(chunk)
        S = np.fft.rfft(chunk)
        H = np.interp(np.linspace(0, 1, len(S)), np.linspace(0, 1, len(self.H_mag)), self.H_mag)
        H_use = 1.0 / np.maximum(H, 0.03) if inverse else H
        H_use = H_use / H_use.max()
        return np.fft.irfft(S * H_use, n=N).astype(np.float32)

    def _apply_eq(self, chunk, gains):
        if len(chunk) < 8: return chunk
        N = len(chunk)
        S = np.fft.rfft(chunk)
        G = np.interp(np.linspace(0, 1, len(S)), np.linspace(0, 1, len(gains)), gains)
        return np.fft.irfft(S * G, n=N).astype(np.float32)

    # -------------------------------------------------------------------------
    def _schedule_refresh(self):
        self._refresh_all_lenses()
        self.root.after(self.REFRESH_MS, self._schedule_refresh)

    def _refresh_all_lenses(self):
        if not self.lenses or not self.ax_list:
            return

        with self._buf_lock:
            snap = list(self._buf)

        if len(snap) < 50:
            return

        n_pts = 3000 if len(self.lenses) > 6 else 6000   # adaptive for performance

        for i, (ax, delay) in enumerate(zip(self.ax_list, self.lenses)):
            if len(snap) < delay * 2 + 10:
                continue

            data = np.array(snap[-n_pts:], dtype=np.float32)
            x = data[       : -2*delay]
            y = data[delay  :   -delay]
            z = data[2*delay:         ]

            mn, sd = np.mean(x), np.std(x) + 1e-9
            x = (x - mn) / sd
            y = (y - mn) / sd
            z = (z - mn) / sd

            ax.clear()
            ax.set_facecolor(BG)
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False
                pane.set_edgecolor(PANEL)
            ax.tick_params(colors=DIM, labelsize=4)

            colors = np.linspace(0, 1, len(x))
            ax.scatter(x, y, z, c=colors, cmap="plasma", s=1.2, alpha=0.75, linewidths=0)

            tau_ms = delay / (self.eeg_fs or 1200) * 1000
            ax.set_title(f"τ={delay} ({tau_ms:.1f}ms)", color=ACCENT, fontsize=6, pad=1)
            ax.set_xlabel("x(t)", color=DIM, fontsize=5)
            ax.set_ylabel(f"x(t+{tau_ms:.0f}ms)", color=DIM, fontsize=5)
            ax.set_zlabel(f"x(t+{2*tau_ms:.0f}ms)", color=DIM, fontsize=5)

        try:
            self.canvas.draw_idle()
        except Exception:
            pass


# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MultiLensTakens(root)
    root.mainloop()