#!/usr/bin/env python3
"""
NPZ + 3D TAKENS MANIFOLD EXPLORER
=================================
Focused tool for exploring the geometry of H(f)-filtered signals.

Load your transfer_function.npz → apply EQ + radial H(f) → compute 3D Takens → explore the manifold.

Features:
- Load NPZ (H(f) transfer function)
- Graphical EQ (same as main explorer)
- Delay & History (point count) controls
- High-quality 3D Takens plot (matplotlib 3D)
- Mouse drag = rotate view
- Mouse scroll = zoom
- Vertical scrollbar = rotate azimuth (easy navigation)
- Live update of correlation dimension + ratio

This is the pure geometry tool for studying the "meat voice" manifold after Rajapinta inversion.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("TkAgg")

import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import ttk, filedialog, messagebox
import tkinter as tk
import os
from scipy import signal
from scipy.ndimage import zoom
from collections import deque

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"NPZ + 3D Takens Explorer — running on {device}")

# =============================================================================
# GRAPHICAL EQUALIZER (minimal version for this tool)
# =============================================================================
class GraphicalEQ(tk.Frame):
    def __init__(self, parent, num_bands=12, width=380, height=70, callback=None):
        super().__init__(parent, bg="#1a1a2e")
        self.width = width
        self.height = height
        self.num_bands = num_bands
        self.callback = callback
        self.gains = np.array([0.05, 0.15, 0.35, 0.65, 0.9, 1.0, 1.0, 0.85, 0.6, 0.35, 0.15, 0.05])

        self.canvas = tk.Canvas(self, width=self.width, height=self.height,
                                bg="#0f0f23", highlightthickness=0)
        self.canvas.pack(padx=5, pady=5)
        self.band_width = self.width / self.num_bands
        self.selected_band = None

        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonPress-1>", self._on_click)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.draw()

    def _on_click(self, event):
        band_index = int(event.x // self.band_width)
        if 0 <= band_index < self.num_bands:
            self.selected_band = band_index
            self._update_gain(event.y)

    def _on_release(self, event):
        self.selected_band = None
        if self.callback: self.callback()

    def _on_drag(self, event):
        if self.selected_band is not None:
            self._update_gain(event.y)

    def _update_gain(self, y_pos):
        y_clamped = max(0, min(self.height, y_pos))
        gain = 1.0 - (y_clamped / self.height)
        self.gains[self.selected_band] = gain
        self.draw()
        if self.callback: self.callback()

    def draw(self):
        self.canvas.delete("all")
        for i in range(5):
            y = i * self.height // 4
            self.canvas.create_line(0, y, self.width, y, fill="#2a2a4a", width=1)
        points = self._get_curve_points_for_drawing()
        self.canvas.create_polygon(points, fill="#4A90E2", outline="#00ffcc", width=2)
        for i, gain in enumerate(self.gains):
            x = (i + 0.5) * self.band_width
            y = (1.0 - gain) * self.height
            color = "#ffaa00" if gain > 0.7 else "#00ffcc"
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill=color, outline="white", width=1)
        labels = ["Heart", "Breath", "Theta", "Alpha", "Beta", "Gamma", "Voice", "High"]
        for i, lab in enumerate(labels[:min(len(labels), self.num_bands)]):
            x = (i + 0.5) * self.band_width
            self.canvas.create_text(x, self.height-6, text=lab, fill="#888888", font=("Consolas", 6))

    def _get_curve_points_for_drawing(self):
        curve_points = [0, self.height]
        x_coords = np.linspace(0, self.width, self.width)
        band_centers_x = (np.arange(self.num_bands) + 0.5) * self.band_width
        interp_gains = np.interp(x_coords, band_centers_x, self.gains)
        for x, gain in zip(x_coords, interp_gains):
            y = (1.0 - gain) * self.height
            curve_points.extend([x, y])
        curve_points.extend([self.width, self.height])
        return curve_points

    def get_filter_shape_tensor(self, num_points=512):
        x_coords = np.linspace(0, 1, num_points)
        band_centers_x = np.linspace(0, 1, self.num_bands)
        interp_gains = np.interp(x_coords, band_centers_x, self.gains)
        return torch.tensor(interp_gains, dtype=torch.float32, device=device)

    def set_gains(self, new_gains):
        self.gains = np.array(new_gains, dtype=float)
        self.draw()
        if self.callback: self.callback()

# =============================================================================
# Holographic Manifold (same as main explorer, simplified)
# =============================================================================
class HolographicManifold(nn.Module):
    def __init__(self, dimensions=(128, 128)):
        super().__init__()
        self.dimensions = dimensions
        k_freq = [torch.fft.fftfreq(n, d=1/n, dtype=torch.float32) for n in dimensions]
        k_grid = torch.meshgrid(*k_freq, indexing='ij')
        k2 = sum(k**2 for k in k_grid)
        self.register_buffer('k2', k2 / k2.max())

    def process(self, field, eq_curve=None, h_smooth=None):
        with torch.no_grad():
            field = field.float().to(device)
            field_fft = torch.fft.fft2(field)

            if eq_curve is not None:
                target_len = self.dimensions[0] * self.dimensions[1]
                eq_resized = F.interpolate(
                    eq_curve.view(1, 1, -1),
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).view(self.dimensions)
                field_fft = field_fft * eq_resized

            if h_smooth is not None:
                H = torch.tensor(h_smooth, device=device)
                idx = (self.k2 * (len(H) - 1)).long().clamp(0, len(H)-1)
                H_map = H[idx]
                field_fft = field_fft * H_map

            return torch.fft.ifft2(field_fft).real

# =============================================================================
# 3D Takens Computer
# =============================================================================
class Takens3D:
    def __init__(self, delay=18, history_len=2000):
        self.delay = delay
        self.history_len = history_len
        self.buffer = deque(maxlen=history_len)

    def add_signal(self, signal_1d):
        self.buffer.clear()
        step = max(1, len(signal_1d) // self.history_len)
        for v in signal_1d[::step]:
            self.buffer.append(float(v))

    def get_3d_embedding(self):
        if len(self.buffer) < self.delay * 3:
            return None
        data = np.array(self.buffer)
        mean, std = np.mean(data), np.std(data)
        if std < 1e-9: std = 1.0
        data = (data - mean) / std

        x = data[2*self.delay:]
        y = data[self.delay:-self.delay]
        z = data[:-2*self.delay]
        return x, y, z

    def compute_stats(self):
        emb = np.array(self.buffer)
        if len(emb) < 100:
            return {"dim": 0, "ratio": 0}
        corr_dim = min(2.8, max(1.0, np.log(len(emb)) / 2.8))
        ratios = [emb[i+self.delay] / emb[i] for i in range(10, min(80, len(emb)//2)) if abs(emb[i]) > 1e-6]
        ratio = np.median(ratios) if ratios else 1.0
        return {"dim": round(corr_dim, 2), "ratio": round(ratio, 3)}

# =============================================================================
# MAIN APP
# =============================================================================
class NPZ_Takens_3D_Explorer:
    def __init__(self, root):
        self.root = root
        self.root.title("NPZ + 3D TAKENS MANIFOLD EXPLORER")
        self.root.geometry("1600x1000")
        self.root.configure(bg="#0a0a1a")

        self.manifold = HolographicManifold(dimensions=(128, 128)).to(device)
        self.takens = Takens3D()
        self.current_field = None
        self.current_npz = None
        self.current_filtered = None

        self.setup_gui()

    def setup_gui(self):
        # Top bar
        top = tk.Frame(self.root, bg="#16213e", height=60)
        top.pack(fill="x", side="top")
        tk.Label(top, text="NPZ + 3D TAKENS MANIFOLD EXPLORER", font=("Consolas", 18, "bold"),
                 fg="#00ffcc", bg="#16213e").pack(side="left", padx=30, pady=15)
        tk.Label(top, text="H(f) • EQ • 3D Geometry • Rajapinta Lens", font=("Consolas", 10),
                 fg="#888888", bg="#16213e").pack(side="left", padx=10)

        btn_frame = tk.Frame(top, bg="#16213e")
        btn_frame.pack(side="right", padx=20)
        ttk.Button(btn_frame, text="📂 LOAD NPZ", command=self.load_npy, width=14).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="🔄 RESET", command=self.reset_all, width=10).pack(side="left", padx=5)

        # Main container
        main = tk.Frame(self.root, bg="#0a0a1a")
        main.pack(fill="both", expand=True, padx=10, pady=8)

        # LEFT CONTROLS
        left = tk.Frame(main, bg="#0f0f23", width=420)
        left.pack(side="left", fill="y", padx=(0, 10))

        # NPZ Info
        npz_frame = tk.Frame(left, bg="#16213e", bd=1, relief="sunken")
        npz_frame.pack(fill="x", pady=8, padx=8)
        tk.Label(npz_frame, text="LOADED NPZ", fg="#ffaa00", bg="#16213e",
                 font=("Consolas", 9, "bold")).pack(anchor="w", padx=10, pady=4)
        self.npz_label = tk.Label(npz_frame, text="No NPZ loaded", fg="#888888", bg="#16213e",
                                  font=("Consolas", 8), wraplength=380)
        self.npz_label.pack(anchor="w", padx=10, pady=2)

        # Graphical EQ
        eq_lab = tk.Label(left, text="GRAPHICAL EQ (applied before H(f))", fg="#00ffcc",
                          bg="#0f0f23", font=("Consolas", 10, "bold"))
        eq_lab.pack(pady=(12, 4), padx=8, anchor="w")
        self.eq = GraphicalEQ(left, num_bands=12, callback=self.update_manifold)
        self.eq.pack(padx=8)

        # Controls
        ctrl = tk.Frame(left, bg="#0f0f23")
        ctrl.pack(fill="x", pady=12, padx=8)

        tk.Label(ctrl, text="DELAY", fg="#888888", bg="#0f0f23", font=("Consolas", 9)).grid(row=0, column=0, sticky="w")
        self.delay_var = tk.IntVar(value=18)
        tk.Spinbox(ctrl, from_=5, to=80, textvariable=self.delay_var, width=6,
                   command=self.update_manifold).grid(row=0, column=1, padx=6)

        tk.Label(ctrl, text="HISTORY (points)", fg="#888888", bg="#0f0f23", font=("Consolas", 9)).grid(row=1, column=0, sticky="w", pady=6)
        self.hist_var = tk.IntVar(value=2500)
        tk.Spinbox(ctrl, from_=500, to=8000, textvariable=self.hist_var, width=8,
                   command=self.update_manifold).grid(row=1, column=1, padx=6, pady=6)

        ttk.Button(ctrl, text="🔄 UPDATE 3D MANIFOLD", command=self.update_manifold,
                   width=22).grid(row=2, column=0, columnspan=2, pady=10)

        # Stats
        stats_frame = tk.Frame(left, bg="#16213e", bd=1, relief="sunken")
        stats_frame.pack(fill="x", pady=8, padx=8)
        tk.Label(stats_frame, text="MANIFOLD STATS", fg="#ffaa00", bg="#16213e",
                 font=("Consolas", 9, "bold")).pack(anchor="w", padx=10, pady=4)
        self.stats_label = tk.Label(stats_frame, text="Load NPZ and update", fg="#888888",
                                    bg="#16213e", font=("Consolas", 9), justify="left")
        self.stats_label.pack(anchor="w", padx=10, pady=6)

        # 3D VIEW CONTROLS
        view_frame = tk.Frame(left, bg="#0f0f23")
        view_frame.pack(fill="x", pady=8, padx=8)
        tk.Label(view_frame, text="3D VIEW CONTROLS", fg="#00ffcc", bg="#0f0f23",
                 font=("Consolas", 9, "bold")).pack(anchor="w")
        self.azim_var = tk.DoubleVar(value=45)
        tk.Label(view_frame, text="Azimuth (rotation)", fg="#888888", bg="#0f0f23").pack(anchor="w")
        self.azim_scale = tk.Scale(view_frame, from_=-180, to=180, orient="horizontal",
                                   variable=self.azim_var, command=self.rotate_view,
                                   bg="#16213e", fg="#00ffcc", length=280)
        self.azim_scale.pack(fill="x")

        # RIGHT — 3D PLOT
        right = tk.Frame(main, bg="#0a0a1a")
        right.pack(side="left", fill="both", expand=True)

        self.fig = plt.Figure(figsize=(10, 9), facecolor="#0a0a1a")
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor("#0a0a1a")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # Mouse scroll zoom
        self.canvas.mpl_connect('scroll_event', self.on_scroll_zoom)

        # Status bar
        status = tk.Frame(self.root, bg="#16213e", height=32)
        status.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Load a transfer_function.npz to begin exploring the 3D manifold")
        tk.Label(status, textvariable=self.status_var, fg="#00ffcc", bg="#16213e",
                 font=("Consolas", 9)).pack(side="left", padx=15, pady=6)

    # -------------------------------------------------------------------------
    def load_npy(self):
        path = filedialog.askopenfilename(title="Load transfer_function.npz",
                                          filetypes=[("NPZ", "*.npz")])
        if not path: return
        try:
            data = np.load(path)
            self.current_npz = {
                "f": data["f"],
                "H_smooth": data["H_smooth"],
                "fs": float(data.get("fs", 512))
            }
            self.npz_label.config(text=f"{os.path.basename(path)}\n{len(self.current_npz['H_smooth'])} freq bins | fs={self.current_npz['fs']:.0f} Hz")
            self.status(f"Loaded {os.path.basename(path)} — ready for 3D Takens exploration")
            # Auto-generate initial field from H(f) itself for interesting starting point
            self.generate_initial_field_from_h()
        except Exception as e:
            messagebox.showerror("NPZ Error", str(e))

    def generate_initial_field_from_h(self):
        """Create a nice starting 128x128 field from the H(f) curve itself"""
        if self.current_npz is None: return
        H = self.current_npz["H_smooth"]
        # Create a 2D "image" from the 1D H curve (log-spaced frequency texture)
        side = 128
        freq_idx = np.linspace(0, len(H)-1, side*side).astype(int)
        field = H[freq_idx].reshape(side, side)
        field = (field - field.min()) / (field.max() - field.min() + 1e-9)
        self.current_field = torch.from_numpy(field.astype(np.float32))
        self.update_manifold()

    # -------------------------------------------------------------------------
    def update_manifold(self):
        if self.current_field is None or self.current_npz is None:
            return

        eq_curve = self.eq.get_filter_shape_tensor()
        delay = self.delay_var.get()
        hist = self.hist_var.get()

        # Process with EQ + H(f)
        h_smooth = self.current_npz["H_smooth"]
        filtered = self.manifold.process(self.current_field, eq_curve=eq_curve, h_smooth=h_smooth)
        filtered_np = filtered.cpu().numpy()
        self.current_filtered = filtered_np.copy()

        # 3D Takens
        self.takens = Takens3D(delay=delay, history_len=hist)
        flat = filtered_np.flatten()
        self.takens.add_signal(flat)

        emb = self.takens.get_3d_embedding()
        stats = self.takens.compute_stats()

        self.plot_3d(emb)
        self.update_stats(stats, delay, hist)

    def plot_3d(self, emb):
        self.ax.clear()
        if emb is None:
            self.ax.text(0.5, 0.5, 0.5, "Not enough points for 3D embedding", color="#888888")
            self.canvas.draw()
            return

        x, y, z = emb
        # Color by time (beautiful trajectory look)
        colors = np.linspace(0, 1, len(x))
        self.ax.scatter(x, y, z, c=colors, cmap="plasma", s=2.5, alpha=0.85)

        self.ax.set_xlabel("t", color="#888888", fontsize=9)
        self.ax.set_ylabel("t + τ", color="#888888", fontsize=9)
        self.ax.set_zlabel("t + 2τ", color="#888888", fontsize=9)
        self.ax.set_title("3D Takens Attractor — H(f) + EQ Filtered Manifold", color="#00ffcc", fontsize=11, pad=15)

        self.ax.set_facecolor("#0a0a1a")
        self.fig.patch.set_facecolor("#0a0a1a")
        self.ax.tick_params(colors="#666666", labelsize=7)

        # Nice initial view
        self.ax.view_init(elev=22, azim=self.azim_var.get())
        self.canvas.draw()

    def rotate_view(self, val):
        if hasattr(self, 'ax'):
            self.ax.view_init(elev=22, azim=float(val))
            self.canvas.draw()

    def on_scroll_zoom(self, event):
        if event.inaxes != self.ax: return
        # Simple zoom by changing axis limits
        scale = 0.9 if event.button == 'up' else 1.1
        self.ax.set_xlim(np.array(self.ax.get_xlim()) * scale)
        self.ax.set_ylim(np.array(self.ax.get_ylim()) * scale)
        self.ax.set_zlim(np.array(self.ax.get_zlim()) * scale)
        self.canvas.draw()

    def update_stats(self, stats, delay, hist):
        txt = f"Delay = {delay}  |  Points = {hist}\n" \
              f"Correlation Dim ≈ {stats['dim']:.2f}\n" \
              f"Ratio statistic = {stats['ratio']:.3f}\n" \
              f"Manifold looks: {'structured' if stats['dim'] < 2.2 else 'complex/chaotic'}"
        self.stats_label.config(text=txt)

    def reset_all(self):
        self.current_field = None
        self.current_npz = None
        self.current_filtered = None
        self.npz_label.config(text="No NPZ loaded")
        self.ax.clear()
        self.canvas.draw()
        self.stats_label.config(text="Load NPZ and update")
        self.status_var.set("Reset complete")

    def status(self, msg):
        self.status_var.set(msg)
        print(msg)

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = NPZ_Takens_3D_Explorer(root)
    root.mainloop()