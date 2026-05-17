#!/usr/bin/env python3
"""
MEAT VOICE PLAYER — Interactive GUI for EEG Inverse Filtering
=============================================================
Load a transfer function (.npz), load an EDF, pick a channel,
render the "meat voice" (inverse-filtered audio), scrub the waveform,
and play any segment.

Matches the visual style of the Rajapinta EEG→Audio Studio.

Usage:
    python meat_voice_gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import mne
import soundfile as sf
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import tempfile
import threading

class MeatVoicePlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("MEAT VOICE PLAYER — EEG Inverse Filter")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a2e")
        
        # State
        self.npz_path = None
        self.edf_path = None
        self.raw = None
        self.fs = None
        self.channels = []
        self.f_bins = None
        self.h_mag = None
        self.audio = None          # full recovered audio (float32)
        self.current_ch = None
        self.temp_wav = None
        
        self._build_ui()
        
    def _build_ui(self):
        # === TOP BAR ===
        top = tk.Frame(self.root, bg="#16213e", height=60)
        top.pack(fill="x", side="top")
        
        tk.Label(top, text="MEAT VOICE PLAYER", font=("Segoe UI", 16, "bold"), 
                 fg="#00ffcc", bg="#16213e").pack(side="left", padx=20, pady=12)
        
        # Load buttons
        ttk.Button(top, text="📂 LOAD TRANSFER (.npz)", command=self.load_npz, width=22).pack(side="left", padx=8)
        ttk.Button(top, text="📂 LOAD EEG (EDF)", command=self.load_edf, width=20).pack(side="left", padx=8)
        
        self.npz_label = ttk.Label(top, text="No transfer function loaded", foreground="#888888")
        self.npz_label.pack(side="left", padx=15)
        
        # === MAIN CONTAINER ===
        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill="both", expand=True, padx=10, pady=8)
        
        # LEFT CONTROLS
        left = tk.Frame(main, bg="#0f0f23", width=280)
        left.pack(side="left", fill="y", padx=(0, 8))
        
        ttk.Label(left, text="EEG CHANNEL", font=("Segoe UI", 10, "bold"), foreground="#00ffcc").pack(pady=(15, 4))
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(left, textvariable=self.channel_var, state="disabled", width=26)
        self.channel_combo.pack(pady=2)
        self.channel_combo.bind("<<ComboboxSelected>>", self.on_channel_change)
        
        ttk.Label(left, text="SEGMENT", font=("Segoe UI", 10, "bold"), foreground="#00ffcc").pack(pady=(15, 4))
        
        time_frame = tk.Frame(left, bg="#0f0f23")
        time_frame.pack()
        ttk.Label(time_frame, text="Start (s):").grid(row=0, column=0, sticky="w")
        self.start_var = tk.DoubleVar(value=5.0)
        ttk.Entry(time_frame, textvariable=self.start_var, width=8).grid(row=0, column=1, padx=4)
        
        ttk.Label(time_frame, text="Duration (s):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.dur_var = tk.DoubleVar(value=40.0)
        ttk.Entry(time_frame, textvariable=self.dur_var, width=8).grid(row=1, column=1, padx=4, pady=(6, 0))
        
        ttk.Button(left, text="🔄 RENDER / UPDATE", command=self.render_audio, width=26).pack(pady=15)
        
        # Status
        self.status = ttk.Label(left, text="Ready — Load transfer function first", 
                                foreground="#888888", wraplength=260)
        self.status.pack(pady=10, padx=10)
        
        # CENTER — WAVEFORM
        center = tk.Frame(main, bg="#1a1a2e")
        center.pack(side="left", fill="both", expand=True)
        
        self.fig = Figure(figsize=(9, 5), facecolor="#0f0f23")
        self.fig.patch.set_facecolor("#0f0f23")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0f0f23")
        self.ax.set_title("Recovered Meat Voice Waveform", color="#00ffcc", fontsize=11)
        self.ax.tick_params(colors="#888888")
        self.ax.set_xlabel("Time (s)", color="#888888")
        self.ax.set_ylabel("Amplitude", color="#888888")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Playhead line
        self.playhead = None
        
        # BOTTOM CONTROLS
        bottom = tk.Frame(self.root, bg="#16213e", height=70)
        bottom.pack(fill="x", side="bottom")
        
        self.time_slider = ttk.Scale(bottom, from_=0, to=100, orient="horizontal", 
                                     command=self.on_slider_move, length=500)
        self.time_slider.pack(side="left", padx=15, pady=15)
        
        ttk.Button(bottom, text="▶ PLAY FROM HERE", command=self.play_from_here, width=16).pack(side="left", padx=6)
        ttk.Button(bottom, text="⏹ STOP", command=self.stop_playback, width=8).pack(side="left", padx=3)
        ttk.Button(bottom, text="💾 SAVE WAV", command=self.save_audio, width=12).pack(side="left", padx=6)
        ttk.Button(bottom, text="⟲ RESET", command=self.reset_view, width=8).pack(side="left", padx=3)
        
        self.time_label = ttk.Label(bottom, text="00:00 / 00:00", foreground="#00ffcc", font=("Segoe UI", 10))
        self.time_label.pack(side="left", padx=12)
        
        # View mode
        view_frame = tk.Frame(bottom, bg="#16213e")
        view_frame.pack(side="right", padx=15)
        ttk.Label(view_frame, text="View:", foreground="#888888").pack(side="left")
        self.view_var = tk.StringVar(value="Spectrogram")
        ttk.Radiobutton(view_frame, text="Waveform", variable=self.view_var, value="Waveform",
                        command=self._update_plot).pack(side="left", padx=4)
        ttk.Radiobutton(view_frame, text="Spectrogram", variable=self.view_var, value="Spectrogram",
                        command=self._update_plot).pack(side="left", padx=4)
        
    # ====================== LOADERS ======================
    def load_npz(self):
        path = filedialog.askopenfilename(title="Select transfer_function.npz", 
                                          filetypes=[("NPZ files", "*.npz")])
        if not path: return
        try:
            calib = np.load(path)
            self.f_bins = calib['f']
            self.h_mag = calib['H_smooth']
            self.fs = float(calib['fs'].item())
            self.npz_path = path
            self.npz_label.config(text=os.path.basename(path), foreground="#00ffcc")
            self.status.config(text="Transfer function loaded. Now load an EDF.")
        except Exception as e:
            messagebox.showerror("Error loading NPZ", str(e))
    
    def load_edf(self):
        if self.f_bins is None:
            messagebox.showwarning("No transfer function", "Load the .npz first.")
            return
        path = filedialog.askopenfilename(title="Select EDF file", 
                                          filetypes=[("EDF files", "*.edf")])
        if not path: return
        try:
            self.raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            self.fs = int(self.raw.info['sfreq'])
            self.channels = self.raw.ch_names
            self.edf_path = path
            
            self.channel_combo.config(state="normal", values=self.channels)
            if self.channels:
                self.channel_var.set(self.channels[0])
            self.status.config(text=f"Loaded: {os.path.basename(path)} | {len(self.channels)} channels")
        except Exception as e:
            messagebox.showerror("Error loading EDF", str(e))
    
    def on_channel_change(self, event=None):
        if self.audio is not None:
            if messagebox.askyesno("Re-render?", "Channel changed. Re-render audio?"):
                self.render_audio()
    
    # ====================== RENDER ======================
    def render_audio(self):
        if self.f_bins is None or self.raw is None:
            messagebox.showwarning("Missing data", "Load both .npz and EDF first.")
            return
        
        ch = self.channel_var.get()
        if not ch:
            messagebox.showwarning("No channel", "Select a channel.")
            return
        
        try:
            self.status.config(text="Rendering inverse filter...")
            self.root.update()
            
            start = float(self.start_var.get())
            dur = float(self.dur_var.get())
            start_samp = int(start * self.fs)
            stop_samp = start_samp + int(dur * self.fs)
            
            eeg = self.raw.get_data(picks=[ch], start=start_samp, stop=stop_samp)[0].astype(np.float32)
            
            # Inverse filter
            inv = 1.0 / (self.h_mag + 1e-8)
            inv = np.clip(inv, 0, np.percentile(inv, 95))
            interp = interp1d(self.f_bins, inv, bounds_error=False, fill_value="extrapolate")
            
            eeg_fft = np.fft.rfft(eeg)
            freqs = np.fft.rfftfreq(len(eeg), 1/self.fs)
            weights = interp(freqs)
            audio_fft = eeg_fft * weights
            audio = np.fft.irfft(audio_fft)
            
            # --- Adaptive bandpass for ANY EEG sampling rate ---
            nyq = self.fs / 2

            # Desired speech band
            low_target = 80
            high_target = 4000

            # Clamp to what is physically possible
            low_cut = min(low_target, nyq * 0.4)
            high_cut = min(high_target, nyq * 0.9)

            # Ensure valid ordering
            if low_cut >= high_cut:
                # fallback: just use a safe fraction band
                low_cut = nyq * 0.1
                high_cut = nyq * 0.9

            # Final safety (still needed)
            if high_cut <= low_cut:
                low_cut = nyq * 0.05
                high_cut = nyq * 0.8

            sos = signal.butter(
                6,
                [low_cut, high_cut],
                btype='band',
                fs=self.fs,
                output='sos'
            )

            audio = signal.sosfiltfilt(sos, audio)
            
            # Normalize
            peak = np.max(np.abs(audio)) + 1e-9
            self.audio = (audio / peak * 0.95).astype(np.float32)
            self.current_ch = ch
            
            # Force Spectrogram view by default upon rendering
            self.view_var.set("Spectrogram")
            self._update_plot()
            
            self.status.config(text=f"Rendered {len(self.audio)/self.fs:.1f}s from {ch}")
            
        except Exception as e:
            messagebox.showerror("Render error", str(e))
            self.status.config(text="Render failed")
    
    def _update_plot(self):
        if self.audio is None: return
        self.ax.clear()
        mode = self.view_var.get()
        t = np.linspace(0, len(self.audio)/self.fs, len(self.audio))
        
        if mode == "Waveform":
            self.ax.plot(t, self.audio, color="#00ffcc", linewidth=0.7)
            self.ax.set_ylabel("Amplitude", color="#888888")
            self.ax.set_title(f"Meat Voice — {self.current_ch}  ({len(self.audio)/self.fs:.1f}s)", 
                              color="#00ffcc", fontsize=11)
        else:  # Spectrogram upgraded for human voice
            f, t_spec, Sxx = signal.spectrogram(
                self.audio,
                fs=self.fs,
                nperseg=256,
                noverlap=200,
                window='hann',
                scaling='spectrum',
                mode='magnitude'
            )
            
            Sxx_db = 20 * np.log10(Sxx + 1e-8)
            
            mesh = self.ax.pcolormesh(
                t_spec, 
                f, 
                Sxx_db, 
                shading='gouraud', 
                cmap='inferno'
            )
                        
            self.ax.set_ylim(0, 4000)
            mesh.set_clim(
                vmin=np.percentile(Sxx_db, 5),
                vmax=np.percentile(Sxx_db, 95)
            )
            
            self.ax.set_ylabel("Frequency (Hz)", color="#888888")
            self.ax.set_title(f"Spectrogram — {self.current_ch}  (clear voice structure)", 
                              color="#00ffcc", fontsize=11)
        
        self.ax.set_facecolor("#0f0f23")
        self.ax.tick_params(colors="#888888")
        self.ax.set_xlabel("Time (s)", color="#888888")
        self.ax.grid(True, alpha=0.15, color="#444444")
        
        # Bulletproof playhead removal
        if getattr(self, 'playhead', None) is not None:
            try:
                self.playhead.remove()
            except Exception:
                pass
            self.playhead = None
            
        if mode == "Waveform":
            self.playhead = self.ax.axvline(0, color="#ffaa00", linewidth=2, alpha=0.8)
        
        self.canvas.draw()
        self.time_slider.config(to=len(self.audio)/self.fs)
        self.time_slider.set(0)
        self._update_time_label(0)
    
    # ====================== PLAYBACK ======================
    def on_slider_move(self, val):
        if self.audio is None: return
        t = float(val)
        if self.playhead:
            self.playhead.set_xdata([t])
        self.canvas.draw_idle()
        self._update_time_label(t)
    
    def _update_time_label(self, t):
        total = len(self.audio)/self.fs if self.audio is not None else 0
        self.time_label.config(text=f"{t:05.1f} / {total:05.1f} s")
    
    def play_from_here(self):
        if self.audio is None: return
        pos = float(self.time_slider.get())
        start_samp = int(pos * self.fs)
        segment = self.audio[start_samp:]
        
        # Write temp WAV
        if self.temp_wav and os.path.exists(self.temp_wav):
            os.remove(self.temp_wav)
        self.temp_wav = os.path.join(tempfile.gettempdir(), "meat_voice_preview.wav")
        sf.write(self.temp_wav, segment, int(self.fs))
        
        # Play with system player (cross-platform)
        try:
            if os.name == "nt":
                os.startfile(self.temp_wav)
            else:
                import subprocess
                subprocess.Popen(["aplay", self.temp_wav] if os.path.exists("/usr/bin/aplay") 
                                 else ["xdg-open", self.temp_wav])
            self.status.config(text=f"Playing from {pos:.1f}s...")
        except Exception as e:
            messagebox.showerror("Playback error", str(e))
    
    def save_audio(self):
        if self.audio is None:
            messagebox.showwarning("Nothing to save", "Render audio first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Meat Voice as WAV",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")]
        )
        if path:
            sf.write(path, self.audio, int(self.fs))
            self.status.config(text=f"Saved: {os.path.basename(path)}")
    
    def stop_playback(self):
        # On Windows we can't easily kill the player, so just inform
        self.status.config(text="Playback stopped (close system player if needed)")
    
    def reset_view(self):
        if self.audio is None: return
        self.time_slider.set(0)
        if self.playhead:
            self.playhead.set_xdata([0])
        self.canvas.draw_idle()
        self._update_time_label(0)

# ====================== MAIN ======================
if __name__ == "__main__":
    root = tk.Tk()
    app = MeatVoicePlayer(root)
    root.mainloop()