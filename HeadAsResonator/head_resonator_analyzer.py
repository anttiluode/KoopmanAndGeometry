#!/usr/bin/env python3
"""
HEAD AS RESONATOR: EEG ↔ Voice Transfer Function Analyzer
=========================================================
Treats the human head + electrode/cable system as a passive acoustic resonator.
Estimates the frequency response H(f) = EEG(f) / Voice(f) from simultaneous
EDF + WAV recordings.

This is the PHYSICAL / mechanical coupling model (microphonic artifact),
NOT the neural "inverse cochlea" mapping.

Key papers this builds on:
- Roussel et al. (2020) J Neural Eng: Acoustic contamination via mechanical
  coupling to cables/connectors (microphonic effect). Provides toolbox.
- Bush et al. (2022) NeuroImage: Speech-induced microphonic artifacts in
  intracranial recordings track F0 and harmonics; detected via coherence
  and spectrogram similarity.

Usage:
  python head_resonator_analyzer.py --edf your_recording.edf --wav your_speech.wav \
      --channel Cz --duration 60 --lag-ms 0 --output-dir ./results

Requirements: mne, numpy, scipy, soundfile, matplotlib (same as your studio)
"""

import argparse
import os
import numpy as np
import soundfile as sf
import mne
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ====================== CORE TRANSFER FUNCTION ======================
def compute_transfer_function(eeg, voice, fs, nperseg=1024, noverlap=512):
    """
    Estimate complex transfer function H(f) = EEG(f) / Voice(f)
    Returns:
        f : frequencies
        H_mean : median |H(f)| across time (robust)
        H_phase : median phase (optional)
        coherence : magnitude-squared coherence
    """
    # STFT
    f, t, S_eeg = signal.stft(eeg, fs=fs, nperseg=nperseg, noverlap=noverlap,
                              window='hann', boundary=None)
    _, _, S_voice = signal.stft(voice, fs=fs, nperseg=nperseg, noverlap=noverlap,
                                window='hann', boundary=None)
    
    # Avoid division by zero / tiny
    eps = 1e-10 * np.max(np.abs(S_voice))
    H = S_eeg / (S_voice + eps)
    
    # Robust average (median over time frames, less sensitive to bursts)
    H_mag = np.abs(H)
    H_mean = np.median(H_mag, axis=1)
    H_phase = np.median(np.angle(H), axis=1)
    
    # Magnitude-squared coherence (standard system ID measure)
    # 1.0 = perfect linear coupling, 0 = no coupling or nonlinear/noise
    coh_f, coherence = signal.coherence(voice, eeg, fs=fs, nperseg=nperseg,
                                        noverlap=noverlap)
    
    return f, H_mean, H_phase, coh_f, coherence, t

def smooth_spectrum(x, sigma=3):
    """Light Gaussian smoothing for nice curves"""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x, sigma=sigma, mode='nearest')

def find_best_lag(eeg, voice, fs, max_lag_ms=50):
    """
    Find best alignment lag (in ms) via cross-correlation on bandpass envelopes.
    For physical microphonic coupling, expect lag ≈ 0–10 ms.
    (384 ms was for neural; here we are measuring mechanical.)
    """
    # Bandpass 20-150 Hz (where microphonic often appears)
    sos = signal.butter(4, [20, min(150, fs/2-10)], btype='band', fs=fs, output='sos')
    eeg_bp = signal.sosfiltfilt(sos, eeg)
    voice_bp = signal.sosfiltfilt(sos, voice)
    
    # Envelope (abs + lowpass)
    env_eeg = signal.sosfiltfilt(signal.butter(4, 10, btype='low', fs=fs, output='sos'),
                                 np.abs(eeg_bp))
    env_voice = signal.sosfiltfilt(signal.butter(4, 10, btype='low', fs=fs, output='sos'),
                                   np.abs(voice_bp))
    
    # Cross-corr
    corr = signal.correlate(env_eeg - env_eeg.mean(), env_voice - env_voice.mean(),
                            mode='full')
    lags = signal.correlation_lags(len(env_eeg), len(env_voice), mode='full')
    
    # Limit search
    max_lag_samp = int(max_lag_ms / 1000 * fs)
    mask = np.abs(lags) <= max_lag_samp
    best_idx = np.argmax(corr[mask])
    best_lag_samp = lags[mask][best_idx]
    best_lag_ms = best_lag_samp / fs * 1000
    
    return best_lag_ms, corr[mask], lags[mask]

# ====================== MAIN ANALYSIS ======================
def analyze(edf_path, wav_path, channel=None, duration=60.0, lag_ms=0.0,
            output_dir="head_resonance_results", n_resonators_for_overlay=32):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("HEAD AS RESONATOR — EEG ↔ VOICE TRANSFER FUNCTION")
    print("=" * 70)
    print(f"EDF : {edf_path}")
    print(f"WAV : {wav_path}")
    print(f"Duration limit : {duration}s | Forced lag : {lag_ms} ms")
    
    # --- LOAD EEG ---
    print("\n[1/6] Loading EDF...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    fs_eeg = int(raw.info['sfreq'])
    print(f"      EEG fs = {fs_eeg} Hz | {len(raw.ch_names)} channels")
    
    if channel is None:
        # Auto-pick: highest variance channel (often Cz or temporal for speech artifacts)
        variances = [np.var(raw.get_data(picks=[ch])[0]) for ch in raw.ch_names]
        channel = raw.ch_names[np.argmax(variances)]
        print(f"      Auto-selected channel with highest variance: {channel}")
    else:
        if channel not in raw.ch_names:
            print(f"      WARNING: {channel} not found. Using first channel.")
            channel = raw.ch_names[0]
    
    # Trim to first N seconds
    max_samples = int(duration * fs_eeg)
    eeg = raw.get_data(picks=[channel], start=0, stop=max_samples)[0].astype(np.float32)
    eeg = eeg - np.mean(eeg)  # DC remove
    
    # --- LOAD AUDIO ---
    print("\n[2/6] Loading WAV...")
    voice, fs_voice = sf.read(wav_path)
    if voice.ndim > 1:
        voice = np.mean(voice, axis=1)  # mono
    voice = voice.astype(np.float32)
    
    # Resample voice to EEG fs (we only care about frequencies < fs_eeg/2)
    if fs_voice != fs_eeg:
        print(f"      Resampling voice {fs_voice} → {fs_eeg} Hz")
        voice = signal.resample_poly(voice, fs_eeg, fs_voice)
    
    # Trim & align
    min_len = min(len(eeg), len(voice))
    eeg = eeg[:min_len]
    voice = voice[:min_len]
    
    # Apply lag (shift EEG forward = positive lag means voice leads EEG)
    if lag_ms != 0:
        lag_samp = int(abs(lag_ms) / 1000 * fs_eeg)
        if lag_ms > 0:
            eeg = eeg[lag_samp:]
            voice = voice[:-lag_samp] if len(voice) > lag_samp else voice
        else:
            voice = voice[lag_samp:]
            eeg = eeg[:-lag_samp] if len(eeg) > lag_samp else eeg
        min_len = min(len(eeg), len(voice))
        eeg = eeg[:min_len]
        voice = voice[:min_len]
        print(f"      Applied {lag_ms} ms lag shift")
    
    print(f"      Final length: {min_len/fs_eeg:.1f} s | {len(eeg)} samples")
    
    # Z-score for stability (focus on shape, not amplitude)
    eeg = (eeg - np.mean(eeg)) / (np.std(eeg) + 1e-8)
    voice = (voice - np.mean(voice)) / (np.std(voice) + 1e-8)
    
    # --- OPTIONAL: AUTO LAG SEARCH ---
    print("\n[3/6] Estimating best mechanical lag (cross-corr on 20-150 Hz envelopes)...")
    auto_lag_ms, _, _ = find_best_lag(eeg, voice, fs_eeg, max_lag_ms=50)
    print(f"      Best estimated lag: {auto_lag_ms:.1f} ms (use --lag-ms {auto_lag_ms:.0f} to apply)")
    if abs(auto_lag_ms) > 5:
        print("      NOTE: Lag >5 ms suggests possible neural component or cable delay.")
    else:
        print("      Lag ≈0 ms — consistent with pure mechanical microphonic coupling.")
    
    # --- COMPUTE TRANSFER FUNCTION ---
    print("\n[4/6] Computing transfer function H(f) = EEG(f) / Voice(f) ...")
    nperseg = min(2048, len(eeg)//4)  # adaptive
    f, H_mean, H_phase, coh_f, coherence, t_spec = compute_transfer_function(
        eeg, voice, fs_eeg, nperseg=nperseg)
    
    # Smooth for nice plot
    H_smooth = smooth_spectrum(H_mean, sigma=4)
    
    # --- PLOTS ---
    print("\n[5/6] Generating plots...")
    fig = plt.figure(figsize=(14, 10), facecolor='#0f0f23')
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1.2, 1.0])
    
    # 1. Voice spectrogram
    ax1 = fig.add_subplot(gs[0, 0])
    f_v, t_v, S_v = signal.spectrogram(voice, fs=fs_eeg, nperseg=512, noverlap=384)
    ax1.pcolormesh(t_v, f_v, 10*np.log10(S_v + 1e-10), shading='gouraud', cmap='magma')
    ax1.set_title(f'VOICE (resampled to {fs_eeg} Hz)', color='#00ffcc', fontsize=11)
    ax1.set_ylabel('Freq (Hz)', color='#888888')
    ax1.tick_params(colors='#888888')
    ax1.set_ylim(0, min(4000, fs_eeg/2))
    ax1.set_facecolor('#0f0f23')
    
    # 2. EEG spectrogram (same scale)
    ax2 = fig.add_subplot(gs[0, 1])
    f_e, t_e, S_e = signal.spectrogram(eeg, fs=fs_eeg, nperseg=512, noverlap=384)
    ax2.pcolormesh(t_e, f_e, 10*np.log10(S_e + 1e-10), shading='gouraud', cmap='magma')
    ax2.set_title(f'EEG {channel} (Z-scored)', color='#00ffcc', fontsize=11)
    ax2.set_ylabel('Freq (Hz)', color='#888888')
    ax2.tick_params(colors='#888888')
    ax2.set_ylim(0, min(4000, fs_eeg/2))
    ax2.set_facecolor('#0f0f23')
    
    # 3. Transfer function |H(f)|
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor('#0f0f23')
    ax3.semilogy(f, H_smooth, color='#ffaa00', linewidth=2.5, label='|H(f)| median (smoothed)')
    ax3.fill_between(f, H_smooth*0.7, H_smooth*1.3, alpha=0.2, color='#ffaa00')
    ax3.axvline(50, color='#888888', linestyle='--', alpha=0.5, label='50 Hz line noise')
    ax3.axvline(100, color='#888888', linestyle=':', alpha=0.5, label='~F0 / bone conduction')
    ax3.set_xlabel('Frequency (Hz)', color='#888888')
    ax3.set_ylabel('|H(f)|  (EEG / Voice)', color='#888888')
    ax3.set_title('HEAD + ELECTRODE + CABLE RESONANCE CURVE  (Transfer Function)', color='#00ffcc', fontsize=12)
    ax3.tick_params(colors='#888888')
    ax3.legend(facecolor='#16213e', labelcolor='white', loc='upper right')
    ax3.grid(True, alpha=0.15, color='#444444')
    ax3.set_xlim(0, min(300, fs_eeg/2))  # Focus on relevant band
    
    # Annotate expected behavior
    ax3.text(0.02, 0.95, "Expected: low-pass (skull attenuates >200-300 Hz)\n"
                         "Peaks = bone conduction / cable resonances\n"
                         "High coherence = strong mechanical coupling",
             transform=ax3.transAxes, fontsize=9, color='#888888',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8))
    
    # 4. Coherence
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor('#0f0f23')
    ax4.plot(coh_f, coherence, color='#00ffcc', linewidth=1.8)
    ax4.axhline(0.15, color='#ffaa00', linestyle='--', label='Weak coupling threshold')
    ax4.fill_between(coh_f, 0, coherence, where=(coherence > 0.15), color='#00ffcc', alpha=0.3)
    ax4.set_xlabel('Frequency (Hz)', color='#888888')
    ax4.set_ylabel('Magnitude-Squared Coherence', color='#888888')
    ax4.set_title('LINEAR COUPLING STRENGTH (1.0 = perfect microphonic)', color='#00ffcc', fontsize=10)
    ax4.tick_params(colors='#888888')
    ax4.set_ylim(0, 1)
    ax4.legend(facecolor='#16213e', labelcolor='white')
    ax4.grid(True, alpha=0.15, color='#444444')
    ax4.set_xlim(0, min(300, fs_eeg/2))
    
    # 5. Time-domain comparison (first 10s)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor('#0f0f23')
    t = np.arange(min(10000, len(eeg))) / fs_eeg
    ax5.plot(t, voice[:len(t)]*0.3 + 2, color='#ffaa00', linewidth=0.8, label='Voice (scaled)')
    ax5.plot(t, eeg[:len(t)], color='#00ffcc', linewidth=0.6, label=f'EEG {channel}')
    ax5.set_xlabel('Time (s)', color='#888888')
    ax5.set_ylabel('Amplitude (Z-score)', color='#888888')
    ax5.set_title('RAW TIME SERIES (first ~10s) — look for shared bursts', color='#00ffcc', fontsize=10)
    ax5.tick_params(colors='#888888')
    ax5.legend(facecolor='#16213e', labelcolor='white', loc='upper right')
    ax5.grid(True, alpha=0.1, color='#444444')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "head_resonance_analysis.png")
    plt.savefig(plot_path, dpi=150, facecolor='#0f0f23', edgecolor='none')
    print(f"      Saved: {plot_path}")
    plt.close()
    
    # --- SUMMARY STATS ---
    print("\n[6/6] SUMMARY")
    max_coh = np.max(coherence)
    max_coh_f = coh_f[np.argmax(coherence)]
    print(f"      Peak coherence: {max_coh:.3f} at {max_coh_f:.1f} Hz")
    if max_coh > 0.25:
        print("      → STRONG mechanical coupling detected (microphonic artifact likely)")
    elif max_coh > 0.1:
        print("      → Moderate coupling — possible weak resonator effect or cable vibration")
    else:
        print("      → Very weak / no detectable linear coupling (good! mostly neural)")
    
    # Save numeric results
    np.savez(os.path.join(output_dir, "transfer_function.npz"),
             f=f, H_mean=H_mean, H_smooth=H_smooth, coherence=coherence,
             coh_f=coh_f, fs=fs_eeg, channel=channel, auto_lag_ms=auto_lag_ms)
    print(f"      Saved numeric data: {output_dir}/transfer_function.npz")
    
    print("\n" + "=" * 70)
    print("DONE. Open the PNG to inspect the resonance curve.")
    print("If you see clear peaks below 200 Hz with high coherence → head/cable acting as resonator.")
    print("Compare to your Prime-Log results — this is the PHYSICAL baseline.")
    print("=" * 70)

# ====================== CLI ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head-as-resonator transfer function from EDF + WAV")
    parser.add_argument("--edf", required=True, help="Path to EDF file")
    parser.add_argument("--wav", required=True, help="Path to synchronized WAV file")
    parser.add_argument("--channel", default=None, help="EEG channel (e.g. Cz). If omitted, auto-selects highest-variance channel")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to analyze from start (default 60)")
    parser.add_argument("--lag-ms", type=float, default=0.0, help="Fixed lag to apply (ms). 0 = physical expectation. Use auto value if >5ms")
    parser.add_argument("--output-dir", default="head_resonance_results", help="Where to save plots & data")
    args = parser.parse_args()
    
    analyze(args.edf, args.wav, channel=args.channel, duration=args.duration,
            lag_ms=args.lag_ms, output_dir=args.output_dir)