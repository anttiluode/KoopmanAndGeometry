import numpy as np
import mne
import scipy.signal as signal
import soundfile as sf
from scipy.interpolate import interp1d

def render_eeg_to_audio(edf_path, npz_path, output_wav, duration=30, start_sec=0):
    # 1. Load the Calibration Data
    print(f"Loading resonator profile: {npz_path}")
    calib = np.load(npz_path)
    f_bins = calib['f']
    h_mag = calib['H_smooth']  # The smoothed transfer function
    fs_eeg = float(calib['fs'].item())
    target_ch = str(calib['channel'])

    # 2. Load the EEG Data
    print(f"Loading EEG channel {target_ch} from {edf_path}...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Select segment
    start_samp = int(start_sec * fs_eeg)
    stop_samp = start_samp + int(duration * fs_eeg)
    eeg_data = raw.get_data(picks=[target_ch], start=start_samp, stop=stop_samp)[0]
    
    # 3. Frequency Domain "De-muffling" (Inverse Filtering)
    # We want to boost the frequencies that the skull/cables attenuated
    print("Inverting the resonance curve...")
    
    # Create an interpolation function for the H(f) curve
    # We add a small epsilon to avoid division by zero
    inv_filter_curve = 1.0 / (h_mag + 1e-6)
    
    # Limit the boost (e.g., max 20dB) to prevent exploding noise
    inv_filter_curve = np.clip(inv_filter_curve, 0, np.percentile(inv_filter_curve, 95))
    
    interp_func = interp1d(f_bins, inv_filter_curve, bounds_error=False, fill_value="extrapolate")
    
    # Perform FFT on the EEG
    eeg_fft = np.fft.rfft(eeg_data)
    fft_freqs = np.fft.rfftfreq(len(eeg_data), 1/fs_eeg)
    
    # Apply the inverse weights
    weights = interp_func(fft_freqs)
    audio_fft = eeg_fft * weights
    
    # Back to time domain
    audio_sig = np.fft.irfft(audio_fft)
    
    # 4. Clean up the resulting Audio
    # Bandpass 80Hz - 500Hz (The "Vocal" band in the artifact)
    sos = signal.butter(10, [80, min(500, fs_eeg/2-1)], btype='band', fs=fs_eeg, output='sos')
    audio_sig = signal.sosfiltfilt(sos, audio_sig)
    
    # Normalize
    audio_sig = audio_sig / (np.max(np.abs(audio_sig)) + 1e-9)
    
    # 5. Save as WAV
    # Note: We keep the EEG sample rate for authenticity
    sf.write(output_wav, audio_sig, int(fs_eeg))
    print(f"Done! Rendered audio to: {output_wav}")

if __name__ == "__main__":
    render_eeg_to_audio(
        edf_path="sub-03_ses-20240821_task-speechopen_acq-pangolin_run-02_eeg.edf", 
        npz_path="output/transfer_function.npz", 
        output_wav="recovered_meat_voice.wav",
        duration=20, # seconds
        start_sec=5  # skip the first 5 seconds
    )