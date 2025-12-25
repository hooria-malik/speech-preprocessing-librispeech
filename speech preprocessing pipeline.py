import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
TARGET_SR = 16000
---

# 🟢 PART F — Open Terminal in Project Folder

### Option 1 (Easiest):
- Open **PyCharm**
- Bottom bar → **Terminal**

Make sure you see:


TOP_DB = 20

FILE_1 = "test-clean/LibriSpeech/test-clean/61/70968/61-70968-0000.flac"
FILE_2 = "test-clean/LibriSpeech/test-clean/61/70968/61-70968-0001.flac"

# ===================== PREPROCESSING =====================
def preprocess_audio(path, target_sr=16000, apply_filter=True):
    # Load audio using librosa (safe on Windows)
    audio, sr = librosa.load(path, sr=None, mono=True)

    # Resample
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Normalize
    audio = audio / np.max(np.abs(audio))

    raw_audio = audio.copy()

    # Silence trimming
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=20)

    # Optional high-pass filter (pre-emphasis)
    if apply_filter:
        trimmed_audio = librosa.effects.preemphasis(trimmed_audio)

    return raw_audio, trimmed_audio


# ===================== FEATURE EXTRACTION =====================
def compute_log_mel(audio, sr=16000):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        fmax=8000
    )
    return librosa.power_to_db(mel)


# ===================== NOISE AUGMENTATION =====================
def add_noise(audio, noise_level=0.02):
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise


# ===================== MAIN =====================
def main():
    print("Loading and preprocessing audio...")

    raw1, proc1 = preprocess_audio(FILE_1)
    raw2, proc2 = preprocess_audio(FILE_2)

    # -------- WAVEFORM VISUALIZATION --------
    print("Plotting waveforms...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))

    axes[0, 0].plot(raw1)
    axes[0, 0].set_title("Sample 1 – Raw Waveform")

    axes[0, 1].plot(proc1)
    axes[0, 1].set_title("Sample 1 – Processed Waveform")

    axes[1, 0].plot(raw2)
    axes[1, 0].set_title("Sample 2 – Raw Waveform")

    axes[1, 1].plot(proc2)
    axes[1, 1].set_title("Sample 2 – Processed Waveform")

    plt.tight_layout()
    plt.show()

    # -------- FEATURE EXTRACTION --------
    print("Extracting log-Mel spectrograms...")
    mel1 = compute_log_mel(proc1)
    mel2 = compute_log_mel(proc2)

    # -------- FEATURE VISUALIZATION --------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    librosa.display.specshow(
        mel1, sr=TARGET_SR, x_axis="time", y_axis="mel", ax=axes[0]
    )
    axes[0].set_title("Sample 1 – Log-Mel Spectrogram")

    librosa.display.specshow(
        mel2, sr=TARGET_SR, x_axis="time", y_axis="mel", ax=axes[1]
    )
    axes[1].set_title("Sample 2 – Log-Mel Spectrogram")

    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()

    # -------- COMBINE FEATURES INTO TENSOR --------
    mel1_t = torch.tensor(mel1)
    mel2_t = torch.tensor(mel2)

    max_len = max(mel1_t.shape[1], mel2_t.shape[1])

    def pad_mel(mel, max_len):
        return torch.nn.functional.pad(mel, (0, max_len - mel.shape[1]))

    mel1_padded = pad_mel(mel1_t, max_len)
    mel2_padded = pad_mel(mel2_t, max_len)

    features_tensor = torch.stack([mel1_padded, mel2_padded])

    print("Final feature tensor shape:", features_tensor.shape)

    # -------- BONUS: NOISE EXPERIMENT --------
    print("Running noise augmentation experiment...")
    noisy_audio = add_noise(proc1)

    mel_clean = compute_log_mel(proc1)
    mel_noisy = compute_log_mel(noisy_audio)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    librosa.display.specshow(
        mel_clean, sr=TARGET_SR, x_axis="time", y_axis="mel", ax=axes[0]
    )
    axes[0].set_title("Clean Log-Mel")

    librosa.display.specshow(
        mel_noisy, sr=TARGET_SR, x_axis="time", y_axis="mel", ax=axes[1]
    )
    axes[1].set_title("Noisy Log-Mel")

    plt.colorbar()
    plt.tight_layout()
    plt.show()

    print("Done!")


# ===================== RUN =====================
if __name__ == "__main__":
    main()
