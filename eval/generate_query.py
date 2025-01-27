import librosa
import soundfile as sf
import random
import numpy as np
import os
from tqdm import tqdm

def generate_query(audio_path, query_dir_path, query_len=10,
                   time_stretch_prob=0, time_stretch_range=(0.75, 1.5),
                   pitch_shift_prob=0, pitch_shift_range=(-3, 3),
                   add_noise_prob=0, noise_level=0.001,
                   bg_noise_prob=0, bg_noise_files=None, max_bg_noise_level=0.2,
                   ir_prob=0, ir_files=None):
    """
    Generate an audio query with various augmentations, including random cut and fill.

    Args:
        audio_path (str): Path to the input audio file.
        query_path (str): Path to save the generated query.
        query_len (int): Length of the output query in seconds.
        time_stretch_prob (float): Probability of applying time stretching.
        time_stretch_range (tuple): Range of time stretch factors.
        pitch_shift_prob (float): Probability of applying pitch shifting.
        pitch_shift_range (tuple): Range of pitch shifting in semitones.
        add_noise_prob (float): Probability of adding Gaussian noise.
        noise_level (float): Standard deviation of Gaussian noise.
        bg_noise_prob (float): Probability of adding background noise.
        bg_noise_files (list): List of file paths for background noise.
        bg_noise_level (float): Scaling factor for background noise.
        ir_prob (float): Probability of applying impulse response (IR).
        ir_files (list): List of file paths for IR recordings.
    """
    audio, sr = librosa.load(audio_path, sr=None)

    # Time stretch
    if random.random() < time_stretch_prob:
        stretch_factor = random.uniform(*time_stretch_range)
        print(stretch_factor)
        if stretch_factor > 0.5:  # Avoid too much stretching
            audio = librosa.effects.time_stretch(y=audio, rate=stretch_factor)

    # Pitch shift
    if random.random() < pitch_shift_prob:
        n_steps = random.randint(*pitch_shift_range)
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

    # Extract query-length segment from audio
    start = 0
    query_samples = int(query_len * sr)  # Total samples for query
    if len(audio) > query_samples:
        start = random.randint(0, len(audio) - query_samples)
        query_audio = audio[start: start + query_samples]
    else:
        query_audio = audio
    end = min(start + query_samples, len(audio))

    # Add Gaussian noise
    if random.random() < add_noise_prob:
        noise = np.random.normal(0, noise_level, len(query_audio))
        query_audio += noise

    # Add background noise
    if bg_noise_files and random.random() < bg_noise_prob:
        bg_noise_file = random.choice(bg_noise_files)
        bg_noise, _ = librosa.load(bg_noise_file, sr=sr, duration=query_len)  # Match query length
        if len(bg_noise) < len(query_audio):
            bg_noise = np.tile(bg_noise, int(np.ceil(len(query_audio) / len(bg_noise))))[:len(query_audio)]
        bg_noise_level = random.uniform(0, max_bg_noise_level)
        query_audio += bg_noise * bg_noise_level

    # Apply impulse response (IR)
    if ir_files and random.random() < ir_prob:
        ir_file = random.choice(ir_files)
        ir, _ = librosa.load(ir_file, sr=sr, duration=2.0)  # Use a short IR for efficiency
        if len(ir) > 1:  # Ensure IR is not empty
            ir = ir / np.max(np.abs(ir))  # Normalize IR
            query_audio = np.convolve(query_audio, ir, mode='full')[:len(query_audio)]

    # Save the processed audio
    query_name = os.path.basename(audio_path)[:-4] + f"_{start}_{end}.mp3"
    query_save_path = os.path.join(query_dir_path, query_name)
    sf.write(query_save_path, query_audio, sr)


def batch_generate_query(audio_dir_path, query_dir_path, query_len=10, query_num=1000, bg_path=None, ir_path=None):
    """
    Batch generate queries for a list of audio files.

    Args:
        audio_dir_path (str): Path to the directory containing audio files.
        query_dir_path (str): Path to save the generated queries.
        query_len (int): Length of the output query in seconds.
        query_num (int): Number of queries to generate for each audio file.
        bg_path (str): Path to the directory containing background noise files.
        ir_path (str): Path to the directory containing impulse response (IR) files
    """
    audio_paths = [os.path.join(root, file) for root, dirs, files in os.walk(audio_dir_path) for file in files if file.endswith('.wav')]
    if bg_path:
        bg_files = [os.path.join(root, file) for root, dirs, files in os.walk(bg_path) for file in files if file.endswith('.wav')]
    if ir_path:
        ir_files = [os.path.join(root, file) for root, dirs, files in os.walk(ir_path) for file in files if file.endswith('.wav')]
    
    if not os.path.exists(query_dir_path):
        os.makedirs(query_dir_path)
    else:
        print(f"Directory {query_dir_path} already exists. Exiting.")
        exit()

    for i in tqdm(range(query_num), desc="Generating queries"):
        source_audio = random.choice(audio_paths)
        generate_query(source_audio, query_dir_path, query_len=query_len)
        # generate_query(source_audio, query_dir_path, query_len=query_len, bg_noise_prob=0.5, bg_noise_files=bg_files)
        # generate_query(source_audio, query_dir_path, query_len=query_len, pitch_shift_prob=1.0, bg_noise_prob=0.5)


if __name__ == '__main__':
    # bg_path="./data/aug/bg"
    # ir_path="./data/aug/ir"
    batch_generate_query("./dataset_db", "./dataset_query", query_len=10, query_num=2000)