import os
from tqdm import tqdm
from collections import Counter, defaultdict
from utils import group_notes_by_frame
import pickle
import random
import numpy as np
import json
import matplotlib.pyplot as plt


def extract_frame_distributions(midi_dir, save_file_path, source_type="performance", time_resolution=0.05):
    frame_sequences = []
    midi_files = [os.path.join(midi_dir, filename) for filename in os.listdir(midi_dir) if filename.endswith('.mid') or filename.endswith('.midi')]

    for midi_file in tqdm(midi_files, desc="Analysing MIDI files"):
        frame_groups = group_notes_by_frame(midi_file, source_type, time_resolution)
        frame_groups = [[note.pitch for note in frame] for frame in frame_groups]     
        frame_sequences.append(frame_groups)

    frame_unigram_counts = Counter()
    frame_bigram_counts = Counter()
    for seq in frame_sequences:
        for i, chd in enumerate(seq):
            chd = tuple(chd)
            frame_unigram_counts[chd] += 1
            if i > 0:
                prev_chd = tuple(seq[i-1])
                frame_bigram_counts[(prev_chd, chd)] += 1

    total_frames = sum(frame_unigram_counts.values())
    frame_unigram = {chd: c/total_frames for chd, c in frame_unigram_counts.items()}

    prev_frame_counts = defaultdict(int)
    for (c1, c2), c in frame_bigram_counts.items():
        prev_frame_counts[c1] += c

    frame_bigram = defaultdict(list)
    for (c1, c2), c in frame_bigram_counts.items():
        frame_bigram[c1].append((c2, c/prev_frame_counts[c1]))

    with open(save_file_path, "wb") as f:
        pickle.dump((frame_unigram, frame_bigram), f)
        print(f"frame distributions cached in {save_file_path}")

    return frame_unigram, frame_bigram


def statistic_distributions(input_file, output_file=None):
    with open(input_file, 'r') as f:
        data = json.load(f)
    pitch_counter = Counter()
    note_count_counter = Counter()
    for _, notes in data.items():
        for time_step in notes:
            pitch_counter.update(time_step)
            note_count_counter[len(time_step)] += 1
    
    combined_data = {
        'pitch_distribution': dict(pitch_counter),
        'note_count_distribution': dict(note_count_counter)
    }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=4)

    return combined_data


def draw_distribution(distribution, save_path=None, title=None):
    plt.figure(figsize=(10, 6))
    plt.bar(distribution.keys(), distribution.values(), color='skyblue')
    plt.xlabel('Pitch Points')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=90)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


if __name__ == '__main__':
    midi_dir = "data/midi_files_dir"
    frame_dist_file = "./frame_dist.pkl"
    dur_dist_file = "./duration_dist.pkl"
    extract_frame_distributions(midi_dir, frame_dist_file, source_type="performance", time_resolution=0.05)
