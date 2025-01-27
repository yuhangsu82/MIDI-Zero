import pretty_midi
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm


def extract_note_times(midi_file):
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    note_times = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note_times.append(note.start)

    note_times = sorted(note_times)

    return note_times


def group_notes_by_frame(midi_file, source_type, time_resolution=0.05):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    frame_groups = []
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append(note)
    notes.sort(key=lambda x: x.start)
    current_frame = []

    # note_times = extract_note_times(midi_file)
    # time_resolution = dynamic_threshold(note_times)
    if source_type == 'performance':
        for note in notes:
            if not current_frame:
                current_frame.append(note)
            else:
                time_diff = note.start - current_frame[0].start
                if time_diff < time_resolution:
                    current_frame.append(note)
                else:
                    current_frame = sorted(set(current_frame), key=lambda x: x.pitch)
                    frame_groups.append(current_frame)
                    current_frame = [note]
        if current_frame:
            current_frame = sorted(set(current_frame), key=lambda x: x.pitch)
            frame_groups.append(current_frame)
    
    elif source_type == 'score':
        for note in notes:
            if not current_frame:
                current_frame.append(note)
            else:
                if note.start == current_frame[0].start:
                    current_frame.append(note)
                else:
                    current_frame = sorted(set(current_frame), key=lambda x: x.pitch)
                    frame_groups.append(current_frame)
                    current_frame = [note]
        if current_frame:
            current_frame = sorted(set(current_frame), key=lambda x: x.pitch)
            frame_groups.append(current_frame)

    return frame_groups


def split_dataset(midi_folder, train_dir_path, val_dir_path, train_ratio=0.8):
    midi_files = [os.path.join(midi_folder, f) for f in os.listdir(midi_folder) if f.endswith('.mid')]
    random.shuffle(midi_files)
    train_size = int(len(midi_files) * train_ratio)
    train_files = midi_files[:train_size]
    val_files = midi_files[train_size:]
    
    for train_file in train_files:
        train_path = os.path.join(train_dir_path, os.path.basename(train_file))
        os.system(f"cp \"{train_file}\" \"{train_path}\"")
    
    for val_file in val_files:
        val_path = os.path.join(val_dir_path, os.path.basename(val_file))
        os.system(f"cp \"{val_file}\" \"{val_path}\"")

    return train_files, val_files


def get_midi_files(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mid')]
    return files


def draw_midi_map(matrix, is_show=False, save_path=None):
    pitch_classes, time_steps = matrix.shape
    fig, ax = plt.subplots(figsize=(time_steps/2, 10))   
    ax.imshow(matrix, cmap='Greys', interpolation='nearest', aspect='auto', extent=[0, time_steps, pitch_classes, 0])
    ax.set_xlabel('time')
    ax.set_ylabel('pitch')
    ax.set_yticks(np.arange(0, pitch_classes, 5))
    ax.set_yticklabels(np.arange(0, pitch_classes, 5))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, time_steps + 1, 1))
    ax.set_xticklabels(np.arange(0, time_steps + 1, 1))

    plt.title('midi map')
    if save_path:
        plt.savefig(save_path)
    if is_show:
        plt.show()
    plt.close()


def frame2matrix(frame_groups, max_time_steps=64, pitch_classes=88):
    time_steps = min(max_time_steps, len(frame_groups))
    matrix = np.zeros((pitch_classes, time_steps))
    for i in range(time_steps):
        for note in frame_groups[i]:
            matrix[note.pitch - 21, i] = 1
    
    return matrix


def generate_query_clips(audio_dir, output_dir, query_length, total_queries=2000):
    """
    Extract fixed-length clips from audio files for query generation.

    Args:
        audio_dir (str): Path to the directory containing audio files (MP3 and WAV).
        output_dir (str): Path to save the extracted query clips.
        query_length (int): Length of each query clip in seconds.
        total_queries (int): Total number of query clips to generate.

    Returns:
        list: A list of dictionaries containing query information (clip path, original track label, length).
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3") or f.endswith(".wav")]
    if not audio_files:
        raise ValueError("No MP3 or WAV files found in the specified directory.")
    
    queries = []
    query_length_ms = query_length * 1000
    pbar = tqdm(total=total_queries, desc="Generating queries")

    # Process each audio file
    num_queries = 0
    while num_queries < total_queries:
        audio_file = random.choice(audio_files)

        audio_path = os.path.join(audio_dir, audio_file)
        # Load the audio file
        if audio_file.endswith(".mp3"):
            audio = AudioSegment.from_file(audio_path, format="mp3")
        elif audio_file.endswith(".wav"):
            audio = AudioSegment.from_file(audio_path, format="wav")
        else:
            continue

        audio_length = len(audio)  # Duration in milliseconds
        if audio_length < query_length_ms:
            continue

        start_time = random.randint(0, audio_length - query_length_ms)
        end_time = start_time + query_length_ms
        clip = audio[start_time:end_time]
        track_label = os.path.splitext(audio_file)[0]  # Use filename (without extension) as label
        clip_name = f"{track_label}_{start_time}_{end_time}.mp3"  # Always export in MP3 format
        clip_path = os.path.join(output_dir, clip_name)
        clip.export(clip_path, format="mp3")

        # Store query information
        queries.append({
            "clip_path": clip_path,
            "track_label": track_label,
            "length": query_length  # In seconds
        })

        num_queries += 1
        pbar.update(1)

    pbar.close()

    if not queries:
        raise ValueError("No suitable audio files found with sufficient length for the specified query length.")

    return queries


def calculate_metrics(retrieval_results, ground_truth, top_k=10):
    """
    Calculate the retrieval metrics for the queries.

    Args:
        retrieval_results: List containing retrieval results.
        ground_truth: List of true song IDs for each query.
        top_k (int): Number of top results to consider.

    Returns:
        Recall@1, Recall@k, mrr, nDCG
    """
    recall_at_1 = []
    recall_at_k = []
    average_precisions = []
    ndcgs = []

    for query_id, result in enumerate(retrieval_results):
        retrieval_ids = result[:top_k]
        true_id = ground_truth[query_id]

        # Compute Recall@1
        is_match = 1 if true_id == retrieval_ids[0] else 0
        # if true_id != retrieval_ids[0]:
        #     print(f"Query {true_id}: {retrieval_ids[0]} (miss)")
        recall_at_1.append(is_match)

        # Compute Recall@k
        relevant_retrieved = [1 if retrieval_id == true_id else 0 for retrieval_id in retrieval_ids]
        recall = sum(relevant_retrieved) / 1  # Ground truth contains only one relevant item
        recall_at_k.append(recall)

        # Compute Average Precision (AP)
        if true_id in retrieval_ids:
            rank = retrieval_ids.index(true_id) + 1
            ap = 1 / rank
            ndcg = 1 / np.log2(rank + 1)
        else:
            ap = 0
            ndcg = 0
        average_precisions.append(ap)
        ndcgs.append(ndcg)

    mean_recall_at_1 = np.mean(recall_at_1)
    mean_recall_at_k = np.mean(recall_at_k)
    mean_rr = np.mean(average_precisions)
    mean_ndcg = np.mean(ndcgs)

    print(f"Recall@1: {mean_recall_at_1:.4f}, Recall@{top_k}: {mean_recall_at_k:.4f}, mrr: {mean_rr:.4f}, nDCG: {mean_ndcg:.4f}")

    return mean_recall_at_1, mean_recall_at_k, mean_rr, mean_ndcg


if __name__ == '__main__':
    midi_file = ""
    frame_groups = group_notes_by_frame(midi_file, "performance", 0.1)[:50]
    matrix = frame2matrix(frame_groups)
    draw_midi_map(matrix, False, "./org.png")
