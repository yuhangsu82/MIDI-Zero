import os
import random
from utils import group_notes_by_frame
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import numpy as np
import pickle
from data.data_generator import generate_random_duration, generate_training_data
import bisect


def preprocess_and_save_midi_data(midi_files, save_path, pitch_only=True, source_type='performance', time_resolution=0.05):
    """
    Preprocess MIDI files to group notes and save the results to a JSON file.

    Args:
        midi_files: List of paths to MIDI files.
        save_path: Path to save the preprocessed data.
        pitch_only: Boolean flag indicating whether to process only pitches.
        source_type: Type of MIDI source ('performance' or 'score').
        time_resolution: Time resolution for grouping notes.
    """
    processed_data = {}

    for midi_file in tqdm(midi_files, desc="Processing MIDI files"):
        try:       
            # Group notes by frame
            frame_groups = group_notes_by_frame(midi_file, source_type, time_resolution)
            if pitch_only:
                frame_groups = [[note.pitch for note in frame] for frame in frame_groups]
            else:
                frame_groups = [[(note.pitch, note.end - note.start) for note in frame] for frame in frame_groups]
            processed_data[os.path.basename(midi_file)] = frame_groups
        except OSError as e:
            print(f"Error reading MIDI file {midi_file}: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while reading {midi_file}: {e}")
            continue

    # Save the preprocessed data to a JSON file
    with open(save_path, 'w') as f:
        json.dump(processed_data, f)
    print(f"Preprocessed data saved to {save_path}")


def force_augment(sequence, pitch_only=True):
    """
    Forcefully apply an augmentation to ensure the sequence differs from the original.

    Args:
        sequence: List of frame groups (each group is a list of note pitches).
        pitch_only: Boolean flag indicating whether to process only pitches.

    Returns:
        augmented: Augmented sequence.
        ref: Similarity reference value for the augmented sequence.
    """
    augmented = sequence[:]
    ref = 1.0
    random_idx = random.randint(0, len(augmented) - 1)
    frame = augmented[random_idx][:]

    if len(frame) > 1 and random.random() < 0.5:
        frame.pop(random.randint(0, len(frame) - 1)) # Randomly drop a note if the frame has more than one note
        ref -= 0.05
    else:
        new_note = random.choice(range(21, 109)) # Randomly choose a new note
        if pitch_only:
            frame.append(new_note)
            frame = sorted(set(frame)) # Ensure no duplicates and sort
        else:
            frame.append([new_note, generate_random_duration()]) # Add a note with a random duration
            frame = sorted(set(tuple(note) for note in frame), key=lambda x: x[0])
        ref -= 0.05
    augmented[random_idx] = frame

    return augmented, ref


def preprocess_sequence(sequence, pitch_only=True, win_length=10, note_max=12, format='sequence'):
    """
    Preprocess a sequence of frame groups into fixed-length inputs with delimiters and padding.

    Args:
        sequence: List of frame groups. 
                  If pitch_only=True, each group is a list of pitches.
                  If pitch_only=False, each group is a list of [pitch, duration].
        pitch_only: Boolean flag indicating whether to process only pitches.
        win_length: Length of the window (number of frames). (How many frames in a sequence)
        note_max: Fixed length of each group (number of notes).
        format: Format of the output (either 'sequence' or 'matrix').

    Returns:
        If pitch_only=True:
            preprocessed_sequence: List of integers representing the tokenized pitch sequence.
        If pitch_only=False:
            pitch_sequence: List of integers representing the tokenized pitch sequence.
            duration_sequence: List of integers representing the tokenized duration sequence.
    """

    sequence = sequence[:win_length] + [[] for _ in range(max(0, win_length - len(sequence)))]

    if format == 'sequence':
        PAD_TOKEN = 88
        SEP_TOKEN = 89
        PAD_DURATION = 0 

        pitch_sequence, duration_sequence = [], []
        for group in sequence:
            if pitch_only:
                pitches = [pitch - 21 for pitch in group]
                pitches = pitches[:note_max] + [PAD_TOKEN] * max(0, note_max - len(pitches))
                pitch_sequence.extend(pitches)
                pitch_sequence.append(SEP_TOKEN)
            else:
                pitches = [item[0] - 21 for item in group]
                durations = [item[1] for item in group]
                pitches = pitches[:note_max] + [PAD_TOKEN] * max(0, note_max - len(pitches))
                durations = durations[:note_max] + [PAD_DURATION] * max(0, note_max - len(durations))
                pitch_sequence.extend(pitches)
                duration_sequence.extend(durations)
                pitch_sequence.append(SEP_TOKEN)
                duration_sequence.append(PAD_DURATION)

        pitch_sequence = pitch_sequence[:-1]
        duration_sequence = duration_sequence[:-1]

        return pitch_sequence, duration_sequence

    else: # Convert to matrix format
        pitch_dim = 88
        pitch_matrix = np.zeros((win_length, pitch_dim), dtype=np.int32)
        duration_matrix = np.zeros((win_length, pitch_dim), dtype=np.float32)
        if pitch_only:
            for i, group in enumerate(sequence):
                for pitch in group:
                    pitch_matrix[i][pitch - 21] = 1
        else:
            for i, group in enumerate(sequence):
                for item in group:
                    pitch_matrix[i][item[0] - 21] = 1
                    duration_matrix[i][item[0] - 21] = item[1]

        return pitch_matrix, duration_matrix


def add_notes(frame, num_notes=1, pitch_only=True):
    """Randomly add a specified number of notes to a frame."""
    for _ in range(num_notes):
        new_note = random.choice(range(21, 109))  # Add a note within the MIDI pitch range
        if pitch_only:
            frame.append(new_note)
        else:
            frame.append([new_note, generate_random_duration()])
    
    if pitch_only:
        frame = sorted(set(frame))
    else:
        frame = sorted(set(tuple(note) for note in frame), key=lambda x: x[0])
    return frame, -0.03 * num_notes


def remove_notes(frame, num_notes=1):
    """Randomly remove a specified number of notes from a frame, ensuring at least one note remains."""
    if len(frame) <= num_notes:
        num_notes = len(frame) - 1  # Ensure that at least one note remains
    for _ in range(num_notes):
        if frame:
            frame.pop(random.randint(0, len(frame) - 1))  # Randomly remove a note
    
    return frame, -0.03 * num_notes  # Return the frame and a penalty for similarity loss


def insert_frame(sequence, num_frames=1, pitch_only=True):
    """Randomly insert one or more frames into a sequence."""
    ref_penalty = -0.1 * num_frames  # Reference penalty for similarity loss
    for _ in range(num_frames):
        if pitch_only:
            new_frame = sorted(set([random.choice(range(21, 109)) for _ in range(random.randint(1, 3))]))
        else:
            new_frame = [[random.choice(range(21, 109)), generate_random_duration()] for _ in range(random.randint(1, 3))]
            new_frame = sorted(set(tuple(note) for note in new_frame), key=lambda x: x[0])
        insert_idx = random.randint(0, len(sequence))  # Randomly choose where to insert the new frame
        sequence.insert(insert_idx, new_frame)
    return sequence, ref_penalty


def delete_frame(sequence, num_frames=1):
    """Randomly delete one or more frames from the sequence."""
    ref_penalty = -0.1 * num_frames  # Reference penalty for similarity loss
    for _ in range(num_frames):
        if sequence:
            delete_idx = random.randint(0, len(sequence) - 1)  # Randomly choose a frame to delete
            sequence.pop(delete_idx)
    return sequence, ref_penalty


def apply_key_change(sequence, key_change, pitch_only=True):
    """Apply a key change to the sequence by shifting all notes by a specified amount."""
    new_seq = []
    ref_penalty = -0.1 * abs(key_change)
    # ref_penalty = =0.05 * abs(key_change)
    for frame in sequence:
        if pitch_only:
            new_frame = [(note + key_change) for note in frame]
        else:
            new_frame = [[note[0] + key_change, note[1]] for note in frame]
        # Ensure notes stay within valid MIDI range
        new_frame = [[min(max(note, 21), 108), duration] if not pitch_only else min(max(note, 21), 108) for note, duration in new_frame] if not pitch_only else [min(max(note, 21), 108) for note in new_frame]
        new_seq.append(new_frame)
    return new_seq, ref_penalty


def note_pitch_change(frame, pitch_change_range = [-2, -1, 1, 2], pitch_only=True):
    """Apply a pitch change to the frame by shifting one note."""
    pitch_change = random.choice(pitch_change_range)
    index = random.randint(0, len(frame) - 1)
    if pitch_only:
        frame[index] += pitch_change
        frame[index] = min(max(frame[index], 21), 108)
        frame = sorted(set(frame))
    else:
        frame[index][0] += pitch_change
        frame[index][0] = min(max(frame[index][0], 21), 108)
        frame = sorted(set(tuple(note) for note in frame), key=lambda x: x[0])

    return frame, -0.03 * abs(pitch_change)


def add_notes(frame, num_notes=1, pitch_only=True):
    """Randomly add a specified number of notes to a frame."""
    for _ in range(num_notes):
        new_note = random.choice(range(21, 109))  # Add a note within the MIDI pitch range
        if pitch_only:
            frame.append(new_note)
        else:
            frame.append([new_note, generate_random_duration()])
    
    if pitch_only:
        frame = sorted(set(frame))
    else:
        frame = sorted(set(tuple(note) for note in frame), key=lambda x: x[0])
    return frame, -0.03 * num_notes


def insert_duplicate_frame(sequence):
    """Randomly insert a duplicate frame into the sequence."""
    ref_penalty = -0.1  # Reference penalty for similarity loss
    insert_idx = random.randint(0, len(sequence) - 1)  # Randomly choose where to insert the new frame
    if insert_idx < len(sequence):
        sequence.insert(insert_idx, sequence[insert_idx])
    return sequence, ref_penalty


class MidiEmbeddingDataset(Dataset):
    """
    Dataset for MIDI sequences with augmentation and reference similarity values.
    Supports training and evaluation modes.
    """
    def __init__(self, data_path, win_length=10, note_max=12, step_size=5, fake_num=2000000, mode='train', data_source='R', pitch_only=True, frame_dist_file=None, data_format='sequence'):
        """
        Initialize the dataset.

        Args:
            data_path (str): Path to the preprocessed data.
            win_length (int): The length of the window. (How many frames in a sequence) (default: 10)
            note_max (int): Maximum number of notes in each group.
            step_size (int): Step size for sliding window to create sequences.
            mode (str): Mode of operation, either 'train' or 'eval'. Default is 'train'.
            data_source (str): Source of the data. (P: Probability-based, S: Statistical-based, C: Completely random, R: Real data)
            pitch_only (bool): Use pitch-only model if True. Default is True.
            dur_dist_file (str): Path to the duration distribution file. Default is None.
            data_format (str): Format of the data ('sequence' or 'matrix'). Default is 'sequence'.
        """
        self.win_length = win_length
        self.note_max = note_max
        self.step_size = step_size
        self.mode = mode
        self.pitch_only = pitch_only
        self.data_format = data_format
        self.frame_dist_file = frame_dist_file
        self.data = []
        self.file_ranges = []
        current_index = 0

        if data_source == 'R':
            # Load the real preprocessed data
            with open(data_path, 'r') as f:
                self.preprocessed_data = json.load(f)

            for file_name, frame_groups in self.preprocessed_data.items():
                start_index = current_index
                for start_idx in range(0, len(frame_groups) - win_length + 1, step_size):
                    end_idx = start_idx + win_length
                    org_slice = frame_groups[start_idx:end_idx]
                    self.data.append(org_slice)
                    current_index += 1

                # load data for the short query sequence when in eval mode
                if mode == 'eval' and len(frame_groups) < win_length:
                    self.data.append(frame_groups)
                    current_index += 1

                end_index = current_index - 1  # The last index added for this file
                self.file_ranges.append((file_name, start_index, end_index))
            
            self.start_indices = [start_idx for _, start_idx, _ in self.file_ranges]
            self.metadata = [(file_name, start_idx, end_idx) for file_name, start_idx, end_idx in self.file_ranges]
            print(f"Loaded {len(self.data)} sequences from {len(self.file_ranges)} files")
            
        else:
            if data_path:
                # Load the generated data
                with open(data_path, 'rb') as f:
                    self.data = pickle.load(f)
                print(f"Loaded {len(self.data)} sequences from {data_path}, please check the data source!")
            else:
                self.data = generate_training_data(data_source, frame_dist_file=self.frame_dist_file, pitch_only=self.pitch_only, num_samples=fake_num, note_max=note_max)


    def __len__(self):
        return len(self.data)


    def augment_sequence(self, sequence,
                         pitch_only=True, target_length=10,
                         add_note_prob=0.1, remove_note_prob=0.1, 
                         insert_frame_prob=0.5, delete_frame_prob=0.5, 
                         insert_duplicate_frame_prob=0, note_pitch_change_prob=0,
                         max_add_notes=3, max_remove_notes=2, max_duplicate_frames=1,
                         max_insert_frames=2, max_delete_frames=2,
                         key_change_prob=0, min_shift=-3, max_shift=3):
        augmented = [frame[:] for frame in sequence]
        ref = 1.0
        MAX_ATTEMPTS = 5  # Maximum number of attempts to ensure the augmented sequence is different

        for attempt in range(MAX_ATTEMPTS):
            # Add and remove notes in each frame based on probabilities
            for i, frame in enumerate(augmented):
                if random.random() < note_pitch_change_prob:
                    augmented[i], penalty = note_pitch_change(augmented[i], pitch_only=pitch_only)
                    ref += penalty

                if random.random() < add_note_prob:
                    num_notes = random.randint(1, max_add_notes)  # Randomly choose how many notes to add
                    augmented[i], penalty = add_notes(augmented[i], num_notes, pitch_only)
                    ref += penalty

                if random.random() < remove_note_prob and len(augmented[i]) > 1:
                    num_notes = random.randint(1, max_remove_notes)  # Randomly choose how many notes to remove
                    augmented[i], penalty = remove_notes(augmented[i], num_notes)
                    ref += penalty

            if random.random() < insert_duplicate_frame_prob:
                for _ in range(random.randint(1, max_duplicate_frames)):
                    augmented, penalty = insert_duplicate_frame(augmented)
                    ref += penalty

            if random.random() < insert_frame_prob:
                num_frames = random.randint(1, max_insert_frames)  # Randomly choose how many frames to insert
                augmented, penalty = insert_frame(augmented, num_frames, pitch_only)
                ref += penalty

            if random.random() < delete_frame_prob and len(augmented) > 1:
                num_frames = random.randint(1, max_delete_frames)  # Randomly choose how many frames to delete
                augmented, penalty = delete_frame(augmented, num_frames)
                ref += penalty

            if len(augmented) > target_length:
                augmented = augmented[:target_length]

            while len(augmented) < target_length:
                if pitch_only:
                    augmented.append([random.choice(range(21, 109))])  # Pad with random frames if the sequence is too short
                else:
                    augmented.append([[random.choice(range(21, 109)), generate_random_duration()]])

            # Employ key change based on probability
            if random.random() < key_change_prob:
                key_change = random.randint(min_shift, max_shift)
                augmented, penalty = apply_key_change(augmented, key_change, pitch_only)
                # ref += penalty

            if ref != 1.0:
                return augmented, ref
        
        return force_augment(sequence, pitch_only)


    def __getitem__(self, idx):
        """
        Retrieve the dataset item.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: In training mode, returns anchor, positive, and reference.
                   In evaluation mode, returns anchor.
        """
        org_slice = self.data[idx]
        anchor_pitch, anchor_duration = preprocess_sequence(org_slice, pitch_only=self.pitch_only, win_length=self.win_length, note_max=self.note_max, format=self.data_format)

        if self.mode == 'eval':
            if self.pitch_only:
                if self.data_format == 'matrix':
                    return np.float32(anchor_pitch)
                else:
                    return np.int32(anchor_pitch)
            else:
                if self.data_format == 'matrix':
                    return np.float32(anchor_pitch), np.float32(anchor_duration)
                else:
                    return np.int32(anchor_pitch), np.float32(anchor_duration)

        aug_slice, ref = self.augment_sequence(org_slice, pitch_only=self.pitch_only, target_length=self.win_length) # BeginnerPiano
        # aug_slice, ref = self.augment_sequence(org_slice, target_length=self.sequence_length, add_note_prob=0, remove_note_prob=0.05, insert_frame_prob=0.5, delete_frame_prob=0.5, max_add_notes=1, max_remove_notes=1, max_insert_frames=1, max_delete_frames=1, key_change_prob=0.5, min_shift=-3, max_shift=3) # guitarset
        positive_pitch, positive_duration = preprocess_sequence(aug_slice, pitch_only=self.pitch_only, win_length=self.win_length, note_max=self.note_max, format=self.data_format)

        if self.pitch_only:
            if self.data_format == 'matrix':
                return np.float32(anchor_pitch), np.float32(positive_pitch), np.float32(ref)
            else:
                return np.int32(anchor_pitch), np.int32(positive_pitch), np.float32(ref)
        else:
            if self.data_format == 'matrix':
                return np.float32(anchor_pitch), np.float32(positive_pitch), np.float32(anchor_duration), np.float32(positive_duration), np.float32(ref)
            else:
                return np.int32(anchor_pitch), np.int32(positive_pitch), np.float32(anchor_duration), np.float32(positive_duration), np.float32(ref)


    def get_file_name(self, anchor_id):
        """
        Retrieve the file path associated with a given anchor ID.

        Args:
            anchor_id (int): The ID of the anchor.

        Returns:
            str: The file path corresponding to the anchor ID.
        """
        idx = bisect.bisect_right(self.start_indices, anchor_id) - 1
        if idx >= 0:
            file_name, start_idx, end_idx = self.metadata[idx]
            if start_idx <= anchor_id <= end_idx:
                rel_pos = anchor_id - start_idx
                return file_name, rel_pos
        return None



if __name__ == "__main__":
    test_data = [[21], [42, 59, 21], [45, 21], [43, 99], [45], [99]]
    # test_data = [[[45, 0.02], [21, 0.14]], [[43, 2.2]], [[45, 0.02], [21, 0.14]], [[45, 0.12], [21, 0.14]]]
    aug_data, ref = insert_duplicate_frame(test_data)
    print(aug_data)
