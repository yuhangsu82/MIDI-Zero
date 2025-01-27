import random
import pickle
import numpy as np
from tqdm import tqdm


def generate_training_data(mode, win_length=10, note_max=12, pitch_range=(21, 108), num_samples=2000000, pitch_only=True, dur_dist=None, save_path=None, **kwargs):
    """
    Generate fake data for training.

    Args:
        mode (str): The mode of generation. (P: Probability-based, S: Statistical-based, C: Completely Random)
        win_length (int): The length of the window. (How many frames in a sequence) (default: 10)
        note_max (int): The maximum number of notes in a frame. (default: 12)
        pitch_range (tuple): The pitch range of the notes. (default: (21, 108))
        num_samples (int): The number of samples to generate. (default: 2000000)
        pitch_only (bool): Whether to generate pitch only. (default: True)
        dur_dist (object): The distribution model for the duration. (default: None)
        save_path (str): The path to save the generated data. (default: None)
        **kwargs: Parameters for the generation.
    
    Returns:
        list: A list of generated data.
    """

    training_data = []    
    if mode == "C":
        print("Mode C: Completely Random")
        for _ in tqdm(range(num_samples), desc="Generating training data"):
            sequence = []
            for _ in range(win_length):
                num_notes = random.randint(1, note_max)
                if pitch_only:
                    notes = sorted(random.sample(range(pitch_range[0], pitch_range[1] + 1), num_notes))
                else:
                    notes = [[pitch, generate_random_duration()] for pitch in sorted(random.sample(range(pitch_range[0], pitch_range[1] + 1), num_notes))]
                sequence.append(notes)
            training_data.append(sequence)

    elif mode == "S":
        print("Mode S: Statistical-based")
        frame_dist_file = kwargs.get("frame_dist_file")
        with open(frame_dist_file, 'rb') as f:
            frame_unigram, frame_bigram = pickle.load(f)
        u_keys = list(frame_unigram.keys())
        u_probs = list(frame_unigram.values())
        unigram_dist = {k:p for k,p in zip(u_keys,u_probs)}
        for _ in tqdm(range(num_samples), desc="Generating training data"):
            current_frame = weighted_choice(unigram_dist)
            sequence = [current_frame]
            for _ in range(win_length - 1):
                if current_frame in frame_bigram:
                    next_candidates = frame_bigram[current_frame]
                    next_frame = weighted_choice(next_candidates)
                else:
                    next_frame = weighted_choice(unigram_dist)
                sequence.append(next_frame)
                current_frame = next_frame
            sequence = [list(item) for item in sequence]
            if pitch_only == False:
                for i in range(len(sequence)):
                    for j in range(len(sequence[i])):
                        sequence[i][j] = [sequence[i][j], generate_random_duration()]
            training_data.append(sequence)

    elif mode == "P":
        print("Mode P: Probability-based")
        p_mean = kwargs.get("p_mean", 64.0)
        p_std = kwargs.get("p_std", 24.0)
        n_mean = kwargs.get("n_mean", 0.0)
        n_std = kwargs.get("n_std", 2.5)
        pitch_probabilities = {pitch: normal_pdf(pitch, p_mean, p_std) for pitch in list(range(pitch_range[0], pitch_range[1] + 1))}
        note_count_probabilities = {note_count: normal_pdf(note_count, n_mean, n_std, 2) for note_count in list(range(1, note_max + 1))}
        for _ in tqdm(range(num_samples), desc="Generating training data"):
            sequence = []
            for _ in range(win_length):
                note_count = random.choices(list(note_count_probabilities.keys()), weights=note_count_probabilities.values())[0]
                notes = random.choices(list(pitch_probabilities.keys()), weights=pitch_probabilities.values(), k=note_count)
                sequence.append(notes)
            if pitch_only == False:
                for i in range(len(sequence)):
                    for j in range(len(sequence[i])):
                        sequence[i][j] = [sequence[i][j], generate_random_duration()]
            training_data.append(sequence)

    else:
        raise ValueError("Invalid mode!")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(training_data, f)
        print(f"Fake data saved to {save_path}")
    
    return training_data


def normal_pdf(x, mu, sigma, weight=1):
    return (weight / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu)**2 / (2 * sigma**2))


def weighted_choice(dist):
    r = random.random()
    cumulative = 0
    for item, prob in (dist.items() if isinstance(dist, dict) else dist):
        cumulative += prob
        if r <= cumulative:
            return item

    return list(dist.keys())[-1] if isinstance(dist, dict) else dist[-1][0]


def generate_random_duration():
    return random.uniform(0.01, 2.0)


def generate_singing_data(num_samples=1000000, group_length=10, save_path=None):
    note_count = 1
    pitch_range = list(range(21, 109))
    p_mean_first = 62.0
    p_std_first = 9.0
    pitch_probabilities = {pitch: normal_pdf(pitch, p_mean_first, p_std_first) for pitch in pitch_range}

    fake_data = []
    for _ in tqdm(range(num_samples)):
        sequence = []
        p_start = random.choices(list(pitch_probabilities.keys()), weights=pitch_probabilities.values(), k=note_count)
        for _ in range(group_length):
            p_pre = sequence[-1] if len(sequence) > 0 else p_start
            p_temp_dist = {pitch: normal_pdf(pitch, p_pre[0], 2.0) for pitch in pitch_range}
            time_step = random.choices(list(p_temp_dist.keys()), weights=p_temp_dist.values(), k=note_count)
            sequence.append(time_step)
        fake_data.append(sequence)

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(fake_data, f)
        print(f"Fake singing data saved to {save_path}")

    return fake_data


if __name__ == "__main__":
    # generate_training_data("C", save_path="data/fake_data_C.pkl")
    # generate_training_data("S", frame_dist_file="data/frame_dist.pkl", save_path="data/fake_data_S.pkl")
    dur_path = "/home/syh/code/Piano-Embedding/data_distribution/giantpiano_dur_distribution.pkl"
    with open(dur_path, 'rb') as f:
        dur_dist = pickle.load(f)

    # generate_training_data("P", save_path="data/fake_data_P.pkl", dur_dist=dur_dist, pitch_only=False)
    # generate_singing_data(save_path="./fake_singing_data.pkl")

    for _ in range(10):
        a = generate_random_duration()
        print(a)
    # print(dur_dist)