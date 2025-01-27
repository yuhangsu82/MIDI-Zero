import pretty_midi
import numpy as np
import os
import faiss
import Levenshtein
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import calculate_metrics
import pandas


def midi_to_bow(file_path, mod=12):
    """
    Convert a MIDI file to a Bag-of-Words representation.
    :param file_path: Path to the MIDI file
    :param mod: Modulo value (12 represents pitch classes, 24 considers octaves)
    :return: Normalized probability distribution (Bag-of-Words features)
    """
    midi_data = pretty_midi.PrettyMIDI(file_path)
    
    pitch_classes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitch_classes.append(note.pitch % mod)
    
    if len(pitch_classes) == 0:
        return np.zeros(mod)
    
    word_counts = np.zeros(mod)
    for pitch_class in pitch_classes:
        word_counts[pitch_class] += 1
    normalized_distribution = word_counts / np.linalg.norm(word_counts, ord=2)

    return np.float32(normalized_distribution)


def extract_melody(midi_file_path):
    """
    Extract melody notes from a MIDI file based on the given Melody Extraction Algorithm.
    
    :param midi_file_path: Path to the input MIDI file
    :return: List of melody notes as PrettyMIDI Note objects
    """
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Collect all notes from all instruments
    all_notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            all_notes.append(note)
    
    # Sort notes by pitch (descending) and start time (ascending)
    sorted_notes = sorted(all_notes, key=lambda n: (-n.pitch, n.start))
    melody_notes = []

    while sorted_notes:
        # Pick the note with the highest pitch
        note = sorted_notes.pop(0)

        # Check if the period of the note is fully covered by existing melody notes
        is_covered = False
        for melody_note in melody_notes:
            if melody_note.start <= note.start and melody_note.end >= note.end:
                is_covered = True
                break
        
        # If the note is not fully covered, process it
        if not is_covered:
            uncovered_periods = [(note.start, note.end)]  # Start with the entire note period
            
            # Find uncovered periods by checking overlap with existing melody notes
            for melody_note in melody_notes:
                new_uncovered_periods = []
                for start, end in uncovered_periods:
                    if end <= melody_note.start or start >= melody_note.end:
                        # No overlap, keep the period as is
                        new_uncovered_periods.append((start, end))
                    else:
                        # Partial overlap, split the uncovered period
                        if start < melody_note.start:
                            new_uncovered_periods.append((start, melody_note.start))
                        if end > melody_note.end:
                            new_uncovered_periods.append((melody_note.end, end))
                uncovered_periods = new_uncovered_periods

            # Create new notes for uncovered periods
            for start, end in uncovered_periods:
                if end > start:  # Ensure the period is valid
                    split_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=start,
                        end=end
                    )
                    melody_notes.append(split_note)

    melody_notes.sort(key=lambda n: n.start)  # Sort the melody notes by start time
    pitch_sequence = [note.pitch for note in sorted(melody_notes, key=lambda n: n.start)]

    # MRR pitch sequence to a single-character string
    pitch_to_char_MRR = {pitch: chr(33 + pitch) for pitch in range(128)}  # MRR MIDI pitch to printable ASCII
    pitch_sequence_str = "".join(pitch_to_char_MRR[pitch] for pitch in pitch_sequence)

    return pitch_sequence_str


def similarity(query: str, data: str, stride_length=0):
    query_len = len(query)
    data_len = len(data)

    if stride_length not in range(1, query_len + 1):
        stride_length=1

    s1 = query
    score = float("inf")
    start = -1
    end = -1

    while end < data_len:
        start = start + stride_length
        end=min(start + query_len, data_len)

        s2 = data[start : end]
        edit_distance = Levenshtein.distance(s1, s2)
        if edit_distance < score:
            score = edit_distance

    sim = 1 - score/query_len
    return sim


def search_algorithm(db_path, query_path, gt_file_path, top_k=50):
    gt = pandas.read_csv(gt_file_path)
    gt_dict = {}
    for i, query_name in enumerate(gt["query"]):
        gt_dict[query_name] = gt["gt"][i]


    # Step 1: Filtering using Bag-of-Words Representation
    dimension = 12  # Number of pitch classes
    db_names = os.listdir(db_path)
    db_bows = []
    for db_name in db_names:
        if db_name.endswith('.mid'):
            bow = midi_to_bow(os.path.join(db_path, db_name))
            db_bows.append(bow)

    db_bows = np.array(db_bows)
    index = faiss.IndexFlatL2(dimension) # We use the Faiss library to create the index, rather than using M-tree.
    index.add(db_bows)

    # We select the top-k most similar MIDI files based on the BoW representation, rather than filtering according to the threshold.
    query_names = os.listdir(query_path)
    bow_results = []
    for query_name in query_names:
        if query_name.endswith('.mid'):
            query_bow = midi_to_bow(os.path.join(query_path, query_name))
            D, I = index.search(query_bow.reshape(-1, dimension), top_k)
            R = [db_names[i] for i in I[0]]
            bow_results.append(R)

    # Step 2: Detailed search using Melody Representation
    melody_db = {}
    for db_name in db_names:
        melogy_seq = extract_melody(os.path.join(db_path, db_name))
        if len(melogy_seq) > 400: # Using the lev-400
            melogy_seq = melogy_seq[:200] + melogy_seq[-200:]
        melody_db[db_name] = melogy_seq

    results = []
    gts = []
    for i, query_name in tqdm(enumerate(query_names)):
        query_melody = extract_melody(os.path.join(query_path, query_name))
        if len(query_melody) > 400:
            query_melody = query_melody[:200] + query_melody[-200:]      
        melogy_scores = dict()
        for match in bow_results[i]:
            similarity_score = similarity(query_melody, melody_db[match]) # Using segmental edit distance to handle length differences
            melogy_scores[match] = similarity_score

        top_10_matches = [x[0] for x in sorted(melogy_scores.items(), key=lambda x: x[1], reverse=True)[:10]]
        query_label = gt_dict[query_name]
        gts.append(query_label)
        results.append(top_10_matches)
        # print(f"Query: {query_name}, GT: {query_label}, Top-10: {top_10_matches}")  

    recall_at_1, recall_at_10, MRR, nDCG = calculate_metrics(results, gts)

    return recall_at_1, recall_at_10, MRR, nDCG


if __name__ == '__main__':
    search_algorithm('/home/syh/code/Piano-Embedding/runs/database/beginnerPiano/db', '/home/syh/code/Piano-Embedding/runs/database/beginnerPiano/query', '/home/syh/code/Piano-Embedding/runs/database/beginnerPiano/query_gt.csv')