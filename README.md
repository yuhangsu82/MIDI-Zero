# MIDI-Zero

**MIDI-Zero** is a self-supervised learning framework for content-based music retrieval (CBMR) tasks, including audio identification, matching, and version identification. Operating entirely on MIDI representations, it eliminates the need for external training data by generating task-specific data automatically, offering a robust and scalable solution for bridging audio and symbolic music retrieval.

## **Method Overview**

![MIDI-Zero Method](https://raw.githubusercontent.com/yuhangsu82/MIDI-Zero/refs/heads/main/assets/f1.jpg?token=GHSAT0AAAAAAC56SEQGHW4SQQ53OUCCRB2CZ4YPQFQ)

### 1. **Input Format (MIDI):**
   - MIDI data, standardized from either digital scores or user performances, forms the input for this framework.

### 2. **MIDI Representation Processing:**
   - **Exact Alignment for Digital Scores:** The symbolic MIDI is directly aligned.
   - **Threshold-based Grouping for Transcribed MIDI:** Due to transcription variations, we apply a **threshold zone** to cluster notes within an acceptable range.

### 3. **Pitch-Matrix Construction:**
   - A **time ablation process** is applied to both digital and transcribed MIDI, converting them into **Pitch-Matrices**.
   - Each Pitch-Matrix has a fixed **window length (e.g., 10 frames)**, and they are extracted with a **step size of 2** to create overlapping segments.

### 4. **Pitch-Matrix Encoding:**
   - The segmented Pitch-Matrices are fed into a **Pitch-Matrix Encoder**(resnet-based), which transforms them into compact **embeddings**.
   - These embeddings are later used for CBMR tasks such as Audio Identification, Audio Matching, and Version Identification.


## Training Data Sources

This module provides the `generate_training_data` function to create training datasets for content-based music retrieval tasks. The function supports **four modes**, each corresponding to a specific data source.

---

### 1. **C Mode (Completely Random)**
- **Source**: Artificially generated data.  
- Notes in each frame are randomly sampled from a specified pitch range.  
- **Real Data Requirement**: Not required.  
- This mode simulates synthetic data for tasks requiring diverse and randomized inputs.  

---

### 2. **S Mode (Statistical-based)**
- **Source**: Predefined statistical distributions.  
- Data is generated based on unigram and bigram probabilities stored in a provided frame distribution file (`frame_dist_file`).  
- **Real Data Requirement**: Requires real-world data to precompute statistical distributions.  
- This mode leverages statistical patterns learned from real-world datasets to create more realistic training data.

---

### 3. **P Mode (Probability-based)**
- **Source**: Customizable probability distributions.  
- Notes and note counts are sampled using Normal distributions for pitch and frame complexity.  
- **Real Data Requirement**: Not required.  
- This mode provides a flexible approach to simulate data with specific statistical properties.

---

### 4. **R Mode (Real Data)**
- **Source**: Authentic MIDI files or pre-collected datasets.  
- **Real Data Requirement**: Requires real-world MIDI files or datasets.  
- This mode directly utilizes real-world music data, ensuring the highest level of authenticity for training and evaluation.

---

### Key Features
- **Diverse Sources**: Supports both synthetic and real-world data for comprehensive training coverage.
- **Customizable Parameters**: Allows users to fine-tune pitch range, window length, and distribution properties.
- **Efficient Generation**: Scales to generate large datasets quickly.
- **Save Capability**: Enables saving generated data locally for reuse.

---

### Example Usage

```python
from data.data_generator import generate_training_data

# Example: Generate data based on probability distributions
data = generate_training_data(
    mode="P",  # Use P mode for probability-based data generation
    win_length=10,  # Length of each sequence (number of frames)
    note_max=12,  # Maximum number of notes per frame
    pitch_range=(21, 108),  # Pitch range for the notes
    num_samples=10000,  # Number of training samples to generate
    pitch_only=True,  # Generate pitch-only data
    p_mean=64.0,  # Mean of the pitch Normal distribution
    p_std=24.0,  # Standard deviation of the pitch Normal distribution
    n_mean=6.0,  # Mean of the note count Normal distribution
    n_std=2.5,  # Standard deviation of the note count Normal distribution
    save_path="probability_based_data.pkl"  # Path to save the generated data
)
