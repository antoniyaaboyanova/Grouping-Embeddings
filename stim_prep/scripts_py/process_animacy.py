import os
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import argparse


def is_animate(word):
    synsets = wn.synsets(word, pos=wn.NOUN)  # Restrict to nouns
    for synset in synsets:
        hypernyms = synset.hypernym_paths()
        for path in hypernyms:
            for hypernym in path:
                if "animal" in hypernym.name():
                    return True
    return False

def process_animacy(project_dir):
    """
    Processes the animacy of categories in fmri_train_stim.npy.

    Args:
        project_dir (str): Path to the project directory.
    """
    # Load stimulus names
    stim_file = os.path.join(project_dir, "files", "fmri_train_stim.npy")
    if not os.path.exists(stim_file):
        raise FileNotFoundError(f"Stimulus file not found: {stim_file}")

    fmri_stim = np.load(stim_file, allow_pickle=True)

    # Extract category names from file names
    category_names = [
        im.split(".")[0][:-4] for im in tqdm(fmri_stim, desc="Processing stimuli")
    ]

    # Determine animacy for each category
    animate_mask = [
        is_animate(cat) for cat in tqdm(category_names, desc="Checking animacy")
    ]

    # Save the resulting mask
    output_file = os.path.join(project_dir, "files", "animate_mask.npy")
    np.save(output_file, animate_mask)

    print(f"Animacy mask saved to: {output_file}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process animacy mask from stimuli.")
    parser.add_argument('--project_dir', type=str, default='/projects/crunchie/boyanova/EEG_Things/Grouping-Embeddings', help="Path to the project directory")
    args = parser.parse_args()

    try:
        process_animacy(args.project_dir)
    except Exception as e:
        print(f"Error: {e}")
