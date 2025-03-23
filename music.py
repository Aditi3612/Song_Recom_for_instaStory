import pickle

def load_precomputed_song_data(pickle_file='song_data.pkl'):
    """
    Load the precomputed song data from a pickle file.
    The data is a list of dictionaries, each containing:
      - 'artist'
      - 'track'
      - 'lyrics'
      - 'description'
      - 'embedding' (a NumPy array)
    """
    with open(pickle_file, 'rb') as f:
        song_data = pickle.load(f)
    return song_data

# You can now use load_precomputed_song_data() anywhere in your app
# instead of re-fetching and processing the data.
