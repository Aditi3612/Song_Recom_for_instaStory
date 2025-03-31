import os
import pickle
from flask import Flask, render_template, request, jsonify
from description import rank_songs_with_sentiment, process_image, process_video

app = Flask(__name__)

# Load precomputed song data once at startup
with open('song_data.pkl', 'rb') as f:
    precomputed_song_data = pickle.load(f)

# Mapping from artist to language category
artist_language = {
    "Sachin-Jigar": "Hindi",
    "The Weeknd": "English",
    "Udit Narayan": "Hindi",
    "Atif Aslam": "Hindi",
    "Taylor Swift": "English",
    "Karan Aujla": "Punjabi",
    "Drake": "English",
    "Tanishk Bagchi": "Hindi",
    "Diljit Dosanjh": "Punjabi",
    "Masoom Sharma": "Haryanvi",
    "Bruno Mars": "English",
    "Vishal Mishra": "Hindi",
    "G. V. Prakash": "Tamil",
    "SZA": "English",
    "Sidhu Moose Wala": "Punjabi",
    "Billie Eilish": "English",
    "Rahat Fateh Ali Khan": "Hindi",
    "Lady Gaga": "English",
    "Darshan Raval": "Hindi",
    "Sachet Tandon": "Hindi",
    "Manoj Muntashir": "Hindi",
    "Pawan Singh": "Bhojpuri",
    "Gur Sidhu": "Punjabi",
    "Jimin": "English",
    "Arjan Dhillon": "Punjabi",
    "AP Dhillon": "Punjabi",
    "Javed Ali": "Hindi",
    "Justin Bieber": "English",
    "Lana Del Rey": "English",
    "Thaman S": "Telugu",
    "Cheema Y": "Punjabi",
    "Jaani": "Punjabi",
    "Ariana Grande": "English"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return jsonify({'error': 'Missing file.'}), 400
    file = request.files['photo']
    if file.filename == "":
        return jsonify({'error': 'No file selected.'}), 400

    manual_description = request.form.get('manual_description', "").strip()
    selected_languages = request.form.getlist('languages')
    selected_artists = request.form.getlist('artists')

    # Check file extension to decide if it's a video or an image.
    filename = file.filename.lower()
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    if any(filename.endswith(ext) for ext in video_extensions):
        refined_description = process_video(file, manual_description)
    else:
        refined_description = process_image(file, manual_description)
    
    # Filter precomputed song data based on selected filters.
    filtered_data = precomputed_song_data
    if selected_languages:
        filtered_data = [
            song for song in filtered_data
            if artist_language.get(song['artist'], "Other") in selected_languages
        ]
    if selected_artists:
        filtered_data = [
            song for song in filtered_data
            if song['artist'] in selected_artists
        ]
    
    # Rank songs using the composite ranking function.
    ranked = rank_songs_with_sentiment(refined_description, filtered_data, top_n=5, sentiment_weight=0.5)
    
    # Build recommendations (without Spotify integration)
    recommendations = [{
        'artist': song['artist'],
        'track': song['track'],
        'description': song.get('description', 'No description available.'),
        'similarity': float(sim)
    } for song, sim in ranked]
    
    return jsonify({
        'refined_description': refined_description,
        'recommendations': recommendations
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)  # Use a fixed port
