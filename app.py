from flask import Flask, render_template, request, jsonify
import pickle
from description import rank_songs, process_image

app = Flask(__name__)

# Load precomputed song data once at startup
with open('song_data.pkl', 'rb') as f:
    precomputed_song_data = pickle.load(f)

# Mapping from artist name to language category based on provided list
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
    "Jaani": "Hindi",
    "Jaani": "Punjabi",
    "Ariana Grande": "English"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    # Check for the uploaded image file
    if 'photo' not in request.files:
        return jsonify({'error': 'Missing image file.'}), 400
    file = request.files['photo']
    if file.filename == "":
        return jsonify({'error': 'No selected file.'}), 400

    # Get the optional manual description from the form
    manual_description = request.form.get('manual_description', "").strip()
    
    # Get selected language filters and artist filters
    selected_languages = request.form.getlist('languages')
    selected_artists = request.form.getlist('artists')
    
    # Process the image using BLIP + Gemini (manual_description is appended if provided)
    refined_description = process_image(file, manual_description)
    
    # Filter the precomputed song data
    filtered_data = precomputed_song_data
    # Filter by language if specified
    if selected_languages:
        filtered_data = [
            song for song in filtered_data
            if artist_language.get(song['artist'], "Other") in selected_languages
        ]
    # Further filter by artist if specified
    if selected_artists:
        filtered_data = [
            song for song in filtered_data
            if song['artist'] in selected_artists
        ]
    
    # Rank songs using the refined description and filtered data.
    ranked = rank_songs(refined_description, filtered_data, top_n=5)
    recommendations = [{
        'artist': song['artist'],
        'track': song['track'],
        'similarity': float(sim)  # convert numpy float32 to native float
    } for song, sim in ranked]
    
    return jsonify({
        'refined_description': refined_description,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
