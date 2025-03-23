from flask import Flask, render_template, request, jsonify
import pickle
from description import rank_songs, process_image

app = Flask(__name__)

# Load precomputed song data once at startup
with open('song_data.pkl', 'rb') as f:
    precomputed_song_data = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    # Expecting an image file under "photo"
    if 'photo' not in request.files:
        return jsonify({'error': 'Missing image file.'}), 400
    file = request.files['photo']
    if file.filename == "":
        return jsonify({'error': 'No selected file.'}), 400

    # Get optional manual description from the form
    manual_description = request.form.get('manual_description', "").strip()
    
    # Process the uploaded image to get a refined description (API key is hard-coded)
    refined_description = process_image(file, manual_description)
    
    # Use the refined description to rank songs from the precomputed data
    ranked = rank_songs(refined_description, precomputed_song_data, top_n=5)
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
