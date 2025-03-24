# Image Song Recommender

![Screenshot](static/uploads)  
*A beautiful UI for recommending songs based on an uploaded image.*

## Overview

Image Song Recommender is a Flask-based web application that recommends songs based on the mood and aesthetics extracted from an uploaded image. The app uses state-of-the-art image captioning (BLIP) and generative AI (Google Gemini) to refine the image description. It then compares this refined description against a precomputed song dataset (with artist, track, and semantic embeddings) to rank and recommend songs.

The application also provides interactive filters so that users can refine their recommendations by selecting specific languages and artists.

## Features

- **Image Processing**:  
  - Upload an image and automatically generate a detailed description using BLIP and Gemini.
  - Option to provide additional manual description to guide the recommendation process.

- **Song Recommendation**:  
  - Precomputed song data (with descriptions and embeddings) for fast recommendations.
  - Ranking based on cosine similarity between the image description and song embeddings.
  - Filter recommendations by language and artist.

- **User Interface**:  
  - A modern, responsive UI with a background gradient and interactive filters.
  - Animated spinner during recommendation processing.
  - Clear error messages and user-friendly feedback.

## Project Structure

image-song-recommender/ ├── app.py # Flask backend that processes image uploads and returns song recommendations. ├── description.py # Contains functions to process images and rank songs based on embeddings. ├── song_data.pkl # Precomputed song data generated via a separate precomputation notebook. ├── requirements.txt # List of required Python packages. ├── templates/ │ └── index.html # Main HTML page for the app. └── static/ ├── css/ │ └── styles.css # CSS styles for the UI. 

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/image-song-recommender.git
   cd image-song-recommender

2. **Create a Virtual Environment and Activate It**
   ```bash
   python -m venv venv
    source venv/bin/activate   # On macOS/Linux
    # or on Windows:
    # venv\Scripts\activate

3.**Install Dependencies**
  ```bash
    pip install -r requirements.txt
```
4.**Running the Application**

Once you have song_data.pkl and all dependencies installed, run the Flask app with:

```bash
python app.py
```
By default, the app runs in debug mode on http://127.0.0.1:5000. Open this URL in your browser to access the app.

Usage

Upload an Image:
On the homepage, click to upload an image.
Add Optional Description:
Enter additional description if desired.
Apply Filters:
Use the checkboxes to filter recommendations by language and/or artist.
Get Recommendations:
Click "Get Song Recommendations". An animated spinner will show while processing.
View Results:
Once processed, the refined image description and top song recommendations (artist and track) will be displayed.
Future Improvements

Responsive and Mobile Design: Further enhancements to ensure perfect display on all devices.
User Accounts & History: Save user preferences and recommendation history.
Audio Previews: Integrate with a music API (like Spotify) to play song previews.
Extended Filtering Options: Allow filtering by genre, mood, or release year.


Acknowledgments

BLIP: For powerful image captioning.

Google Gemini: For generative AI capabilities.

Spotify API & Lyrics.ovh: For music metadata and lyrics.

Sentence Transformers: For semantic embedding and ranking.




