# 🎵 Image-Based Song Recommender

<p align='center'>
<img src="static/uploads/Screenshot 2025-03-24 at 7.33.23 PM.png" width="450" height="330">
</p>

**A full-stack AI application that recommends songs based on the aesthetic and mood of an uploaded image.**

---

## 📖 Overview

This Flask web app uses computer vision and generative AI to recommend songs based on an uploaded image. It processes the image via **BLIP**, refines the description using **Google Gemini**, and matches it with a precomputed song dataset using **semantic embeddings**.

Users can refine results using **language** and **artist** filters.

---

## 🚀 Features

- 🎨 Image captioning with BLIP  
- 🤖 Description refinement via Google Gemini  
- 🧠 Semantic matching with cosine similarity  
- 🎧 Filter by language and artist  
- ⚡ Responsive UI with loading animations and user-friendly feedback  

---

## 🛠️ Tech Stack

| Frontend | Backend | AI & NLP | Tools |
|----------|---------|----------|-------|
| HTML/CSS | Flask   | BLIP, Gemini, SentenceTransformers | Python, Jupyter, Git |

---

## 🗂️ Project Structure

```
image-song-recommender/
│
├── app.py             # Flask server
├── description.py     # AI-based image-to-song logic
├── song_data.pkl      # Precomputed song data
├── requirements.txt   # Python dependencies
│
├── templates/
│   └── index.html     # Main HTML page
│
└── static/
    └── css/
        └── styles.css # Styling
```

---

## 💻 Setup Instructions

```bash
git clone https://github.com/Aditi3612/Song_Recom_for_instaStory.git
cd Song_Recom_for_instaStory

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## 📋 How to Use

1. Upload an image  
2. (Optional) Add your own description  
3. Apply filters for language or artist  
4. Click **Get Recommendations**  
5. View AI-generated description + song suggestions  

---

## 🔮 Future Improvements

- Mobile-responsive design  
- User login & saved history  
- Spotify preview integration  
- Filters for mood, genre, release year  

---

## 🙏 Acknowledgements

- [BLIP](https://github.com/salesforce/BLIP) – Image Captioning  
- [Google Gemini](https://deepmind.google/technologies/gemini/)  
- [Sentence Transformers](https://www.sbert.net/)  
- [Spotify API](https://developer.spotify.com/)  
- [Lyrics.ovh](https://lyricsovh.docs.apiary.io/)  

---

## 🧭 Application Flowchart

<p align="center">
  <img src="https://github.com/Aditi3612/Song_Recom_for_instaStory/blob/main/project_flowchart.png?raw=true" width="600">
</p>








