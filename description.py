import pickle
import math
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Load Sentence Transformer model for initial embedding
st_model = SentenceTransformer('all-mpnet-base-v2')
# Load a cross-encoder for refined semantic similarity
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# Initialize sentiment analysis pipeline (works well for English)
sentiment_analyzer = pipeline("sentiment-analysis")

def get_image_embedding(image_description):
    return st_model.encode(image_description)

def get_sentiment_score(text):
    """
    Returns a normalized sentiment score between -1 and 1.
    The sentiment pipeline returns a score between 0 and 1.
    We convert it to a range where negative sentiments become negative values.
    Text is truncated to 512 tokens to avoid errors.
    """
    if not text.strip():
        return 0.0

    # Enable truncation so the text does not exceed the model's max length.
    result = sentiment_analyzer(text, truncation=True, max_length=512)[0]
    score = result['score']
    return score if result['label'] == 'POSITIVE' else -score


def composite_similarity(image_desc, song_desc, semantic_score, sentiment_weight=0.5):
    """
    Computes a composite similarity score that combines semantic similarity and sentiment alignment.
    - semantic_score: the refined similarity score from the cross-encoder (can be negative).
    - sentiment similarity: computed as 1 - absolute difference between sentiment scores, so that
      a smaller difference gives a value closer to 1.
    sentiment_weight controls the relative importance of sentiment similarity.
    """
    image_sentiment = get_sentiment_score(image_desc)
    song_sentiment = get_sentiment_score(song_desc)
    # Map difference to a similarity (if difference is 0, similarity is 1; difference of 2 yields -1, so we clip)
    sentiment_similarity = max(0, 1 - abs(image_sentiment - song_sentiment))
    
    # Normalize the semantic score using sigmoid to bring it roughly into [0,1]
    semantic_norm = 1 / (1 + math.exp(-semantic_score))
    
    # Composite score: weighted combination of normalized semantic score and sentiment similarity.
    composite_score = (1 - sentiment_weight) * semantic_norm + sentiment_weight * sentiment_similarity
    return composite_score

def rank_songs_with_sentiment(image_description, precomputed_song_data, top_n=5, sentiment_weight=0.5):
    """
    Ranks songs using a combination of refined semantic similarity (via cross-encoder) 
    and sentiment similarity between the image description and the song's stored description.
    
    Process:
    1. Compute cosine similarity between the image description embedding and song embeddings.
    2. Select top candidates (e.g. top 10) based on cosine similarity.
    3. Re-rank these candidates using a cross-encoder to obtain refined semantic scores.
    4. Compute sentiment similarity between the image description and each song's description.
    5. Combine both signals into a composite score and sort.
    """
    # Step 1: Compute cosine similarity for initial ranking
    image_embedding = get_image_embedding(image_description)
    cosine_scores = []
    for song in precomputed_song_data:
        song_embedding = song['embedding']
        sim = cosine_similarity([image_embedding], [song_embedding])[0][0]
        cosine_scores.append((song, sim))
    
    # Sort by cosine similarity and select top candidates (e.g., top 10)
    cosine_ranked = sorted(cosine_scores, key=lambda x: x[1], reverse=True)
    top_candidates = cosine_ranked[:min(10, len(cosine_ranked))]
    
    # Step 2: Use cross-encoder to refine semantic scores on the top candidates
    cross_input = [[image_description, candidate[0].get('description', '')] for candidate in top_candidates]
    cross_scores = cross_encoder.predict(cross_input)
    
    # Step 3: Compute composite score by combining semantic and sentiment similarity
    composite_candidates = []
    for i, candidate in enumerate(top_candidates):
        song = candidate[0]
        semantic_score = cross_scores[i]
        song_description = song.get('description', '')
        comp_score = composite_similarity(image_description, song_description, semantic_score, sentiment_weight)
        composite_candidates.append((song, comp_score))
    
    # Step 4: Sort by composite score and return the top N songs
    final_ranked = sorted(composite_candidates, key=lambda x: x[1], reverse=True)
    return final_ranked[:top_n]

def process_image(image_file, manual_description=None):
    """
    Processes an uploaded image file:
     1. Uses the BLIP model to generate an initial caption.
     2. Optionally appends a manual description.
     3. Uses Google Gemini to refine the combined description.
    Returns the refined description.
    """
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    import google.generativeai as genai
    # Hard-coded Gemini API key
    GEMINI_API_KEY = 'AIzaSyC2KQPEjT-RDGoQwFJW2pgryK7gjr_ueqo'
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Open image and convert to RGB
    image = Image.open(image_file).convert('RGB')
    
    # Generate initial caption using BLIP
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(image, return_tensors="pt")
    out = model_blip.generate(**inputs)
    initial_caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Combine BLIP caption with manual description if provided
    if manual_description:
        combined_prompt = f"Description: {initial_caption}. Additional details: {manual_description}"
    else:
        combined_prompt = f"Description: {initial_caption}"
    
    # Refine the combined description using Google Gemini
    gemini_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction= (
            "You are an expert in analyzing visual content and emotions. Given the description of an image, "
            "provide a single, detailed, and expressive description that captures the overall mood, background, "
            "and key visual elements (such as gestures, facial expressions, and environment). "
            "The response should be unified and vivid, description should be in very detail"
        )
    )
    response = gemini_model.generate_content([combined_prompt])
    refined_description = None
    try:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content'):
            if isinstance(candidate.content, str):
                refined_description = candidate.content
            elif hasattr(candidate.content, 'parts'):
                refined_description = candidate.content.parts[0].text
            else:
                refined_description = str(candidate.content)
    except Exception:
        refined_description = initial_caption  # fallback if Gemini fails
    return refined_description

