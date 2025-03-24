from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load Sentence Transformer model once for ranking operations
st_model = SentenceTransformer('all-mpnet-base-v2')

def get_image_embedding(image_description):
    return st_model.encode(image_description)

def rank_songs(image_description, precomputed_song_data, top_n=5):
    """
    Computes the embedding for the image description and compares it with precomputed song embeddings.
    Returns the top_n song recommendations as a list of tuples: (song_dict, similarity_score).
    """
    image_embedding = get_image_embedding(image_description)
    similarities = []
    for song in precomputed_song_data:
        song_embedding = song['embedding']
        sim = cosine_similarity([image_embedding], [song_embedding])[0][0]
        similarities.append((song, sim))
    ranked = sorted(similarities, key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

def process_image(image_file, manual_description=None):
    """
    Processes an uploaded image file:
      1. Uses the BLIP model to generate an initial caption.
      2. If provided, appends a manual description.
      3. Uses Google Gemini (with a hard-coded API key) to refine the combined description.
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
    
    # Combine BLIP caption with the manual description if provided
    if manual_description:
        combined_prompt = f"Description: {initial_caption}. Additional details: {manual_description}"
    else:
        combined_prompt = f"Description: {initial_caption}"
    
    # Refine the combined description using Google Gemini
    gemini_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=(
            "You are an expert in analyzing visual content and emotions. "
            "For the given description, refine it into a more detailed and expressive description in 50-60 words, "
            "capturing the emotions, background, and key elements such as gestures, facial expressions, and environment. "
            "The description should evoke the mood and context to inform a musical recommendation. Keep the language vivid and precise."
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
