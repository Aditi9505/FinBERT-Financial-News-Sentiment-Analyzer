import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, BertTokenizer
from torch.nn.functional import softmax
import os

# --- 0. Configuration and Setup ---

# This configuration MUST match your training setup.
# In your notebook, the mapping was {'neutral': 0, 'negative': 1, 'positive': 2}
# Adjust this dictionary if your training generated a different mapping.
SENTIMENT_DICT = {'neutral': 0, 'negative': 1, 'positive': 2}
LABEL_MAP_INVERSE = {v: k for k, v in SENTIMENT_DICT.items()}

# Ensure this matches the file name you downloaded
MODEL_PATH = "finetuned_finBERT_epoch_2.model" 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 1. Load Model and Tokenizer (Uses Streamlit Caching for speed) ---

@st.cache_resource
def load_finbert_artifacts():
    try:
        # Load Tokenizer
        tokenizer = BertTokenizer.from_pretrained(
            "ProsusAI/finbert",
            do_lower_case=True
        )
        
        # Load Model Architecture
        model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert",
            num_labels=len(SENTIMENT_DICT),
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Load Fine-tuned Weights
        # We map to the current device (CPU is safest for loading)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE)
        )
            
        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    except FileNotFoundError:
        st.error(f"Error: Model file not found at: {MODEL_PATH}. Please ensure it is in the same directory as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        st.stop()
        
tokenizer, model = load_finbert_artifacts()


# --- 2. Prediction Function ---

# Function now returns the predicted label, its confidence, and a dictionary of all probabilities
def get_sentiment_prediction(text_input):
    
    # Tokenize Input
    encoded_input = tokenizer.encode_plus(
        text_input,
        add_special_tokens=True,
        max_length=256, 
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move tensors to device
    input_ids = encoded_input['input_ids'].to(DEVICE)
    attention_mask = encoded_input['attention_mask'].to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Calculate probabilities from logits
        probabilities = softmax(logits, dim=-1).cpu().numpy()[0]
    
    # Determine the predicted label and confidence
    predicted_index = np.argmax(probabilities)
    predicted_label = LABEL_MAP_INVERSE[predicted_index]
    confidence = probabilities[predicted_index]

    # Create a dictionary of results to be displayed
    # Ensure order is consistent (e.g., sort keys by label name)
    results_dict = {
        LABEL_MAP_INVERSE[i]: probabilities[i] for i in range(len(SENTIMENT_DICT))
    }

    return predicted_label, confidence, results_dict


# --- 3. Streamlit UI Layout ---

st.set_page_config(page_title="FinBERT Sentiment Analyzer", layout="centered")
st.title("Financial News Sentiment Classifier 💰")
st.markdown("Analyze financial news headlines using a fine-tuned FinBERT model.")

# Text input widget
user_input = st.text_area(
    "Enter a financial headline or sentence:",
    height=120,
    placeholder="E.g., 'The unexpected resignation of the CEO caused the stock price to plunge.'"
)

# Initialize button click state
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False

# Function to handle analysis button click
def analyze_click():
    st.session_state.analyze_clicked = True

# Analyze button
st.button("Analyze Sentiment", on_click=analyze_click)

# Display results only after button is clicked and input is present
if st.session_state.analyze_clicked and user_input:
    # Get predictions
    label, confidence, all_probs = get_sentiment_prediction(user_input)

    st.subheader("Analysis Result:")
    
    # Set display color and markdown text
    if label == 'positive':
        color = 'green'
        st.markdown(f"Sentiment: :{color}[**{label.upper()}**]")
    elif label == 'negative':
        color = 'red'
        st.markdown(f"Sentiment: :{color}[**{label.upper()}**]")
    else:
        color = 'orange'
        st.markdown(f"Sentiment: :blue[**{label.upper()}**]") # Use blue for neutral/stable tone

    st.markdown("---")

    # --- Confidence Visualization (Bar Chart) ---

    st.subheader(f"Confidence Breakdown")

    # Create DataFrame for bar chart
    df_probs = pd.DataFrame(
        {'Sentiment': list(all_probs.keys()), 'Probability': list(all_probs.values())}
    )
    
    # Use st.bar_chart for quick visualization
    st.bar_chart(df_probs, x='Sentiment', y='Probability', color="#5A8DDC", use_container_width=True)


    # --- Final Metric Display ---
    
    st.metric(
        label=f"Highest Confidence Score",
        value=f"{confidence:.4f}",
        delta=f"Predicted Class: {label.upper()}"
    )

    st.caption(f"Note: Model is running on {DEVICE}.")

# Reset the click state if the user modifies the input box, forcing a re-click for analysis
if user_input != st.session_state.get('last_input', ''):
    st.session_state.analyze_clicked = False
st.session_state.last_input = user_input
