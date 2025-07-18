# job_matcher.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Load the job descriptions
def load_job_descriptions(path='data/job_descriptions.csv'): 
    df = pd.read_csv(path)
    df = df.dropna(subset=['Job Title', 'Job Description'])
    df = df.reset_index(drop=True)
    return df

# Generate BERT embeddings
def generate_embeddings(text_list):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & fast
    embeddings = model.encode(text_list, show_progress_bar=True)
    return embeddings

# Save embeddings to disk
def save_embeddings(embeddings, filename='data/job_embeddings.npy'):
    np.save(filename, embeddings)

# Master function
def preprocess_and_encode():
    df = load_job_descriptions()
    combined_text = df['Job Title'] + " - " + df['Job Description']
    embeddings = generate_embeddings(combined_text.tolist())
    save_embeddings(embeddings)
    print(f"[✓] Processed {len(df)} job descriptions.")
    return df, embeddings


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load saved job embeddings
def load_embeddings(path='data/job_embeddings.npy'):
    return np.load(path)

# Match a resume text to job descriptions
def get_top_matches(resume_text, job_df, job_embeddings, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embedding = model.encode([resume_text])

    similarities = cosine_similarity(resume_embedding, job_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    top_jobs = job_df.iloc[top_indices].copy()
    top_jobs['similarity'] = similarities[top_indices]
    return top_jobs
