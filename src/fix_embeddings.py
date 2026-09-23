"""
Tool to fix embeddings file issues
"""

import os
import pickle
import numpy as np
import json
import traceback
from sentence_transformers import SentenceTransformer

def fix_embeddings():
    """Fix embeddings file issues by regenerating it if necessary"""
    # Always use the data directory for embeddings and data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    embeddings_file = os.path.join(data_dir, "embeddings.pkl")
    data_file = os.path.join(data_dir, "processed_data.json")
    model_name = "all-MiniLM-L6-v2"

    print("Checking embeddings file...")

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from {data_file}")
    except Exception as e:
        print(f"Error loading chunks: {str(e)}")
        return False

    needs_regeneration = False
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)

            # Verify the shape of embeddings
            if len(chunks) != embeddings.shape[0]:
                print(f"[Warning] Embeddings shape mismatch: {embeddings.shape[0]} embeddings for {len(chunks)} chunks")
                needs_regeneration = True
            else:
                print(f"[OK] Embeddings file is valid: {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
                return True
        except Exception as e:
            print(f"[Warning] Error loading embeddings file: {str(e)}")
            needs_regeneration = True
    else:
        print("[Notice] Embeddings file does not exist")
        needs_regeneration = True

    # Regenerate embeddings if needed
    if needs_regeneration:
        print("\n[Regenerating] Regenerating embeddings...")
        try:
            # Load the model
            print(f"[Loading] Loading embedding model: {model_name}")
            try:
                model = SentenceTransformer(model_name)
            except Exception:
                model = SentenceTransformer(model_name, trust_remote_code=True)
            print("[OK] Model loaded successfully!")

            # Generate embeddings
            print(f"[Processing] Generating embeddings for {len(chunks)} chunks...")
            texts = [chunk['text'] for chunk in chunks]

            batch_size = 32
            embeddings_list = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                batch_embeddings = model.encode(batch, show_progress_bar=True)
                embeddings_list.append(batch_embeddings)

            # Combine all embeddings
            embeddings = np.vstack(embeddings_list)

            # Save embeddings for future use
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)

            print(f"[OK] Generated and saved embeddings with shape: {embeddings.shape} to {embeddings_file}")
            return True
        except Exception as e:
            print(f"[Error] Error regenerating embeddings: {str(e)}")
            print(traceback.format_exc())
            return False

def main():
    """Main entry point for the script"""
    print("\n[Checking] Checking and fixing embeddings file if needed...")
    result = fix_embeddings()
    if result:
        print("\n[OK] Embeddings check/fix completed successfully!")
        return 0
    else:
        print("\n[Error] Embeddings check/fix failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
