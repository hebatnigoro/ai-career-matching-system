# NLP Career Path Drift Detection (BERT Embedding, Pure Approach)

This project implements an end-to-end pipeline using pretrained multilingual BERT-style sentence embeddings to analyze CV texts against career profiles, compute semantic similarity via cosine similarity, detect potential career path drift, and recommend alternative careers. No TF-IDF or classic NLP methods are used.

Key properties:
- Pretrained multilingual sentence-transformer model (default: paraphrase-multilingual-MiniLM-L12-v2)
- Pure semantic approach with vector embeddings and cosine similarity
- Transparent, lightweight methodology suitable for S1 thesis

Architecture overview:
1) Preprocessing
   - Normalize whitespace and punctuation; avoid heavy transformations
   - Preserve semantics; minimal lowercase only for consistency
2) Encoding
   - Convert CVs and career profiles to embeddings via SentenceTransformer
   - L2-normalization to ensure cosine similarity stability
3) Similarity & Ranking
   - Compute cosine similarity between student CV embeddings and career embeddings
   - Rank careers per student by similarity
4) Drift Analysis
   - If declared interest exists: compare similarity to that interest vs best alternative
   - Categorize drift into Aligned / Minor Drift / Major Drift based on thresholds
5) Recommendation
   - Suggest top-k careers above minimum similarity threshold

Methodological notes (for thesis writing):
- Embedding Model: Pretrained sentence-transformer generates dense semantic vectors per text segment. Chosen model offers multilingual support (including Indonesian) with efficient inference for CPU.
- Similarity Metric: Cosine similarity measures angular proximity between vectors, representing semantic closeness; values in [−1, 1], typically [0, 1] for positive spaces.
- Drift Definition: Drift is operationalized as the relative advantage of an alternative career over the declared interest, combined with absolute suitability of the declared interest.
- Evaluation: Use descriptive analysis of similarity distributions, threshold-based categorization, and case study walkthroughs.

Usage
1) Install dependencies
   pip install -r requirements.txt

2) Run the pipeline on sample data
   python app.py --cv data/students.json --careers data/careers.json --topk 5 --min-sim 0.55

3) Optional: Specify model
   python app.py --model paraphrase-multilingual-MiniLM-L12-v2

Outputs
- Tabular console output per student: ranked careers, similarity scores
- Drift classification with rationale
- Recommended alternatives above threshold

Data format
- data/careers.json: list of { id, title, description, skills }
- data/students.json: list of { id, name, cv_text, declared_interest? }

Design choices & limitations
- No fine-tuning: leverages general-purpose embeddings to remain lightweight and reproducible.
- Explainability: thresholds and drift formula documented; outputs include rationale and scores.
- Limitations: Embeddings may miss domain-specific nuances; classification thresholds are heuristic.
