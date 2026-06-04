# 📚 Book Recommender System

A book recommendation system built using NLP, semantic search, and a simple Gradio interface.

Instead of relying only on keywords, the application uses vector embeddings and ChromaDB to find books that are semantically similar to a user's query.

For example, a query like:

> "a dark fantasy with betrayal and political conflict"

can return relevant books even if those exact words do not appear in the title.

---

## Features

- Semantic book search using embeddings
- Vector database powered by ChromaDB
- Emotion-aware filtering
- Category-based filtering
- Interactive Gradio dashboard
- Preprocessed and cleaned book dataset
- Fast retrieval of relevant books

---

## Tech Stack

- Python
- Pandas
- NumPy
- LangChain
- Hugging Face Embeddings
- ChromaDB
- Gradio

---

## Project Structure

```text
Book Recommender/
│
├── app/
│   └── gradio-dashboard.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── data-exploration.ipynb
│   ├── vector-search.ipynb
│   ├── text-classification.ipynb
│   └── sentiment-analysis.ipynb
│
├── Chroma_DB/
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Book-Recommender-System
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Move into the app directory:

```bash
cd app
```

Run the dashboard:

```bash
python gradio-dashboard.py
```

A local Gradio URL will be generated in the terminal.

---

## How It Works

1. Book metadata is cleaned and processed.
2. Book descriptions are converted into vector embeddings.
3. Embeddings are stored in ChromaDB.
4. User queries are converted into embeddings.
5. Similarity search retrieves the most relevant books.
6. Results can be refined using category and emotion filters.

---

## Dataset

The project uses a book dataset containing:

- Title
- Author
- Description
- Categories
- Cover Images
- ISBN Information

The data is cleaned and enriched before being used for search and recommendation.

---

## Future Improvements

- User-based recommendations
- Hybrid recommendation system
- Better ranking strategies
- Deployment on a cloud platform
- Personalized recommendation history

---

## Author

Ayush Gairola
