# Book Recommender System

A semantic book recommender that finds books by meaning rather than keywords, with optional filtering by category and ranking by emotional tone, served through a Gradio web interface.

A query such as:

> "a dark fantasy with betrayal and political conflict"

is embedded into a vector and compared against embeddings of about 5,200 book descriptions stored in ChromaDB, so relevant books are returned even when the exact words do not appear in their titles or descriptions.

## Features

- Semantic search over book descriptions using sentence embeddings (`all-MiniLM-L6-v2`)
- Persistent vector store built with ChromaDB through LangChain
- Category filter (Fiction, Nonfiction, Children's Fiction, Children's Nonfiction), with missing categories filled in by zero-shot classification
- Emotion-based ranking (joy, sadness, fear, anger, disgust, surprise, neutral) using a fine-tuned emotion classifier
- Gradio dashboard with a cover-image gallery and a detail panel for each selected book

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.11+ |
| Data processing | pandas, NumPy |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` |
| Vector store | ChromaDB via `langchain-chroma` |
| Category classification | `facebook/bart-large-mnli` (zero-shot) via Hugging Face Transformers |
| Emotion classification | `j-hartmann/emotion-english-distilroberta-base` via Hugging Face Transformers |
| User interface | Gradio |

## Project Structure

```text
Book-Recommender-System/
├── app/
│   └── gradio-dashboard.py          # Gradio application
├── data/
│   ├── raw/
│   │   └── books.csv                # Original dataset (6,810 books)
│   └── processed/
│       ├── books_cleaned.csv        # After cleaning (5,197 books)
│       ├── books_with_categories.csv
│       ├── books_with_emotions.csv  # Final table used by the app
│       └── tagged_description.txt   # "<isbn13> <description>" lines used for embedding
├── notebooks/
│   ├── data-exploration.ipynb       # Cleaning and feature preparation
│   ├── text-classification.ipynb    # Category mapping and zero-shot classification
│   ├── sentiment-analysis.ipynb     # Emotion scoring
│   └── vector-search.ipynb          # Embedding and ChromaDB index creation
├── Chroma_DB/                       # Prebuilt vector index loaded by the app
├── requirements.txt
└── README.md
```

## Installation

Prerequisites: Python 3.11 or newer, and Git.

1. Clone the repository:

   ```bash
   git clone https://github.com/ayushcpp/Book-Recommender-System.git
   cd Book-Recommender-System
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv

   # Linux / macOS
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   On Linux, pip installs the CUDA build of PyTorch by default, which is several GB. If you do not have an NVIDIA GPU, install the smaller CPU-only build first, then install the rest:

   ```bash
   pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

## Usage

Run the dashboard from inside the `app/` directory, because the data and index paths in the script are relative to it:

```bash
cd app
python gradio-dashboard.py
```

The app loads the processed dataset and the prebuilt index in `Chroma_DB/`, then opens the Gradio interface in your browser. On the first run, the embedding model is downloaded from Hugging Face, so an internet connection is required.

In the interface:

1. Describe the kind of book you want in the text box.
2. Optionally choose a category and an emotion.
3. Click **Find Books** to see up to 16 recommendations.
4. Click a cover to view the full description.

## How It Works

The pipeline is built in the notebooks and consumed by the app.

1. **Cleaning** (`data-exploration.ipynb`): books missing a description, page count, rating or publication year are removed, as are books whose description has fewer than 25 words. This reduces the dataset from 6,810 to 5,197 books. Each description is prefixed with its ISBN-13 so search results can be mapped back to the book table.
2. **Category labels** (`text-classification.ipynb`): the most frequent original categories are mapped to four simple categories. Books that remain unlabelled (1,454) are classified as Fiction or Nonfiction with zero-shot classification using `facebook/bart-large-mnli`.
3. **Emotion scores** (`sentiment-analysis.ipynb`): each description is split into sentences, every sentence is scored by `j-hartmann/emotion-english-distilroberta-base`, and the maximum score per emotion across sentences is stored as seven columns.
4. **Vector index** (`vector-search.ipynb`): the tagged descriptions are embedded with `all-MiniLM-L6-v2` and persisted to ChromaDB.
5. **Recommendation** (`app/gradio-dashboard.py`): the query is embedded and the 50 nearest descriptions are retrieved. Their ISBNs are used to look up the books, which are kept in order of similarity, filtered by the selected category and, if an emotion is chosen, sorted by that emotion's score. The top 16 are displayed.

To rebuild the processed data or the index from scratch, run the notebooks in the order listed above.

## Results

The zero-shot category classifier was checked on 600 books whose category was already known (300 Fiction, 300 Nonfiction) and labelled 77.8% of them correctly. The recommendations themselves have not yet been evaluated with a quantitative metric.

## Limitations

- About one in five zero-shot category labels is expected to be wrong, based on the accuracy above, so the category filter is approximate.
- Filtering happens after retrieving 50 candidates, so a narrow category can return fewer than 16 books.
- The emotion option re-orders results rather than removing books that do not match the emotion. It sorts purely by emotion score, so a less similar book can rank above a closer match.
- Different editions of the same book have separate ISBNs, so the same title can appear more than once in the results.
- The catalogue is small (about 5,200 titles), so niche queries may return loosely related books.

## Dataset

The raw data is the [7k Books](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) dataset by Dylan Castillo on Kaggle, released under CC0 (public domain). It combines ISBNs from a Goodreads dataset with metadata from the Google Books API, and includes title, subtitle, authors, categories, cover thumbnail, description, publication year, average rating, page count and ratings count.

## Future Improvements

- Show a relevance score next to each result
- Merge duplicate editions of the same title
- Evaluate recommendation quality with a labelled query set or user feedback
- Add user-based or hybrid (content plus collaborative) recommendations
- Deploy the app to a cloud platform such as Hugging Face Spaces

## License

No license file is included yet, so default copyright applies to the code. The dataset is CC0.

## Author

Ayush Gairola ([ayushcpp](https://github.com/ayushcpp))
