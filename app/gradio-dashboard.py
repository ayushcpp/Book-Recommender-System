from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

log = logging.getLogger(__name__)


BOOKS_CSV = "../data/processed/books_with_emotions.csv"
TAGGED_TXT = "../data/processed/tagged_description.txt"
CHROMA_DIR = "../Chroma_DB"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

PLACEHOLDER_COVER = "https://placehold.co/128x192?text=No+Cover"

EMOTION_COLS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral",
]

CATEGORY_OPTIONS = [
    "All",
    "Fiction",
    "Nonfiction",
    "Children's Fiction",
    "Children's Nonfiction",
]

EMOTION_OPTIONS = [
    "All",
    "Joy",
    "Sadness",
    "Fear",
    "Anger",
    "Disgust",
    "Surprise",
    "Neutral",
]


def load_books():
    df = pd.read_csv(BOOKS_CSV)

    df["title"] = df["title"].fillna("Unknown Title")
    df["authors"] = df["authors"].fillna("Unknown Author")
    df["description"] = df["description"].fillna("No description available.")
    df["thumbnail"] = df["thumbnail"].fillna(PLACEHOLDER_COVER)
    df["simple_categories"] = df["simple_categories"].fillna("Unknown")

    for col in EMOTION_COLS:
        df[col] = pd.to_numeric(
            df.get(col, 0),
            errors="coerce"
        ).fillna(0)

    df["isbn13"] = pd.to_numeric(
        df["isbn13"],
        errors="coerce"
    )

    df = df.dropna(subset=["isbn13"])
    df["isbn13"] = df["isbn13"].astype(int)

    return df


def build_or_load_db():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    try:
        db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

        db.similarity_search("test", k=1)

        log.info("Loaded existing Chroma database")
        return db

    except Exception:

        log.info("Building Chroma database...")

        raw_docs = TextLoader(
            TAGGED_TXT,
            encoding="utf-8"
        ).load()

        docs = CharacterTextSplitter(
            separator="\n",
            chunk_size=6000,
            chunk_overlap=0
        ).split_documents(raw_docs)

        db = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )

        log.info("Database created")
        return db


def semantic_search(query, top_k=50):

    docs = _DB.similarity_search(
        query,
        k=top_k
    )

    isbn_list = []

    for doc in docs:

        try:
            isbn = int(
                doc.page_content.strip('"').split()[0]
            )
            isbn_list.append(isbn)

        except Exception:
            pass

    return isbn_list


def filter_and_rank(
    isbn_list,
    category,
    emotion,
    top_n=16
):

    results = _BOOKS_DF[
        _BOOKS_DF["isbn13"].isin(isbn_list)
    ].copy()

    if category != "All":
        results = results[
            results["simple_categories"] == category
        ]

    emotion_col = emotion.lower()

    if emotion_col in EMOTION_COLS:
        results = results.sort_values(
            emotion_col,
            ascending=False
        )

    return results.head(top_n)


def to_gallery(df):

    gallery = []

    for row in df.itertuples():

        rating = getattr(
            row,
            "average_rating",
            None
        )

        stars = ""

        if pd.notna(rating):
            stars = f" ★ {rating:.1f}"

        caption = (
            f"{row.title}\n"
            f"{row.authors}{stars}"
        )

        gallery.append(
            (
                row.thumbnail,
                caption
            )
        )

    return gallery


_last_results = pd.DataFrame()


def recommend(
    query,
    category,
    emotion
):

    global _last_results

    query = query.strip()

    if not query:

        gr.Warning(
            "Enter a book description first."
        )

        return (
            [],
            "",
            "",
            "",
            gr.update(visible=False)
        )

    isbn_list = semantic_search(query)

    results = filter_and_rank(
        isbn_list,
        category,
        emotion
    )

    if results.empty:

        gr.Warning(
            "No books found."
        )

        return (
            [],
            "",
            "",
            "",
            gr.update(visible=False)
        )

    _last_results = results.reset_index(
        drop=True
    )

    return (
        to_gallery(results),
        "",
        "",
        "",
        gr.update(visible=False)
    )


def show_detail(evt: gr.SelectData):

    idx = evt.index

    if (
        _last_results.empty
        or idx >= len(_last_results)
    ):
        return (
            "",
            "",
            "",
            gr.update(visible=False)
        )

    row = _last_results.iloc[idx]

    title = row["title"]
    authors = row["authors"]
    description = row["description"]

    header = f"### {title}\n**{authors}**"

    return (
        header,
        "",
        description,
        gr.update(visible=True)
    )


_BOOKS_DF = load_books()
_DB = build_or_load_db()


with gr.Blocks(
    title="Semantic Book Recommender"
) as demo:

    gr.Markdown(
        "# 📚 Semantic Book Recommender"
    )

    with gr.Row():

        query_box = gr.Textbox(
            label="Describe your ideal book",
            lines=2
        )

        category_dd = gr.Dropdown(
            CATEGORY_OPTIONS,
            value="All",
            label="Category"
        )

        emotion_dd = gr.Dropdown(
            EMOTION_OPTIONS,
            value="All",
            label="Emotion"
        )

    search_btn = gr.Button(
        "Find Books"
    )

    gallery = gr.Gallery(
        label="Recommendations",
        columns=4,
        rows=4
    )

    with gr.Group(
        visible=False
    ) as detail_panel:

        detail_header = gr.Markdown()

        detail_meta = gr.Markdown()

        detail_desc = gr.Textbox(
            lines=8,
            interactive=False
        )

    outputs = [
        gallery,
        detail_header,
        detail_meta,
        detail_desc,
        detail_panel
    ]

    search_btn.click(
        recommend,
        inputs=[
            query_box,
            category_dd,
            emotion_dd
        ],
        outputs=outputs
    )

    gallery.select(
        show_detail,
        outputs=[
            detail_header,
            detail_meta,
            detail_desc,
            detail_panel
        ]
    )


if __name__ == "__main__":

    demo.launch(
        inbrowser=True,
        show_error=True
    )