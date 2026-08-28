# Basic-RAG-Q-A-bot
This is my basic RAG Q & A Bot. I made this for MSA recruitment task. I made this using python.
# Simple RAG Mini Q&A Bot

A beginner-friendly **Retrieval-Augmented Generation (RAG) Mini Q&A Bot** built using **Python and TF-IDF (Scikit-learn)**.

This project reads information from a `.txt` file, finds the most relevant passage for a user's question, calculates a similarity score, and displays the most relevant passage as the answer.

--

## What is RAG?

**RAG stands for Retrieval-Augmented Generation.**

A RAG system first **retrieves relevant information** from a collection of documents and then uses that information to answer a question.

In this mini project, we are focusing mainly on the **retrieval part**.

Instead of using a large AI model, this project uses **TF-IDF and Cosine Similarity** to find the most relevant passage.

--

## How This Project Works

The basic workflow is:

User Question
      ↓
Read Documents
      ↓
Split Document into Passages
      ↓
Convert Text into TF-IDF Vectors
      ↓
Calculate Cosine Similarity
      ↓
Find Highest Similarity Score
      ↓
Return Most Relevant Passage
