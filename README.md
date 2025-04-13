# 🧠 Local LLaMA + RAG Terminal App

A lightweight terminal-based interface for running a **local LLaMA model** with **Retrieval-Augmented Generation (RAG)**, tailored for offline domain-specific question answering.

---

## 🔍 Project Overview

This project tests how a small local language model (LLaMA) can be enhanced using RAG techniques to provide **more informed, context-specific answers** to user queries. It allows:

- Model selection
- Prompt template editing
- Querying based on document retrieval from a local knowledge base
- Fully terminal-based operation using `curses`

---

## 🧪 Experiment Background

Two domain-specific knowledge bases were created to evaluate how well the RAG setup can differentiate context when answering the same question:

1. `nicotine_cessation/` – academic papers on smoking and addiction
2. `climate_change/` – research on environmental change and climate theories

A single question was posed to both knowledge bases:

> **"What is the meaning of the lost paradise hypothesis?"**

---

## 📚 Results Summary

### 📁 Climate Change KB
**Response:**
- Suggests that the "Lost Paradise Hypothesis" refers to a theory in paleoclimatology.
- Posits that Earth was once more hospitable, and current climate shifts may be natural cycles.
- Discusses debates over anthropogenic vs. natural causes of climate change.
- Mentions implications such as sea level rise and extreme weather.
- Cites U.S. Army War College analysis.

### 📁 Nicotine Cessation KB
**Response:**
- Defines the "Lost Paradise Hypothesis" as a psychological theory in addiction research.
- Describes reduced pleasure responses to non-nicotine rewards in dependent individuals.
- Connects this with motivational deficits and relapse post-cessation.
- Based on fMRI studies of ventral striatum activity.
- Cites: *Isomura et al., Addiction Research and Theory (2014)*.

---

## 🧠 Interpretation

The system **correctly grounded the same query** in radically different domains, demonstrating the value of local RAG even with small models. Each response was:

- **Comprehensive**
- **Cited** from domain-specific documents
- **Context-aware**

---

## ⚙️ Features

- Run LLaMA locally via CLI (`llama.cpp` assumed)
- Select model interactively
- Edit prompt templates with validation
- Load and retrieve documents via vector search
- View responses in a scrollable UI
