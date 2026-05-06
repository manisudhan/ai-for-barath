# 🏛️ AI For Bharat: Unified Business Identifier (UBID) Pipeline

**Theme:** Unified Business Identifier (UBID) and Active Business Intelligence by Karnataka Commerce & Industry.

## 📖 Overview
This project solves the challenge of fragmented government data by creating a **Unified Business Identifier (UBID)** for the state of Karnataka. It safely ingests data from various departmental silos without disrupting legacy systems, cleans it, and uses **Semantic AI (Natural Language Processing)** to intelligently link duplicate records. 

Coupled with a **Transactional State Machine**, the system accurately classifies businesses as *Active*, *Dormant*, or *Closed*, and provides a blazing-fast "Omni-Search" interface for government officials to query the Master Registry.

## ✨ Key Features
* **Safe Data Consolidation (Part A):** Fetches and isolates legacy department data, grouping businesses geographically by Pincode.
* **Semantic AI Linkage (Part B):** Uses `all-MiniLM-L6-v2` transformer models to generate vector embeddings, allowing the system to match businesses based on *meaning* rather than just rigid keywords.
* **Activity State Machine:** Tracks historical transactional events (like tax filings or license cancellations) to automatically determine if a business is Active, Dormant, or Closed.
* **Human-in-the-Loop UI:** High-confidence AI matches are auto-merged, while ambiguous edge cases are routed to a secure Human Review Queue for manual approval.
* **Omni-Search Dashboard:** A Google-like search engine for government officials to instantly query the state's unified business directory.

## 🛠️ Tech Stack
* **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript
* **Backend:** Python, Flask
* **Database:** NeonDB (Serverless PostgreSQL), SQLAlchemy
* **AI & Data Processing:** `sentence-transformers`, `scikit-learn` (Cosine Similarity), `pandas`, `numpy`

---

## 🚀 Getting Started

Follow these steps to run the UBID pipeline on your local machine.

### 1. Prerequisites
* Python 3.8+ installed on your machine.
* A [NeonDB](https://neon.tech/) account (or any standard PostgreSQL database).

### 2. Installation
Clone the repository and install the required Python dependencies:

```bash
git clone [https://github.com/your-username/aiforbharat-ubid.git](https://github.com/your-username/aiforbharat-ubid.git)
cd aiforbharat-ubid

# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install pandas numpy flask flask-cors sqlalchemy python-dotenv sentence-transformers scikit-learn
