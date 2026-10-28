# ChronoTrace / SynapseState: Streamlit Community Cloud Deployment Guide

This guide provides end-to-end instructions for deploying **ChronoTrace** directly to **Streamlit Community Cloud** (`share.streamlit.io`).

---

## 1. Prerequisites & GitHub Repository Setup

1. **Initialize Git in the ChronoTrace Project Folder:**
   ```bash
   cd ChronoTrace
   git init
   git add .
   git commit -m "Initial commit of ChronoTrace multi-meeting intelligence engine"
   ```

2. **Push to GitHub:**
   - Create a new public or private repository on GitHub (e.g., `https://github.com/your-username/ChronoTrace`).
   - Link and push your local branch:
     ```bash
     git remote add origin https://github.com/your-username/ChronoTrace.git
     git branch -M main
     git push -u origin main
     ```

---

## 2. Deploy on Streamlit Community Cloud

1. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **"New app"** in the top right corner.
3. Configure the deployment parameters:
   - **Repository:** `your-username/ChronoTrace`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
   - **App URL:** Customize your subdomain (e.g., `chronotrace-engine.streamlit.app`).

---

## 3. Configure Secrets (Environment Variables)

In the Streamlit Cloud deployment modal, click **"Advanced settings..."** (or navigate to **App Settings > Secrets** after deploying).
Add your optional API tokens:

```toml
# Streamlit Cloud Secrets Configuration
GEMINI_API_KEY = "AIzaSy..."
OPENAI_API_KEY = "sk-..."
HUGGINGFACE_TOKEN = "hf_..."
CHRONOTRACE_DB_PATH = "data/chronotrace.db"
```

> [!NOTE]
> ChronoTrace features a zero-configuration fallback mode. If no API keys or HuggingFace tokens are provided, the system seamlessly runs using its deterministic semantic extractors and offline lexical grounding verifier.

---

## 4. Handling Database & Vector Persistence on Cloud

- Streamlit Community Cloud provides an ephemeral file system.
- ChronoTrace automatically handles this:
  - On app startup, `init_session_state()` checks if `data/chronotrace.db` exists.
  - If the database is missing or freshly mounted, it runs `seed_database(reset=False)` to guarantee that Sessions 101, 102, and 103 are immediately available for evaluation and grading.
  - Users can click **"🌱 Re-Seed Sample Database"** in the sidebar at any time to restore the initial test state.

---

## 5. Verifying Deployment

Once deployed:
1. Verify the **Executive Overview** loads with 3 sessions, 6 decisions, and 4 questions.
2. Navigate to **Decision Timeline** to inspect the evolution of MongoDB to PostgreSQL.
3. Check **Conversation Debt Tracker** to see the 80.0 dynamic severity index for Session 101 carry-overs.
4. Test the **DBMS SQL Playground** by executing Query 1 through Query 5 live.
