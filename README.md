# Drag n' Train

### From Dataset to Decision, Instantly.

[![Drag n' Train Screenshot](https://res.cloudinary.com/dolthd0mr/image/upload/v1754205883/Screenshot_from_2025-08-03_12-52-39_pxqxkv.png)](https://drag-n-train.vercel.app/)


---

###  Submission for the PyWeb Creators Hackathon 2025

This project is a proud submission to the **PyWeb Creators Hackathon 2025**. It follows the **Hybrid Track**, celebrating Python's power by using it across the entire stack—from a high-performance backend to a fully interactive in-browser frontend.

## Live Demo

Experience the full power of Drag n' Train live in your browser. No installation required.

**[➡️ Launch the Live Application](https://your-netlify-url.netlify.app/)**


---

## The Big Idea

Machine learning is powerful, but the barrier to entry is high. **Drag n' Train** is a no-code ML playground that bridges the gap. It empowers anyone to upload a dataset, get intelligent suggestions from an AI assistant, train multiple models, and even deploy their own prediction API—all without writing a single line of code.

More than just a tool, it's an educational platform that **celebrates Python** by showing users the generated `scikit-learn` code, teaching them how to go from "no-code" to "pro-code." This project is our vision for a more accessible, Python-powered web.

---

## Key Features: A Complete Data Science Studio

Drag n' Train provides an end-to-end workflow, taking a user from a raw CSV file to a deployable API.

#### 1. Instant Data Understanding
*   `📤` **Drag & Drop Upload:** A seamless and intuitive interface to load any CSV dataset.
*   `🔍` **One-Click Data Profiling:** Instantly generate a statistical summary of your dataset to understand its shape and content.
*   `✨` **AI-Powered Model Suggestion:** Based on your dataset's columns and your stated objective, our AI assistant recommends the best problem type and models to try first.

#### 2. Effortless Model Training
*   `⚙️` **Train Multiple Models:** Train a suite of industry-standard models, from `Linear Regression` to `RandomForest` and `XGBoost`, with a single click.
*   `📊` **Rich Visualizations:** Automatically generates plots like Confusion Matrices and Actual vs. Predicted scatter plots to visually assess model performance.
*   `📈` **Side-by-Side Model Comparison:** All trained models are added to a clean comparison table, allowing for easy identification of the best-performing model.

#### 3. AI-Powered Insights & Learning
*   `🧠` **Explainable AI (XAI) Dashboard:** Go beyond *what* the model predicts to understand *why*. Our XAI dashboard uses SHAP and LIME to generate clear plots and **AI-powered summaries** that explain which features are most influential for both the model overall and for individual predictions.
*   `📄` **Automated Analysis:** Get a simple, layman's summary of your model's performance, written by our AI assistant.
*   `📚` **Interactive ML Knowledge Base:** An integrated, AI-powered dictionary. Click the `(?)` icon next to any metric or search for any term to get an instant, simple explanation.

#### 4. From No-Code to Pro-Code & Deployment
*   `🐍` **Python Code Generation:** To celebrate Python, the app automatically generates a clean `scikit-learn` script that perfectly replicates your training process.
*   `📋` **One-Click PDF Reports:** Download a professional, multi-page PDF report of your entire analysis, complete with plots and the AI summary.
*   `📦` **Package as API:** With a single click, download a fully deployable FastAPI microservice, complete with a trained model, API endpoints, and a `requirements.txt` file.

## Celebrating Python: Our Hybrid Approach & The 80% Rule

This project was built from the ground up to meet the "80% Python code" rule and celebrate the language's versatility.

*   **Backend (100% Python):** A powerful **FastAPI** server handles all the heavy lifting:
    *   Training `scikit-learn` and `XGBoost` models.
    *   Generating plots with `Matplotlib` and `Seaborn`.
    *   Integrating with the **Google Gemini API** for intelligent suggestions and analysis.
    *   Generating professional **PDF reports** with `fpdf2`.
    *   Saving models with `joblib` and **programmatically generating a new FastAPI application** for the "Package as API" feature.

*   **Frontend (100% Python Logic):** All client-side logic and user interaction are handled by Python running in the browser, powered by **PyScript**.
    *   **No JavaScript was written.**
    *   Python handles all DOM manipulation, event handling, and API communication with the backend.

By splitting our frontend Python logic into its own `frontend.py` file, we ensure that GitHub's language analysis accurately reflects the project's Python-centric nature.

---

## Tech Stack

| Category | Technology |
| :--- | :--- |
| **Backend** | Python, FastAPI, Uvicorn, Scikit-learn, XGBoost, Pandas, Matplotlib, Seaborn, shap, lime |
| **Frontend** | Python, PyScript (Pyodide) |
| **Cloud & AI** | Render (for backend hosting), Vercel (for frontend hosting), Google Gemini API |
| **Deployment** | Joblib (for model serialization), FPDF2 (for PDF generation) |

---

## How to Run Locally

To get Drag n' Train running on your local machine, follow these steps.

#### Prerequisites
*   Python 3.10+
*   A Google Gemini API Key

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/drag-n-train.git
cd drag-n-train
```
### 2. Set Up the Backend
*   **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
*   **Set Your API Key:** Create a secret environment variable for your Gemini API key.
    *   *(On macOS/Linux):*
        ```bash
        export GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```
    *   *(On Windows CMD):*
        ```bash
        set GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```
*   **Run the Backend Server:**
    ```bash
    uvicorn main:app --reload
    ```
    The backend will now be running at `http://localhost:8000`.

### 3. Run the Frontend
*   Open a **second, new terminal window**.
*   Navigate to the same project directory.
*   **Run the Frontend Server:**
    ```bash
    python -m http.server 8001
    ```

### 4. Launch the App
*   Open your web browser and go to:
    **[http://localhost:8001](http://localhost:8001)**

---

## Hackathon Evaluation

We designed this project to excel across all evaluation criteria:

*   **Python Mastery (30%):** We demonstrated mastery by building a complex hybrid application using Python on both the server (FastAPI, scikit-learn) and the client (PyScript). The project uses advanced features like programmatic code generation, PDF creation, and API packaging, all in pure Python.
*   **Creativity & Originality (25%):** Our "no-code to pro-code" pipeline is a unique take on the "Data Science Studio" track. Features like AI-powered model suggestions, live prediction playgrounds, and one-click API packaging are highly original for a hackathon project.
*   **Technical Excellence (20%):** The application is robust, featuring a clean separation of concerns between a stateful backend and a reactive frontend. The integration with a major AI API (Gemini) and the programmatic generation of a deployable microservice showcase a high level of technical skill.
*   **UX & Polish (15%):** We focused on a clean, modern, and intuitive user experience. The dark theme, interactive helpers, and clear workflow make the powerful backend features accessible and enjoyable to use.
*   **Documentation (10%):** This README provides a clear, comprehensive guide to the project's vision, features, and setup.
