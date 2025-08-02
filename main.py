import pandas as pd
import base64
import io
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# ML & Plotting Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                             accuracy_score, f1_score, precision_score, recall_score, confusion_matrix)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# Configure Gemini API
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"⚠️ WARNING: Could not configure Gemini API. AI features will not work. Details: {e}")
    gemini_model = None

explanation_cache = {}
# --- NEW: Store the last trained model in memory ---
last_trained_model = {"model": None, "features": []}

# Pydantic Models
class Visualizations(BaseModel):
    confusion_matrix_plot: Optional[str] = None; actual_vs_predicted_plot: Optional[str] = None
class FeatureImportance(BaseModel):
    feature: str; importance: float
class TrainResponse(BaseModel):
    status: str; model_name: str; metrics: Dict[str, float]
    visualizations: Visualizations; feature_importances: Optional[List[FeatureImportance]] = None; features: List[str] # Add features to response
class TrainRequest(BaseModel):
    dataset: List[Dict[str, Any]]; target_column: str; model_name: str
class SummarizeRequest(BaseModel):
    model_name: str; target_column: str; metrics: Dict[str, Any]
class ExplainRequest(BaseModel):
    term: str
class SuggestRequest(BaseModel):
    columns: List[str]; objective: Optional[str] = None # Add optional objective
class ProfileRequest(BaseModel):
    dataset: List[Dict[str, Any]]
class ProfileResponse(BaseModel):
    profile: Dict[str, Dict[str, Any]]
class PredictRequest(BaseModel):
    data: Dict[str, Any]

# FastAPI App Setup
app = FastAPI(title="Drag n' Train AI API", version="6.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8001"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Model Mappings
REGRESSION_MODELS = { "Linear Regression": LinearRegression, "Ridge": Ridge, "Lasso": Lasso, "Random Forest Regressor": RandomForestRegressor, "XGBoost Regressor": XGBRegressor }
CLASSIFICATION_MODELS = { "Logistic Regression": LogisticRegression, "Decision Tree Classifier": DecisionTreeClassifier, "K-Nearest Neighbors": KNeighborsClassifier, "Random Forest Classifier": RandomForestClassifier, "XGBoost Classifier": XGBClassifier }
ALL_MODELS = {**REGRESSION_MODELS, **CLASSIFICATION_MODELS}

# API Endpoints
@app.get("/models", summary="Get suggested models")
async def get_models(type: str):
    if type == "regression": return {"models": list(REGRESSION_MODELS.keys())}
    if type == "classification": return {"models": list(CLASSIFICATION_MODELS.keys())}
    raise HTTPException(status_code=400, detail="Invalid problem type.")

@app.post("/train", response_model=TrainResponse, summary="Train model")
async def train_model(request: TrainRequest):
    global last_trained_model
    if request.model_name not in ALL_MODELS: raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not found.")
    df = pd.DataFrame(request.dataset)
    X = df.drop(columns=[request.target_column]); y = df[request.target_column]
    features_list = X.columns.tolist()
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    preprocessor = ColumnTransformer(transformers=[('num', 'passthrough', numeric_features), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', ALL_MODELS[request.model_name]())])
    model_pipeline.fit(X_train, y_train); y_pred = model_pipeline.predict(X_test)
    
    # Store the trained model
    last_trained_model["model"] = model_pipeline
    last_trained_model["features"] = features_list

    metrics = {}; visuals = {}; feature_importances = None
    def plot_to_base64(fig):
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight'); buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    if request.model_name in REGRESSION_MODELS:
        metrics["r2_score"] = round(r2_score(y_test, y_pred), 4); metrics["root_mean_squared_error"] = round(mean_squared_error(y_test, y_pred, squared=False), 4); metrics["mean_absolute_error"] = round(mean_absolute_error(y_test, y_pred), 4)
        fig, ax = plt.subplots(); ax.scatter(y_test, y_pred, alpha=0.7); ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        ax.set_xlabel("Actual Values"); ax.set_ylabel("Predicted Values"); ax.set_title("Actual vs. Predicted")
        visuals["actual_vs_predicted_plot"] = plot_to_base64(fig); plt.close(fig)
    elif request.model_name in CLASSIFICATION_MODELS:
        metrics["accuracy_score"] = round(accuracy_score(y_test, y_pred), 4); metrics["f1_score"] = round(f1_score(y_test, y_pred, average='weighted'), 4)
        metrics["precision"] = round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4); metrics["recall"] = round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4)
        cm = confusion_matrix(y_test, y_pred); labels = sorted(list(y_test.unique()))
        fig, ax = plt.subplots(); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted Labels'); ax.set_ylabel('True Labels'); ax.set_title('Confusion Matrix')
        visuals["confusion_matrix_plot"] = plot_to_base64(fig); plt.close(fig)
    if hasattr(model_pipeline.named_steps['model'], 'feature_importances_'):
        ohe_features = []
        if len(categorical_features) > 0: ohe_features = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
        feature_names = numeric_features.tolist() + ohe_features
        importances = model_pipeline.named_steps['model'].feature_importances_
        feature_importances = [{"feature": f, "importance": round(float(i), 4)} for f, i in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)]
    
    return TrainResponse(status="training_complete", model_name=request.model_name, metrics=metrics, visualizations=visuals, feature_importances=feature_importances, features=features_list)

@app.post("/summarize", summary="Generate a simple summary of model results")
async def summarize_results(request: SummarizeRequest):
    if not gemini_model: return {"summary": "Gemini API not configured."}
    metrics_str = ", ".join([f"{k.replace('_', ' ')} of {v}" for k, v in request.metrics.items()])
    prompt = f"Analyze the performance of a '{request.model_name}' model that predicted '{request.target_column}'. The key metrics are: {metrics_str}. 1. Start with a one-sentence summary of the model's overall performance for a non-technical user. 2. Identify the single most important metric from the list and explain what it means in this specific context (e.g., what does an accuracy of 0.85 mean for predicting the target?). 3. Provide a short, actionable insight or next step. For example, 'This is a good starting point,' or 'This model seems to struggle with...' Keep the entire response to 3-4 sentences."
    try:
        response = gemini_model.generate_content(prompt); return {"summary": response.text}
    except Exception as e:
        return {"summary": f"Could not generate summary: {e}"}

@app.post("/explain", summary="Explain an ML term using Gemini")
async def explain_term(request: ExplainRequest):
    term = request.term.strip()
    if term in explanation_cache: return {"explanation": explanation_cache[term]}
    if not gemini_model: return {"explanation": "Gemini API not configured."}
    prompt = f"Explain the machine learning term '{term}' in two simple sentences for a complete beginner. Use a real-world analogy to make it clear (e.g., 'Precision is like a cautious archer...'). Do not use technical jargon."
    try:
        response = gemini_model.generate_content(prompt); explanation_cache[term] = response.text; return {"explanation": response.text}
    except Exception as e:
        return {"explanation": f"Could not get explanation: {e}"}

@app.post("/suggest_model", summary="Suggest a model based on dataset columns")
async def suggest_model(request: SuggestRequest):
    if not gemini_model: return {"suggestion": "Gemini API not configured."}
    columns_str = ", ".join(request.columns)
    available_models = ", ".join(list(ALL_MODELS.keys()))
    # --- NEW: Incorporate user objective into the prompt ---
    user_objective_prompt = f"The user's objective is: '{request.objective}'." if request.objective else "The user has not specified an objective."
    
    prompt = f"""
    As an expert data scientist, analyze these dataset columns: {columns_str}.
    {user_objective_prompt}
    
    1.  **Task Recommendation:** Based on the columns and user objective, is this a 'Regression' or 'Classification' task?
    2.  **Target Suggestion:** Which column is the most likely target variable?
    3.  **Model Suggestions:** From this list of available models ({available_models}), which are the top 2 you would recommend trying first for this problem?
    4.  **Reasoning:** Provide a brief, simple justification for your recommendations.
    Format the response as a single, well-written paragraph.
    """
    try:
        response = gemini_model.generate_content(prompt); return {"suggestion": response.text}
    except Exception as e:
        return {"suggestion": f"Could not generate suggestion: {e}"}

@app.post("/profile", response_model=ProfileResponse, summary="Generate a basic profile of the dataset")
async def profile_dataset(request: ProfileRequest):
    try:
        df = pd.DataFrame(request.dataset); profile = {}
        for col in df.columns:
            stats = {}; col_data = df[col].dropna()
            if pd.api.types.is_numeric_dtype(col_data):
                stats['type'] = 'Numeric'; stats['mean'] = float(round(col_data.mean(), 2)); stats['median'] = float(round(col_data.median(), 2))
                stats['std_dev'] = float(round(col_data.std(), 2)); stats['min'] = float(round(col_data.min(), 2)); stats['max'] = float(round(col_data.max(), 2))
            else:
                stats['type'] = 'Categorical'; stats['unique_values'] = int(col_data.nunique())
                stats['top_value'] = col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A'
            profile[col] = stats
        return ProfileResponse(profile=profile)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not profile dataset: {e}")

# --- NEW FEATURE: /predict ENDPOINT ---
@app.post("/predict", summary="Make a prediction using the last trained model")
async def predict(request: PredictRequest):
    if not last_trained_model["model"]:
        raise HTTPException(status_code=400, detail="No model has been trained yet.")
    
    try:
        # Create a pandas DataFrame from the input data, ensuring columns are in the correct order
        input_df = pd.DataFrame([request.data], columns=last_trained_model["features"])
        
        prediction = last_trained_model["model"].predict(input_df)
        
        # Convert numpy types to standard Python types for JSON serialization
        prediction_value = prediction[0]
        if hasattr(prediction_value, 'item'):
            prediction_value = prediction_value.item()

        return {"prediction": prediction_value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not make a prediction: {e}")