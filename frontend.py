import asyncio
import json
from pyodide.http import pyfetch
from js import document, alert, console, event, Blob, URL
from pyodide.ffi import create_proxy
from pyscript import when

# --- Global State ---
uploaded_data = None
column_names = []
comparison_results = []
latest_training_data = None
last_trained_model_info = {} 
BACKEND_URL = "https://drag-n-train.onrender.com"
FAQ_TERMS = ["Accuracy Score", "Classification", "R2 Score", "Confusion Matrix", "Feature Importance", "Random Forest", "Regression", "XGBoost"]

# --- UI Helper Functions ---
def check_train_button_state():
    problem_selected = document.getElementById("problem-type").value != ""
    file_loaded = uploaded_data is not None
    document.getElementById("train-button").disabled = not (problem_selected and file_loaded)

def render_comparison_table():
    comp_div = document.getElementById("comparison-section")
    if not comparison_results:
        comp_div.style.display = 'none'
        return
    
    comp_div.innerHTML = "<h3>Model Comparison</h3>"
    table = document.createElement("table")
    all_metric_keys = sorted(list(set(key for res in comparison_results for key in res['metrics'].keys())))
    headers = ["Model"] + [key.replace('_',' ').title() for key in all_metric_keys]
    
    thead = table.createTHead(); header_row = thead.insertRow()
    for h in headers:
        th = document.createElement("th"); th.innerText = h; header_row.appendChild(th)
        
    tbody = table.createTBody()
    for result in comparison_results:
        body_row = tbody.insertRow()
        body_row.insertCell().innerHTML = "<strong>" + result['model_name'] + "</strong>"
        for key in all_metric_keys:
            body_row.insertCell().innerText = str(result['metrics'].get(key, 'N/A'))
    
    comp_div.appendChild(table); comp_div.style.display = 'block'

# --- Core Application Logic ---
async def handle_file_upload(file):
    global uploaded_data, comparison_results, column_names, latest_training_data
    import pandas as pd
    
    comparison_results = []; latest_training_data = None
    document.getElementById("comparison-section").style.display = 'none'
    document.getElementById("results-col-1").innerHTML = ""
    document.getElementById("results-col-2").innerHTML = ""
    document.getElementById("status").innerText = ""
    document.getElementById("ai-suggestion-card").style.display = 'none'
    document.getElementById("data-profile-card").style.display = 'none'
    document.getElementById("download-report-button").style.display = 'none'
    document.getElementById("package-model-button").style.display = 'none'
   
    document.getElementById("xai-dashboard").style.display = 'none'
    
    try:
        text = await file.text(); df = pd.read_csv(pd.io.common.StringIO(text))
        uploaded_data = df.to_dict(orient='records'); column_names = df.columns.tolist()
        
        target_select = document.getElementById("target-column"); target_select.innerHTML = ""
        for col in df.columns:
            opt = document.createElement("option"); opt.value = col; opt.text = col
            target_select.appendChild(opt)
        target_select.disabled = False
        
        document.querySelector(".drop-zone__prompt span").innerText = "Loaded: " + file.name
        document.getElementById("objective-section").style.display = 'block'
        document.getElementById("action-buttons").style.display = 'grid'
        document.getElementById("controls-section").style.display = 'block'
    except Exception as e:
        alert("Invalid CSV file. Please check the format."); console.log("Error during file upload:", str(e))
    finally:
        check_train_button_state()

@when("change", "#problem-type")
async def on_problem_change(event):
    model_select = document.getElementById("model-name")
    model_select.innerHTML = "<option>Loading...</option>"; model_select.disabled = True
    try:
        res = await pyfetch(f"{BACKEND_URL}/models?type={event.target.value}")
        data = await res.json(); model_select.innerHTML = ""
        for m in data['models']:
            o = document.createElement("option"); o.value = m; o.text = m
            model_select.appendChild(o)
        model_select.disabled = False
    except Exception as e:
        model_select.innerHTML = f"<option>Error: {str(e)}</option>"
    finally:
        check_train_button_state()

@when("click", "#train-button")
async def on_train(event):
    global comparison_results, latest_training_data, last_trained_model_info
    btn = document.getElementById("train-button"); btn.disabled = True
    status = document.getElementById("status"); col1 = document.getElementById("results-col-1"); col2 = document.getElementById("results-col-2")
    col1.innerHTML = ""; col2.innerHTML = ""; status.innerText = "Training model..."
    document.getElementById("download-report-button").style.display = 'none'
    document.getElementById("package-model-button").style.display = 'none'
    
    document.getElementById("xai-dashboard").style.display = 'none'
    
    try:
        model_name = document.getElementById("model-name").value
        payload = { "dataset": uploaded_data, "target_column": document.getElementById("target-column").value, "model_name": model_name }
        res = await pyfetch(url=f"{BACKEND_URL}/train", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps(payload))
        data = await res.json()
        if not res.ok:
            alert(f"Server Error: {str(data.get('detail'))}"); return

        info_card = document.createElement("div"); info_card.className = "result-card"; info_card.innerHTML = f"<h3>Model: {data['model_name']}</h3><p><strong>Status:</strong> {data['status']}</p>"; col1.appendChild(info_card)
        
        metrics_card = document.createElement("div"); metrics_card.className = "result-card"
        metrics_table = document.createElement("table"); metrics_card.innerHTML = "<h3>Performance Metrics</h3>"
        for key, val in data['metrics'].items():
            row = metrics_table.insertRow(-1); th = document.createElement("th"); th.className = "metrics-header-cell"; formatted_key = key.replace('_',' ').title(); th.innerText = f"{formatted_key} "
            help_icon = document.createElement("span"); help_icon.className = "help-icon"; help_icon.innerText = "?"; help_icon.addEventListener("click", create_proxy(lambda e, t=formatted_key: asyncio.ensure_future(on_help_click(t)))); th.appendChild(help_icon); row.appendChild(th); td = document.createElement("td"); td.innerText = str(val); row.appendChild(td)
        metrics_card.appendChild(metrics_table); col1.appendChild(metrics_card)
        
        summary_res = await pyfetch(url=f"{BACKEND_URL}/summarize", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps({"model_name": data['model_name'], "metrics": data['metrics'], "target_column": payload['target_column']}))
        summary_data = await summary_res.json()
        summary_card = document.createElement("div"); summary_card.className = "result-card ai-summary"; summary_card.innerHTML = f"<h3>Analysis</h3><p>{summary_data['summary']}</p>"; col1.appendChild(summary_card)

        if data.get("feature_importances"):
            fi_card = document.createElement("div"); fi_card.className = "result-card"; fi_card.innerHTML = "<h3>Feature Importances</h3>"
            fi_table = document.createElement("table")
            for f in data["feature_importances"]:
                fi_row = fi_table.insertRow(-1); fi_th = document.createElement("th"); fi_th.innerText = f['feature']; fi_row.appendChild(fi_th); fi_td = document.createElement("td"); fi_td.innerText = str(f['importance']); fi_row.appendChild(fi_td)
            fi_card.appendChild(fi_table); col1.appendChild(fi_card)

        visuals = data.get("visualizations", {})
        for name, base64str in visuals.items():
            if base64str:
                card = document.createElement("div"); card.className = "result-card"; card.innerHTML = f"<h3>{name.replace('_',' ').title()}</h3>"
                img = document.createElement("img"); img.src = f"data:image/png;base64,{base64str}"; card.appendChild(img); col2.appendChild(card)
        
        if data.get("generated_code"):
            code_card = document.createElement("div"); code_card.className = "result-card"; code_card.innerHTML = "<h3>Generated Python Code</h3>"
            code_block = document.createElement("pre"); code_block.className = "code-block"; code_content = document.createElement("code"); code_content.innerText = data["generated_code"]
            code_block.appendChild(code_content); code_card.appendChild(code_block); col2.appendChild(code_card)
        
        if data.get("features"):
            last_trained_model_info["features"] = data["features"]
            pred_card = document.createElement("div"); pred_card.className = "result-card"; pred_card.innerHTML = "<h3>Live Prediction</h3>"
            for feature in data["features"]:
                label = document.createElement("label"); label.innerText = feature; pred_card.appendChild(label)
                inp = document.createElement("input"); inp.type = "text"; inp.id = f"pred_inp_{feature}"; inp.placeholder = f"Enter value for {feature}"; inp.className = "w-full bg-slate-700 border-slate-600 rounded-md p-2 text-slate-100"
                pred_card.appendChild(inp)
            pred_btn = document.createElement("button"); pred_btn.innerText = "Predict"; pred_btn.style.marginTop = "1rem"
            pred_btn.addEventListener("click", create_proxy(on_predict_click))
            pred_card.appendChild(pred_btn)
            pred_result = document.createElement("p"); pred_result.id = "prediction-result"; pred_result.style.fontWeight = "bold"; pred_result.style.marginTop = "1rem"
            pred_card.appendChild(pred_result)
            col1.appendChild(pred_card)

        latest_training_data = { "model_name": data['model_name'], "metrics": data['metrics'], "summary": summary_data['summary'], "visualizations": data['visualizations'] }
        document.getElementById("download-report-button").style.display = 'block'
        document.getElementById("package-model-button").style.display = 'block'
        
        status.innerText = "✅ Model trained successfully! Choose another model to compare."
        comparison_results = [r for r in comparison_results if r['model_name'] != data['model_name']]
        comparison_results.append(data); render_comparison_table()

        
        document.getElementById("xai-dashboard").style.display = 'block'
        document.getElementById("explanation-plot-img").style.display = 'none'
        document.getElementById("explanation-status").innerText = ''

    except Exception as e:
        status.innerText = "❌ Error during training"; alert(f"An unexpected error occurred during training: {str(e)}")
        console.log("Training Error:", str(e))
    finally:
        btn.disabled = False



@when("click", "#explain-shap-button")
@when("click", "#explain-lime-button")
async def on_explain_click(event):
    button_id = event.target.id
    status_el = document.getElementById("explanation-status")
    plot_img_el = document.getElementById("explanation-plot-img")
   
    summary_el = document.getElementById("explanation-summary")
    
    status_el.innerText = "Generating explanation, please wait..."
    plot_img_el.style.display = 'none'
    
    summary_el.innerText = ""
    
    base_url = f"{BACKEND_URL}/explain"
    params = ""
    
    if button_id == "explain-shap-button":
        params = "?type=shap_summary"
    elif button_id == "explain-lime-button":
        instance_index = document.getElementById("lime-instance-index").value
        params = f"?type=lime_instance&instance_index={instance_index}"

    try:
        res = await pyfetch(url=f"{base_url}{params}", method="GET")
        data = await res.json()
        
        if res.ok:
            status_el.innerText = ""
            plot_img_el.src = f"data:image/png;base64,{data['explanation_plot']}"
            plot_img_el.style.display = 'block'
           
            if data.get("summary"):
                summary_el.innerText = data["summary"]
        else:
            status_el.innerText = f"Error: {data.get('detail', 'Could not generate explanation.')}"
            
    except Exception as e:
        status_el.innerText = f"Client-side Error: {str(e)}"
        console.log("Explanation Error:", str(e))
@when("click", "#download-report-button")
async def on_download_report_click(event):
    if not latest_training_data: alert("Please train a model first."); return
    btn = event.target; btn.disabled = True; btn.innerText = "Generating PDF..."
    try:
        res = await pyfetch(url=f"{BACKEND_URL}/generate_report", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps(latest_training_data))
        if res.ok:
            js_res = res.js_response; array_buffer = await js_res.arrayBuffer()
            pdf_blob = Blob.new([array_buffer], {"type": "application/pdf"})
            url = URL.createObjectURL(pdf_blob)
            hidden_link = document.getElementById("download-link"); hidden_link.href = url
            hidden_link.download = "drag_n_train_report.pdf"; hidden_link.click(); URL.revokeObjectURL(url)
        else:
            data = await res.json(); alert(f"Failed to generate report: {data.get('detail')}")
    except Exception as e:
        alert(f"An error occurred while generating the report: {e}")
    finally:
        btn.disabled = False; btn.innerText = "Download Report"
        
@when("click", "#package-model-button")
async def on_package_click(event):
    btn = event.target; btn.disabled = True; btn.innerText = "Packaging API..."
    try:
        res = await pyfetch(url=f"{BACKEND_URL}/package_model")
        if res.ok:
            js_res = res.js_response; array_buffer = await js_res.arrayBuffer()
            zip_blob = Blob.new([array_buffer], {"type": "application/zip"})
            url = URL.createObjectURL(zip_blob)
            hidden_link = document.getElementById("download-link"); hidden_link.href = url
            hidden_link.download = "drag_n_train_model_api.zip"; hidden_link.click(); URL.revokeObjectURL(url)
        else:
            data = await res.json(); alert(f"Failed to package model: {data.get('detail')}")
    except Exception as e:
        alert(f"An error occurred while packaging the model: {e}")
    finally:
        btn.disabled = False; btn.innerText = "Package as API"

async def on_predict_click(event):
    pred_result = document.getElementById("prediction-result"); pred_result.innerText = "Predicting..."
    features = last_trained_model_info.get("features", [])
    input_data = {}
    for f in features:
        val = document.getElementById(f"pred_inp_{f}").value
        try: input_data[f] = float(val)
        except (ValueError, TypeError): input_data[f] = val
    try:
        res = await pyfetch(url=f"{BACKEND_URL}/predict", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps({"data": input_data}))
        data = await res.json()
        if res.ok: pred_result.innerText = "Predicted Value: " + str(data.get('prediction'))
        else: pred_result.innerText = "Error: " + data.get('detail')
    except Exception as e:
        pred_result.innerText = "Error: " + str(e)
        
async def on_help_click(term):
    modal = document.getElementById("ai-modal"); document.getElementById("modal-title").innerText = term; document.getElementById("modal-body").innerText = "Analyzing..."; modal.style.display = "flex"
    try:
        res = await pyfetch(url=f"{BACKEND_URL}/explain", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps({"term": term}))
        data = await res.json()
        document.getElementById("modal-body").innerText = data.get("explanation", "Sorry, could not get an explanation.")
    except Exception as e:
        document.getElementById("modal-body").innerText = f"Error: {str(e)}"

@when("click", "#ai-modal")
def close_ai_modal(event):
    document.getElementById("ai-modal").style.display = "none"

@when("click", ".modal-content")
def stop_propagation(event):
    event.stopPropagation()

@when("click", "#suggest-button")
async def on_suggest_click(event):
    btn = event.target; btn.disabled = True; btn.innerText = "Analyzing..."
    suggestion_card = document.getElementById("ai-suggestion-card"); suggestion_card.style.display = 'block'; suggestion_card.innerHTML = "<h3>Model Suggestion</h3><p>Analyzing columns...</p>"
    user_objective = document.getElementById("user-objective").value
    try:
        res = await pyfetch(url=f"{BACKEND_URL}/suggest_model", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps({"columns": column_names, "objective": user_objective}))
        data = await res.json()
        suggestion_card.innerHTML = "<h3>Model Suggestion</h3><p>" + data.get('suggestion', 'Could not get a suggestion.') + "</p>"
    except Exception as e:
        suggestion_card.innerHTML = f"<h3>Model Suggestion</h3><p>Error getting suggestion: {str(e)}</p>"
    finally:
        btn.disabled = False; btn.innerText = "Suggest Model"

@when("click", "#profile-button")
async def on_profile_click(event):
    btn = event.target; btn.disabled = True; btn.innerText = "Profiling..."
    profile_card = document.getElementById("data-profile-card"); profile_card.style.display = 'block'; profile_card.innerHTML = "<h3>Data Profile</h3><p>Analyzing dataset...</p>"
    try:
        res = await pyfetch(url=f"{BACKEND_URL}/profile", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps({"dataset": uploaded_data}))
        data = await res.json()
        profile_card.innerHTML = "<h3>Data Profile</h3>"
        profile_table = document.createElement("table"); thead = profile_table.createTHead(); row = thead.insertRow();
        for head in ["Column", "Type", "Details"]:
            th = document.createElement("th"); th.innerText = head; row.appendChild(th)
        tbody = profile_table.createTBody()
        for col, stats in data['profile'].items():
            row = tbody.insertRow(); details = ""
            if stats['type'] == 'Numeric': details = f"Mean: {stats['mean']}, Median: {stats['median']}"
            else: details = f"Unique: {stats['unique_values']}, Top: {stats['top_value']}"
            row.insertCell().innerText = col; row.insertCell().innerText = stats['type']; row.insertCell().innerText = details
        profile_card.appendChild(profile_table)
    except Exception as e:
        profile_card.innerHTML = f"<h3>Data Profile</h3><p>Error profiling data: {str(e)}</p>"
    finally:
        btn.disabled = False; btn.innerText = "Profile Dataset"

@when("click", "#kb-search-button")
def on_kb_search(event):
    term = document.getElementById("kb-search-input").value
    if term: asyncio.ensure_future(on_help_click(term))

@when("change", "#file-upload-input")
async def on_file_select(event):
    if event.target.files.length > 0: await handle_file_upload(event.target.files.item(0))

def prevent_defaults(event): event.preventDefault(); event.stopPropagation()

@when("dragenter", "#drop-zone")
@when("dragover", "#drop-zone")
def highlight(event):
    prevent_defaults(event)
    document.getElementById("drop-zone").classList.add("drop-zone--over")

@when("dragleave", "#drop-zone")
@when("dragend", "#drop-zone")
def unhighlight(event):
    prevent_defaults(event)
    document.getElementById("drop-zone").classList.remove("drop-zone--over")

@when("drop", "#drop-zone")
async def on_file_drop(event):
    unhighlight(event); prevent_defaults(event)
    if event.dataTransfer.files.length > 0: await handle_file_upload(event.dataTransfer.files.item(0))

@when("click", "#drop-zone")
def on_drop_zone_click(event):
    document.getElementById("file-upload-input").click()

@when("click", "#help-problem-type")
def on_help_problem_click(event):
    asyncio.ensure_future(on_help_click("Problem Type"))

@when("click", "#help-target-column")
def on_help_target_click(event):
    asyncio.ensure_future(on_help_click("Target Column"))

@when("click", "#close-welcome-button")
def close_welcome_modal(event):
    document.getElementById("welcome-modal").style.display = "none"

@when("click", "#welcome-modal")
def close_welcome_modal_bg(event):
    document.getElementById("welcome-modal").style.display = "none"


faq_grid = document.getElementById("faq-grid")
for term in FAQ_TERMS:
    item = document.createElement("div"); item.className = "faq-item"; item.innerText = term
    item.addEventListener("click", create_proxy(lambda e, t=term: asyncio.ensure_future(on_help_click(t))))
    faq_grid.appendChild(item)

document.getElementById("welcome-modal").style.display = "flex"