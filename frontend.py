 
import asyncio
import json
from pyodide.http import pyfetch
from js import document, alert, console, event
from pyodide.ffi import create_proxy
from pyscript import when

uploaded_data = None; column_names = []; comparison_results = []
BACKEND_URL = "http://localhost:8000"
FAQ_TERMS = ["Accuracy Score", "R2 Score", "Confusion Matrix", "Feature Importance", "Random Forest", "XGBoost"]

def check_train_button_state():
        problem_selected = document.getElementById("problem-type").value != ""
        file_loaded = uploaded_data is not None
        document.getElementById("train-button").disabled = not (problem_selected and file_loaded)

async def handle_file_upload(file):
        global uploaded_data, comparison_results, column_names; import pandas as pd
        comparison_results = []; document.getElementById("comparison-section").style.display = 'none'
        document.getElementById("results-col-1").innerHTML = ""; document.getElementById("results-col-2").innerHTML = ""
        document.getElementById("status").innerText = ""; document.getElementById("ai-suggestion-card").style.display = 'none'
        document.getElementById("data-profile-card").style.display = 'none'
        try:
            text = await file.text(); df = pd.read_csv(pd.io.common.StringIO(text))
            uploaded_data = df.to_dict(orient='records'); column_names = df.columns.tolist()
            target = document.getElementById("target-column"); target.innerHTML = ""
            for col in df.columns:
                opt = document.createElement("option"); opt.value = col; opt.text = col; target.appendChild(opt)
            target.disabled = False
            document.querySelector(".drop-zone__prompt").innerText = "Loaded: " + file.name
            document.getElementById("objective-section").style.display = 'block'
            document.getElementById("action-buttons").style.display = 'grid'
            document.getElementById("controls-section").style.display = 'block'
        except Exception as e:
            alert("Invalid CSV file."); console.log("Error:", str(e))
        finally:
            check_train_button_state()

async def on_problem_change(event):
        model_select = document.getElementById("model-name"); model_select.innerHTML = "<option>Loading...</option>"; model_select.disabled = True
        try:
            res = await pyfetch(BACKEND_URL + "/models?type=" + event.target.value)
            data = await res.json(); model_select.innerHTML = ""
            for m in data['models']:
                o = document.createElement("option"); o.value = m; o.text = m; model_select.appendChild(o)
            model_select.disabled = False
        except Exception as e:
            model_select.innerHTML = "<option>Error: " + str(e) + "</option>"
        finally:
            check_train_button_state()
    
def render_comparison_table():
        comp_div = document.getElementById("comparison-section")
        if not comparison_results: comp_div.style.display = 'none'; return
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

async def on_train(event):
        btn = document.getElementById("train-button"); btn.disabled = True
        status = document.getElementById("status"); col1 = document.getElementById("results-col-1"); col2 = document.getElementById("results-col-2")
        col1.innerHTML = ""; col2.innerHTML = ""; status.innerText = "Training model..."
        try:
            model_name = document.getElementById("model-name").value
            payload = { "dataset": uploaded_data, "target_column": document.getElementById("target-column").value, "model_name": model_name }
            res = await pyfetch(url=BACKEND_URL + "/train", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps(payload))
            data = await res.json()
            if not res.ok: alert("Server Error: " + str(data.get('detail'))); return
            
            info_card = document.createElement("div"); info_card.className = "result-card"; info_card.innerHTML = "<h3>Model: " + data['model_name'] + "</h3><p><strong>Status:</strong> " + data['status'] + "</p>"; col1.appendChild(info_card)
            
            metrics_card = document.createElement("div"); metrics_card.className = "result-card"; metrics_card.innerHTML = "<h3>Performance Metrics</h3>"
            metrics_table = document.createElement("table")
            for key, val in data['metrics'].items():
                row = metrics_table.insertRow(-1)
                th = document.createElement("th")
                # --- PYTHON FIX: Add the specific class to the 'th' element ---
                th.className = "metrics-header-cell"
                formatted_key = key.replace('_',' ').title()
                th.appendChild(document.createTextNode(formatted_key + " "))
                help_icon = document.createElement("span"); help_icon.className = "help-icon"; help_icon.innerText = "?"
                help_icon.addEventListener("click", create_proxy(lambda e, t=formatted_key: asyncio.ensure_future(on_help_click(t))))
                th.appendChild(help_icon); row.appendChild(th)
                td = document.createElement("td"); td.innerText = str(val); row.appendChild(td)
            metrics_card.appendChild(metrics_table); col1.appendChild(metrics_card)
            
            summary_res = await pyfetch(url=BACKEND_URL + "/summarize", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps({"model_name": data['model_name'], "metrics": data['metrics'], "target_column": payload['target_column']}))
            summary_data = await summary_res.json()
            summary_card = document.createElement("div"); summary_card.className = "result-card ai-summary"; summary_card.innerHTML = "<h3>Analysis</h3><p>" + summary_data['summary'] + "</p>"; col1.appendChild(summary_card)
            
            if data.get("feature_importances"):
                fi_card = document.createElement("div"); fi_card.className = "result-card"; fi_card.innerHTML = "<h3>Feature Importances</h3>"; fi_table = document.createElement("table")
                for f in data["feature_importances"]:
                    fi_row = fi_table.insertRow(-1); fi_th = document.createElement("th"); fi_th.innerText = f['feature']; fi_row.appendChild(fi_th); fi_td = document.createElement("td"); fi_td.innerText = str(f['importance']); fi_row.appendChild(fi_td)
                fi_card.appendChild(fi_table); col1.appendChild(fi_card)
            
            visuals = data.get("visualizations", {});
            for name, base64str in visuals.items():
                if base64str:
                    card = document.createElement("div"); card.className = "result-card"; card.innerHTML = "<h3>" + name.replace('_',' ').title() + "</h3>"; img = document.createElement("img"); img.src = "data:image/png;base64," + base64str; card.appendChild(img); col2.appendChild(card)
            
            status.innerText = "✅ Model trained successfully! Choose another model to compare."
            global comparison_results
            comparison_results = [r for r in comparison_results if r['model_name'] != data['model_name']]
            comparison_results.append(data); render_comparison_table()
        except Exception as e:
            status.innerText = "❌ Error during training"; alert("Unexpected error: " + str(e))
        finally:
            btn.disabled = False
    
async def on_help_click(term):
        modal = document.getElementById("ai-modal"); document.getElementById("modal-title").innerText = term; document.getElementById("modal-body").innerText = "Analyzing..."; modal.style.display = "flex"
        try:
            res = await pyfetch(url=BACKEND_URL + "/explain", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps({"term": term}))
            data = await res.json()
            document.getElementById("modal-body").innerText = data.get("explanation", "Sorry, could not get an explanation.")
        except Exception as e:
            document.getElementById("modal-body").innerText = "Error: " + str(e)
            
def close_modal(event): document.getElementById("ai-modal").style.display = "none"

async def on_suggest_click(event):
        btn = event.target; btn.disabled = True; btn.innerText = "Analyzing..."
        suggestion_card = document.getElementById("ai-suggestion-card"); suggestion_card.style.display = 'block'; suggestion_card.innerHTML = "<h3>Model Suggestion</h3><p>Analyzing columns...</p>"
        user_objective = document.getElementById("user-objective").value
        try:
            res = await pyfetch(url=BACKEND_URL + "/suggest_model", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps({"columns": column_names, "objective": user_objective}))
            data = await res.json()
            suggestion_card.innerHTML = "<h3>Model Suggestion</h3><p>" + data.get('suggestion', 'Could not get a suggestion.') + "</p>"
        except Exception as e:
            suggestion_card.innerHTML = "<h3>Model Suggestion</h3><p>Error getting suggestion: " + str(e) + "</p>"
        finally:
            btn.disabled = False; btn.innerText = "Suggest Model"

async def on_profile_click(event):
        btn = event.target; btn.disabled = True; btn.innerText = "Profiling..."
        profile_card = document.getElementById("data-profile-card"); profile_card.style.display = 'block'; profile_card.innerHTML = "<h3>Data Profile</h3><p>Analyzing dataset...</p>"
        try:
            res = await pyfetch(url=BACKEND_URL + "/profile", method="POST", headers={"Content-Type": "application/json"}, body=json.dumps({"dataset": uploaded_data}))
            data = await res.json()
            profile_card.innerHTML = "<h3>Data Profile</h3>"
            profile_table = document.createElement("table")
            thead = profile_table.createTHead(); row = thead.insertRow();
            for head in ["Column", "Type", "Details"]:
                th = document.createElement("th"); th.innerText = head; row.appendChild(th)
            tbody = profile_table.createTBody()
            for col, stats in data['profile'].items():
                row = tbody.insertRow()
                details = ""
                if stats['type'] == 'Numeric':
                    details = "Mean: " + str(stats['mean']) + ", Median: " + str(stats['median'])
                else:
                    details = "Unique: " + str(stats['unique_values']) + ", Top: " + str(stats['top_value'])
                row.insertCell().innerText = col
                row.insertCell().innerText = stats['type']
                row.insertCell().innerText = details
            profile_card.appendChild(profile_table)
        except Exception as e:
            profile_card.innerHTML = "<h3>Data Profile</h3><p>Error profiling data: " + str(e) + "</p>"
        finally:
            btn.disabled = False; btn.innerText = "Profile Dataset"

def on_kb_search():
        term = document.getElementById("kb-search-input").value
        if term:
            asyncio.ensure_future(on_help_click(term))

def on_drop_zone_click(event): document.getElementById("file-upload-input").click()
def prevent_defaults(event): event.preventDefault(); event.stopPropagation()
def highlight(event): prevent_defaults(event); document.getElementById("drop-zone").classList.add("drop-zone--over")
def unhighlight(event): prevent_defaults(event); document.getElementById("drop-zone").classList.remove("drop-zone--over")
async def on_file_drop(event): unhighlight(event); prevent_defaults(event); await handle_file_upload(event.dataTransfer.files.item(0))
async def on_file_select(event): await handle_file_upload(event.target.files.item(0))
    
def main():
        drop_zone = document.getElementById("drop-zone"); file_input = document.getElementById("file-upload-input")
        drop_zone.addEventListener("click", create_proxy(on_drop_zone_click))
        for ev in ["dragenter", "dragover"]: drop_zone.addEventListener(ev, create_proxy(highlight))
        for ev in ["dragleave", "dragend"]: drop_zone.addEventListener(ev, create_proxy(unhighlight))
        drop_zone.addEventListener("drop", create_proxy(on_file_drop))
        document.getElementById("ai-modal").addEventListener("click", create_proxy(close_modal))
        document.querySelector(".modal-content").addEventListener("click", create_proxy(lambda e: e.stopPropagation()))
        when("change", "#file-upload-input")(on_file_select)
        when("change", "#problem-type")(on_problem_change)
        when("click", "#train-button")(on_train)
        when("click", "#suggest-button")(on_suggest_click)
        when("click", "#profile-button")(on_profile_click)
        when("click", "#kb-search-button")(on_kb_search)
        
        document.getElementById("help-problem-type").addEventListener("click", create_proxy(lambda e: asyncio.ensure_future(on_help_click("Problem Type"))))
        document.getElementById("help-target-column").addEventListener("click", create_proxy(lambda e: asyncio.ensure_future(on_help_click("Target Column"))))
        
        faq_grid = document.getElementById("faq-grid")
        for term in FAQ_TERMS:
            item = document.createElement("div"); item.className = "faq-item"; item.innerText = term
            item.addEventListener("click", create_proxy(lambda e, t=term: asyncio.ensure_future(on_help_click(t))))
            faq_grid.appendChild(item)
    
main()
