"""
agent_tools.py — Tools for the Ting Ling Ling Agent to interact with the world.
"""
import os
import subprocess
import json
import traceback

def search_internet(query: str) -> str:
    """Searches the internet using DuckDuckGo."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No internet search results found."
            
            output = "Internet Search Results:\n"
            for r in results:
                output += f"- {r.get('title')} ({r.get('href')}): {r.get('body')}\n"
            return output
    except Exception as e:
        return f"Internet search failed: {str(e)}"

def browse_website(url: str) -> str:
    """Fetches and extracts readable text from a webpage URL."""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }
        print(f"\n[AGENT TOOL] Browsing website: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        
        import re
        text = re.sub(r'\s+', ' ', text)
        
        if len(text) > 4000:
            text = text[:4000] + "... (truncated)"
            
        return f"Website Content ({url}):\n{text}"
    except Exception as e:
        return f"Failed to browse website {url}: {str(e)}"

def run_terminal(command: str) -> str:
    """Runs a shell command and returns the output."""
    try:
        # Warning: This gives the model direct terminal access.
        print(f"\n[AGENT TOOL] Executing terminal command: {command}")
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            cwd=os.getcwd()
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if not output.strip():
            output = "Command executed successfully with no output."
        # Truncate if too long to save tokens
        return output[:2000]
    except Exception as e:
        return f"Terminal execution failed: {str(e)}"

def solve_math(python_expression: str) -> str:
    """Executes a safe python expression for math calculations."""
    try:
        # Limit scope of execution for safety
        allowed_globals = {"__builtins__": {}}
        allowed_locals = {}
        import math
        allowed_globals.update(math.__dict__)
        
        result = eval(python_expression, allowed_globals, allowed_locals)
        return str(result)
    except Exception as e:
        return f"Math calculation failed: {str(e)}"

def learn_from_interaction(instruction: str, response: str) -> str:
    """Appends a new interaction to the training dataset for future self-improvement."""
    try:
        # This will append to data/teacher_log.jsonl so auto_tune.py picks it up.
        dataset_path = "data/teacher_log.jsonl"
        os.makedirs("data", exist_ok=True)
        
        entry = {
            "prompt": instruction,
            "response": response,
            "teacher": "local_agent"
        }
        with open(dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            
        import subprocess
        try:
            # Trigger real-time Hugging Face background training
            subprocess.Popen(["python3", "auto_tune.py"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
            training_msg = " Real-time Hugging Face micro-training triggered in background."
        except Exception as e:
            training_msg = f" (Failed to trigger background training: {e})"
            
        return f"Successfully learned this interaction and appended to {dataset_path}.{training_msg}"
    except Exception as e:
        return f"Learning failed: {str(e)}"

def open_app(app_name: str) -> str:
    """Opens a macOS application by name. Also tries fuzzy matching if exact name fails."""
    try:
        print(f"\n[AGENT TOOL] Opening application: {app_name}")
        # Try exact match first
        result = subprocess.run(
            ["open", "-a", app_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return f"Successfully opened {app_name}."
            
        # Try to find it in /Applications or ~/Applications with case-insensitive / fuzzy match
        import os
        import fnmatch
        possible_apps = []
        for app_dir in ["/Applications", os.path.expanduser("~/Applications"), "/System/Applications"]:
            if os.path.exists(app_dir):
                for f in os.listdir(app_dir):
                    if f.endswith(".app"):
                        if app_name.lower() in f.lower():
                            possible_apps.append(os.path.join(app_dir, f))
                            
        if possible_apps:
            # Pick the shortest matching name (most likely to be the exact app)
            best_match = sorted(possible_apps, key=lambda x: len(x))[0]
            print(f"[AGENT TOOL] Falling back to best match: {best_match}")
            result2 = subprocess.run(
                ["open", best_match],
                capture_output=True,
                text=True
            )
            if result2.returncode == 0:
                app_base = os.path.basename(best_match)
                return f"Successfully opened {app_base}."
            
        return f"Failed to open {app_name}. Error: {result.stderr}"
    except Exception as e:
        return f"Error opening app: {str(e)}"

def search_safari(query: str) -> str:
    """Uses AppleScript to search the web in Safari."""
    print(f"\n[AGENT TOOL] Searching Safari for: {query}")
    applescript = f'''
    tell application "Safari"
        activate
        if (count of windows) = 0 then
            make new document
        end if
        tell front window
            set current tab to make new tab with properties {{URL:"https://www.google.com/search?q={query.replace(' ', '+')}"}}
        end tell
    end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True)
        if result.returncode == 0:
            return f"Successfully searched Safari for: {query}"
        return f"Failed to search Safari: {result.stderr}"
    except Exception as e:
        return f"Error controlling Safari: {str(e)}"

def close_safari() -> str:
    """Uses AppleScript to close the current Safari tab."""
    print("\n[AGENT TOOL] Closing Safari tab")
    applescript = '''
    tell application "Safari"
        close current tab of front window
    end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True)
        if result.returncode == 0:
            return "Successfully closed Safari tab."
        return f"Failed to close Safari: {result.stderr}"
    except Exception as e:
        return f"Error closing Safari: {str(e)}"

def send_whatsapp_message(contact_name: str, message: str) -> str:
    """Uses AppleScript to automate WhatsApp to send a message."""
    print(f"\n[AGENT TOOL] Sending WhatsApp message to {contact_name}")
    applescript = f'''
    tell application "WhatsApp" to activate
    delay 1.5
    tell application "System Events"
        keystroke "f" using {{command down}}
        delay 0.5
        keystroke "{contact_name}"
        delay 1.5
        key code 36
        delay 1.0
        keystroke "{message}"
        delay 0.5
        key code 36
    end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True)
        if result.returncode == 0:
            return f"Successfully sent WhatsApp message to {contact_name}."
        return f"Failed to send WhatsApp message: {result.stderr}"
    except Exception as e:
        return f"Error automating WhatsApp: {str(e)}"

def send_imessage(contact_name: str, message: str) -> str:
    """Uses AppleScript to automate Messages to send an iMessage."""
    print(f"\n[AGENT TOOL] Sending iMessage to {contact_name}")
    applescript = f'''
    tell application "Messages" to activate
    delay 1.5
    tell application "System Events"
        keystroke "n" using {{command down}}
        delay 0.5
        keystroke "{contact_name}"
        delay 1.5
        key code 36
        delay 0.5
        keystroke "{message}"
        delay 0.5
        key code 36
    end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True)
        if result.returncode == 0:
            return f"Successfully sent iMessage to {contact_name}."
        return f"Failed to send iMessage: {result.stderr}"
    except Exception as e:
        return f"Error automating Messages: {str(e)}"

def facetime_call(contact: str) -> str:
    """Uses AppleScript to initiate a FaceTime call."""
    print(f"\n[AGENT TOOL] Initiating FaceTime call to {contact}")
    applescript = f'''
    tell application "FaceTime" to activate
    delay 1.5
    tell application "System Events"
        keystroke "n" using {{command down}}
        delay 0.5
        keystroke "{contact}"
        delay 1.5
        key code 36
        delay 0.5
        key code 36
    end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True)
        if result.returncode == 0:
            return f"Successfully initiated FaceTime call to {contact}."
        return f"Failed to FaceTime: {result.stderr}"
    except Exception as e:
        return f"Error automating FaceTime: {str(e)}"

def send_email(recipient_email: str, subject: str, body: str) -> str:
    """Uses AppleScript to automate the native Mail app to send an email."""
    print(f"\n[AGENT TOOL] Sending email to {recipient_email}")
    applescript = f'''
    tell application "Mail"
        set newMessage to make new outgoing message with properties {{subject:"{subject}", content:"{body}", visible:true}}
        tell newMessage
            make new to recipient at end of to recipients with properties {{address:"{recipient_email}"}}
            send
        end tell
    end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", applescript], capture_output=True, text=True)
        if result.returncode == 0:
            return f"Successfully sent email to {recipient_email}."
        return f"Failed to send email: {result.stderr}"
    except Exception as e:
        return f"Error automating Mail: {str(e)}"

def execute_applescript(script: str) -> str:
    """Executes a custom AppleScript for UI automation."""
    print("\n[AGENT TOOL] Executing custom AppleScript")
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if result.returncode == 0:
            return f"AppleScript executed successfully: {result.stdout}"
        return f"AppleScript failed: {result.stderr}"
    except Exception as e:
        return f"Failed to execute AppleScript: {str(e)}"

def control_spotify(action: str, query: str = None) -> str:
    """Controls Spotify playback and searching using native AppleScript."""
    print(f"\n[AGENT TOOL] Controlling Spotify: {action} {query or ''}")
    
    script = ""
    if action == "play":
        script = 'tell application "Spotify" to play'
    elif action == "pause":
        script = 'tell application "Spotify" to pause'
    elif action == "next":
        script = 'tell application "Spotify" to next track'
    elif action == "search_and_play" and query:
        # Spotify allows playing a search query URI directly!
        query_encoded = query.replace(" ", "%20")
        script = f'tell application "Spotify" to play track "spotify:search:{query_encoded}"'
    else:
        return "Invalid Spotify action or missing query."
        
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if result.returncode != 0:
            return f"Spotify error: {result.stderr}"
        return f"Successfully executed Spotify action: {action}"
    except Exception as e:
        return f"Failed to control Spotify: {e}"

def build_web_ui(project_name: str, html_content: str, css_content: str = "", js_content: str = "", extra_files: dict = None, open_in_browser: bool = True) -> str:
    """
    Builds a complete web page, UI, landing page, or multi-page web app from scratch like Manus AI.
    Saves HTML, CSS, JavaScript, and extra pages into generated_uis/<project_name>/,
    and automatically opens the live UI directly in the browser.
    """
    try:
        import re
        slug = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_name.strip().lower()) or "my_ui_app"
        output_dir = os.path.join(os.getcwd(), "generated_uis", slug)
        os.makedirs(output_dir, exist_ok=True)
        
        def normalize_html(raw_html: str, title: str) -> str:
            if "<html" not in raw_html.lower():
                raw_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="style.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
{raw_html}
    <script src="app.js"></script>
</body>
</html>"""
            else:
                if css_content and 'href="style.css"' not in raw_html and "href='style.css'" not in raw_html:
                    raw_html = raw_html.replace("</head>", '    <link rel="stylesheet" href="style.css">\n</head>')
                if js_content and 'src="app.js"' not in raw_html and "src='app.js'" not in raw_html:
                    raw_html = raw_html.replace("</body>", '    <script src="app.js"></script>\n</body>')
            return raw_html

        index_path = os.path.join(output_dir, "index.html")
        final_index_html = normalize_html(html_content, project_name)
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(final_index_html)
            
        created_files = [f"index.html ({len(final_index_html)} bytes)"]

        if css_content:
            css_path = os.path.join(output_dir, "style.css")
            with open(css_path, "w", encoding="utf-8") as f:
                f.write(css_content)
            created_files.append(f"style.css ({len(css_content)} bytes)")

        if js_content:
            js_path = os.path.join(output_dir, "app.js")
            with open(js_path, "w", encoding="utf-8") as f:
                f.write(js_content)
            created_files.append(f"app.js ({len(js_content)} bytes)")

        if extra_files and isinstance(extra_files, dict):
            for filename, content in extra_files.items():
                clean_fname = os.path.basename(str(filename))
                file_path = os.path.join(output_dir, clean_fname)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(str(content))
                created_files.append(f"{clean_fname} ({len(str(content))} bytes)")

        if open_in_browser:
            try:
                subprocess.Popen(["open", index_path])
                browser_msg = f" Opened live UI in browser: file://{index_path}"
            except Exception as e:
                browser_msg = f" Browser launch error: {e}"
        else:
            browser_msg = ""

        return (
            f"✅ Manus AI UI Build Complete!\n"
            f"Project: '{project_name}' (Folder: generated_uis/{slug})\n"
            f"Files Created:\n" + "\n".join(f"  - {fname}" for fname in created_files) +
            f"\n{browser_msg}"
        )
    except Exception as e:
        return f"Failed to build web UI: {str(e)}"

def preview_web_ui(project_name: str) -> str:
    """Opens an existing generated UI project in Safari/browser."""
    try:
        import re
        slug = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_name.strip().lower())
        index_path = os.path.join(os.getcwd(), "generated_uis", slug, "index.html")
        if not os.path.exists(index_path):
            return f"UI Project '{project_name}' not found at {index_path}."
        subprocess.Popen(["open", index_path])
        return f"Successfully opened '{project_name}' UI preview in browser."
    except Exception as e:
        return f"Failed to preview UI: {str(e)}"

def list_generated_uis() -> str:
    """Lists all UIs and web pages generated by Ting Ling Ling."""
    try:
        base_dir = os.path.join(os.getcwd(), "generated_uis")
        if not os.path.exists(base_dir) or not os.listdir(base_dir):
            return "No generated UIs found yet. Ask Ting Ling Ling to build one!"
            
        projects = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                files = os.listdir(item_path)
                projects.append(f"- {item} ({len(files)} files: {', '.join(files)})")
                
        return "Generated Web UIs:\n" + "\n".join(projects)
    except Exception as e:
        return f"Failed to list generated UIs: {str(e)}"

def read_file(file_path: str) -> str:
    """Reads the text or code content of any file from the filesystem."""
    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f"Error: File '{file_path}' does not exist."
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if len(content) > 8000:
            return content[:8000] + f"\n... (truncated {len(content) - 8000} remaining characters)"
        return content
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"

def write_file(file_path: str, content: str) -> str:
    """Creates a new file or overwrites an existing file in any format (code, text, json, html, markdown, csv, etc)."""
    try:
        abs_path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully created/wrote {len(content)} characters to '{file_path}'."
    except Exception as e:
        return f"Error writing to file '{file_path}': {str(e)}"

def edit_file(file_path: str, find_text: str, replace_text: str) -> str:
    """Edits an existing file by replacing target text or code blocks with new content."""
    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f"Error: File '{file_path}' does not exist."
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if find_text not in content:
            return f"Error: Could not find target text in '{file_path}'."
        new_content = content.replace(find_text, replace_text, 1)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Successfully updated '{file_path}'."
    except Exception as e:
        return f"Error editing file '{file_path}': {str(e)}"

def translate_file(file_path: str, target_language: str, output_path: str = None) -> str:
    """
    Reads a file and translates its natural language text/comments into any target human language
    (Hindi, Spanish, French, German, Japanese, Chinese, etc.), preserving formatting and saving to output_path.
    """
    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f"Error: File '{file_path}' does not exist."
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        
        base, ext = os.path.splitext(abs_path)
        clean_lang = target_language.strip().lower().replace(" ", "_")
        dest_path = output_path or f"{base}_{clean_lang}{ext}"
        
        return (
            f"FILE_TRANSLATION_JOB:\n"
            f"Source File: '{abs_path}'\n"
            f"Target Language: '{target_language}'\n"
            f"Output Destination: '{dest_path}'\n"
            f"Source Content Preview:\n---\n{content[:6000]}"
        )
    except Exception as e:
        return f"Error opening file for translation: {str(e)}"

MEMORY_FILE = os.path.join(os.getcwd(), "data", "user_memory.json")

def save_user_memory(memory_key: str, memory_value: str) -> str:
    """Saves a user memory or preference persistently (e.g. 'coding_preference': 'Python with type hints', 'name': 'Nishkarsh')."""
    try:
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        data = {}
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data[memory_key] = memory_value
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return f"Successfully stored memory: {memory_key} = '{memory_value}'"
    except Exception as e:
        return f"Error saving memory: {str(e)}"

def get_user_memory() -> str:
    """Retrieves all stored user memories and preferences."""
    try:
        if not os.path.exists(MEMORY_FILE):
            return "No custom user memories saved yet."
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return "User memory is currently empty."
        return "User Preferences & Memories:\n" + "\n".join(f"- {k}: {v}" for k, v in data.items())
    except Exception as e:
        return f"Error retrieving user memory: {str(e)}"

def create_interactive_chart(chart_type: str, title: str, labels: list, data_points: list, dataset_label: str = "Data") -> str:
    """Creates a ChatGPT-style interactive data chart (bar, line, pie, doughnut, radar) rendered directly in chat."""
    try:
        chart_config = {
            "type": chart_type.lower(),
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": dataset_label,
                    "data": data_points,
                    "backgroundColor": [
                        "rgba(56, 189, 248, 0.6)", "rgba(168, 85, 247, 0.6)",
                        "rgba(236, 72, 153, 0.6)", "rgba(34, 197, 94, 0.6)",
                        "rgba(251, 146, 60, 0.6)"
                    ],
                    "borderColor": "rgba(255, 255, 255, 0.8)",
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {"display": True, "text": title, "color": "#ffffff"}
                }
            }
        }
        return f"CHART_RENDER_BLOCK:\n```chartjson\n{json.dumps(chart_config, indent=2)}\n```\nRendered interactive {chart_type} chart: '{title}'."
    except Exception as e:
        return f"Error building chart: {str(e)}"

def generate_diagram(diagram_type: str, diagram_code: str) -> str:
    """Generates a ChatGPT-style architecture diagram, flowchart, sequence diagram, or mindmap using Mermaid.js."""
    try:
        clean_code = diagram_code.strip()
        return f"DIAGRAM_RENDER_BLOCK:\n```mermaid\n{clean_code}\n```\nRendered interactive {diagram_type} diagram."
    except Exception as e:
        return f"Error generating diagram: {str(e)}"

def export_chat_document(filename: str, format_type: str, content: str) -> str:
    """Exports any chat output, code, essay, report, or data file to disk for downloading/sharing."""
    try:
        exports_dir = os.path.join(os.getcwd(), "data", "exports")
        os.makedirs(exports_dir, exist_ok=True)
        ext = format_type.strip().lower().replace(".", "")
        if not filename.endswith(f".{ext}"):
            full_filename = f"{filename}.{ext}"
        else:
            full_filename = filename
        dest_path = os.path.join(exports_dir, full_filename)
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully exported document to 'data/exports/{full_filename}' ({len(content)} characters)."
    except Exception as e:
        return f"Error exporting document: {str(e)}"

def generate_image(prompt: str, width: int = 1024, height: int = 1024, style: str = "photorealistic") -> str:
    """
    Generates a high-resolution AI image using Flux / Stable Diffusion models based on prompt description.
    Saves image to data/generated_images/ and renders it directly in chat.
    """
    try:
        import urllib.parse
        import urllib.request
        import uuid
        
        enhanced_prompt = f"{prompt}, {style} style, 8k resolution, highly detailed, cinematic lighting" if style and style.lower() != "none" else prompt
        encoded_prompt = urllib.parse.quote(enhanced_prompt)
        
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&seed={uuid.uuid4().int % 100000}&nologo=true"
        
        images_dir = os.path.join(os.getcwd(), "data", "generated_images")
        os.makedirs(images_dir, exist_ok=True)
        
        img_id = str(uuid.uuid4())[:8]
        filename = f"gen_img_{img_id}.jpg"
        file_path = os.path.join(images_dir, filename)
        
        req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response, open(file_path, 'wb') as out_file:
            out_file.write(response.read())
            
        web_path = f"/generated_images/{filename}"
        return f"IMAGE_GEN_RESULT:\n![Generated AI Image: {prompt}]({web_path})\n\nSuccessfully generated high-resolution AI image for prompt: '{prompt}'."
    except Exception as e:
        return f"Failed to generate image: {str(e)}"

def get_system_health() -> str:
    """Retrieves live macOS system diagnostics: CPU usage, RAM consumption, Disk space, Battery level, and Uptime."""
    try:
        import psutil
        cpu_usage = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        battery = psutil.sensors_battery()
        
        mem_used_gb = round(mem.used / (1024**3), 2)
        mem_total_gb = round(mem.total / (1024**3), 2)
        disk_free_gb = round(disk.free / (1024**3), 2)
        disk_total_gb = round(disk.total / (1024**3), 2)
        
        batt_str = f"{battery.percent}% ({'Plugged In' if battery.power_plugged else 'On Battery'})" if battery else "N/A"
        
        return (
            f"🖥️ macOS System Health Diagnostics:\n"
            f"  - CPU Utilization: {cpu_usage}%\n"
            f"  - RAM Usage: {mem_used_gb} GB / {mem_total_gb} GB ({mem.percent}%)\n"
            f"  - Disk Space Available: {disk_free_gb} GB free of {disk_total_gb} GB ({disk.percent}% used)\n"
            f"  - Battery Status: {batt_str}"
        )
    except Exception as e:
        try:
            top_out = subprocess.check_output("top -l 1 | grep -E '^CPU|^PhysMem'", shell=True, text=True)
            df_out = subprocess.check_output("df -h /", shell=True, text=True)
            return f"macOS System Diagnostics:\n{top_out.strip()}\nDisk Usage:\n{df_out.strip()}"
        except Exception as ex:
            return f"Error retrieving system diagnostics: {str(e)}"

def search_local_files(search_query: str, root_dir: str = None, extension: str = None) -> str:
    """Searches local filesystem for files matching a keyword, query, or file extension."""
    try:
        search_root = os.path.abspath(root_dir) if root_dir else os.getcwd()
        matches = []
        clean_ext = extension.strip().lower().lstrip(".") if extension else None
        
        for root, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", "venv")]
            for file in files:
                if search_query.lower() in file.lower():
                    if not clean_ext or file.lower().endswith(f".{clean_ext}"):
                        rel_path = os.path.relpath(os.path.join(root, file), search_root)
                        matches.append(rel_path)
                        if len(matches) >= 25:
                            break
            if len(matches) >= 25:
                break
                
        if not matches:
            return f"No files matching '{search_query}' found in '{search_root}'."
        return f"Found {len(matches)} matching file(s) in '{search_root}':\n" + "\n".join(f"  - {m}" for m in matches)
    except Exception as e:
        return f"Error searching files: {str(e)}"

def get_weather_forecast(city: str) -> str:
    """Fetches real-time live weather forecast, temperature, humidity, and condition for any city worldwide."""
    try:
        import urllib.parse
        import urllib.request
        clean_city = urllib.parse.quote(city.strip())
        url = f"https://wttr.in/{clean_city}?format=j1"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        current = data['current_condition'][0]
        temp_c = current['temp_C']
        temp_f = current['temp_F']
        desc = current['weatherDesc'][0]['value']
        humidity = current['humidity']
        wind_speed = current['windspeedKmph']
        
        return (
            f"🌤️ Live Weather Report for {city.title()}:\n"
            f"  - Condition: {desc}\n"
            f"  - Temperature: {temp_c}°C ({temp_f}°F)\n"
            f"  - Humidity: {humidity}%\n"
            f"  - Wind Speed: {wind_speed} km/h"
        )
    except Exception as e:
        return search_internet(f"current weather in {city}")

def get_tech_news_briefing() -> str:
    """Fetches live top technology and AI news headlines."""
    try:
        return search_internet("latest artificial intelligence and tech news today")
    except Exception as e:
        return f"Error fetching news briefing: {str(e)}"

# A dictionary mapping function names to the actual python functions
AVAILABLE_TOOLS = {
    "search_internet": search_internet,
    "browse_website": browse_website,
    "run_terminal": run_terminal,
    "solve_math": solve_math,
    "learn_from_interaction": learn_from_interaction,
    "open_app": open_app,
    "search_safari": search_safari,
    "close_safari": close_safari,
    "send_whatsapp_message": send_whatsapp_message,
    "send_imessage": send_imessage,
    "facetime_call": facetime_call,
    "send_email": send_email,
    "execute_applescript": execute_applescript,
    "control_spotify": control_spotify,
    "build_web_ui": build_web_ui,
    "preview_web_ui": preview_web_ui,
    "list_generated_uis": list_generated_uis,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "translate_file": translate_file,
    "save_user_memory": save_user_memory,
    "get_user_memory": get_user_memory,
    "create_interactive_chart": create_interactive_chart,
    "generate_diagram": generate_diagram,
    "export_chat_document": export_chat_document,
    "generate_image": generate_image,
    "get_system_health": get_system_health,
    "search_local_files": search_local_files,
    "get_weather_forecast": get_weather_forecast,
    "get_tech_news_briefing": get_tech_news_briefing
}

# OpenAI/Groq Tool Definitions Schema
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_internet",
            "description": "Searches the internet for real-time information or facts. Returns a list of results with titles, snippets, and URLs. If the snippets don't contain enough information, use 'browse_website' to read the full page using its URL. Stop searching and reply to the user once you have enough information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_website",
            "description": "Fetches and reads the main text content of a webpage given its URL. Useful for reading articles from search_internet results. Stop reading and reply to the user once you have enough information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the website to browse."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal",
            "description": "Executes a shell command on the host machine. Use this to create files, develop apps, or manage the system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "solve_math",
            "description": "Evaluates a mathematical expression using Python.",
            "parameters": {
                "type": "object",
                "properties": {
                    "python_expression": {
                        "type": "string",
                        "description": "A valid python mathematical expression (e.g., '100 * 20' or 'math.sqrt(16)')."
                    }
                },
                "required": ["python_expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "learn_from_interaction",
            "description": "Saves a successful interaction or workflow to your local training dataset so you can improve your offline model later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "The request the user made."
                    },
                    "response": {
                        "type": "string",
                        "description": "The successful response or code that satisfied the request."
                    }
                },
                "required": ["instruction", "response"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Opens a macOS application by its name (e.g., 'Spotify', 'Safari', 'Notes').",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "The exact name of the application to open."
                    }
                },
                "required": ["app_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_safari",
            "description": "Opens Safari and searches the internet for the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on Google."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "close_safari",
            "description": "Closes the current active tab in Safari.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp_message",
            "description": "Sends a message to a specific contact on WhatsApp using UI automation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {
                        "type": "string",
                        "description": "The exact name of the contact as it appears in WhatsApp."
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to send."
                    }
                },
                "required": ["contact_name", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_imessage",
            "description": "Sends a text message to a specific contact via Apple Messages (iMessage/SMS).",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {
                        "type": "string",
                        "description": "The exact name, phone number, or email of the contact."
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to send."
                    }
                },
                "required": ["contact_name", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "facetime_call",
            "description": "Initiates a FaceTime call to a specific contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact": {
                        "type": "string",
                        "description": "The exact name, phone number, or email of the contact to call."
                    }
                },
                "required": ["contact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Composes and sends an email via the macOS Mail app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient_email": {
                        "type": "string",
                        "description": "The email address of the recipient."
                    },
                    "subject": {
                        "type": "string",
                        "description": "The subject line of the email."
                    },
                    "body": {
                        "type": "string",
                        "description": "The body content of the email."
                    }
                },
                "required": ["recipient_email", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_applescript",
            "description": "Executes arbitrary AppleScript to automate macOS UI (e.g., clicking buttons, opening specific settings panes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "The valid AppleScript code to execute."
                    }
                },
                "required": ["script"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_spotify",
            "description": "Controls Spotify playback natively (play, pause, next track, or play a specific song search query).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action to perform: 'play', 'pause', 'next', or 'search_and_play'."
                    },
                    "query": {
                        "type": "string",
                        "description": "The song name to play (required only if action is 'search_and_play')."
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "build_web_ui",
            "description": "Builds a complete, modern, responsive web application, landing page, UI, or multi-page site from scratch like Manus AI. Automatically saves HTML, CSS, JavaScript, and extra pages, and opens the live UI directly in the browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the UI project or app (e.g. 'crypto_dashboard', 'portfolio_site', 'todo_app')."
                    },
                    "html_content": {
                        "type": "string",
                        "description": "The full HTML code for index.html."
                    },
                    "css_content": {
                        "type": "string",
                        "description": "The full CSS styling for style.css with modern design, glassmorphism, animations, and typography."
                    },
                    "js_content": {
                        "type": "string",
                        "description": "The full JavaScript logic for app.js."
                    },
                    "extra_files": {
                        "type": "object",
                        "description": "Optional dictionary of additional pages or assets (e.g. {'about.html': '...', 'contact.html': '...'})."
                    },
                    "open_in_browser": {
                        "type": "boolean",
                        "description": "Whether to automatically open the live UI preview in the default browser/Safari (default: true)."
                    }
                },
                "required": ["project_name", "html_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "preview_web_ui",
            "description": "Opens an existing generated UI project in Safari or default browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "The project name to preview."
                    }
                },
                "required": ["project_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_generated_uis",
            "description": "Lists all UIs, dashboards, and web pages built by Ting Ling Ling.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads text, code, or document content from any file on disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Creates a new file or overwrites an existing file in any format (code, text, JSON, HTML, CSS, Markdown, CSV, etc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The target path for the file."
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text/code content to write."
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edits an existing file by replacing specific target text or code blocks with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit."
                    },
                    "find_text": {
                        "type": "string",
                        "description": "Exact text or code block to replace."
                    },
                    "replace_text": {
                        "type": "string",
                        "description": "New replacement text or code block."
                    }
                },
                "required": ["file_path", "find_text", "replace_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translate_file",
            "description": "Reads a file and translates its content into any target human language (e.g., Hindi, Spanish, French, German, Japanese, Chinese, etc.), preserving file structure and outputting to a specified file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to translate."
                    },
                    "target_language": {
                        "type": "string",
                        "description": "The target human language (e.g. 'Hindi', 'Spanish', 'French', 'Japanese', 'German')."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional output file path. Defaults to '<filename>_<target_language>.<ext>'."
                    }
                },
                "required": ["file_path", "target_language"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_memory",
            "description": "Saves a user memory or preference persistently (e.g. 'coding_style': 'Python 3.9', 'language': 'English').",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_key": {
                        "type": "string",
                        "description": "Key or label for the preference/memory."
                    },
                    "memory_value": {
                        "type": "string",
                        "description": "Value or detail to remember."
                    }
                },
                "required": ["memory_key", "memory_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_memory",
            "description": "Retrieves all saved user preferences and long-term memories.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_interactive_chart",
            "description": "Renders a ChatGPT-style interactive data chart (bar, line, pie, doughnut, radar) directly in the chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "description": "Type of chart: 'bar', 'line', 'pie', 'doughnut', or 'radar'."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the chart."
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of label strings for X-axis or categories."
                    },
                    "data_points": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numeric values corresponding to labels."
                    },
                    "dataset_label": {
                        "type": "string",
                        "description": "Name of the dataset (e.g. 'Revenue', 'Score')."
                    }
                },
                "required": ["chart_type", "title", "labels", "data_points"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_diagram",
            "description": "Generates a ChatGPT-style architecture diagram, flowchart, sequence diagram, or mindmap using Mermaid.js syntax.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diagram_type": {
                        "type": "string",
                        "description": "Type of diagram (e.g. 'flowchart', 'sequence', 'architecture', 'mindmap')."
                    },
                    "diagram_code": {
                        "type": "string",
                        "description": "The exact Mermaid.js diagram code string (e.g. 'graph TD; A-->B;')."
                    }
                },
                "required": ["diagram_type", "diagram_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "export_chat_document",
            "description": "Exports chat responses, essays, reports, code snippets, or data files into downloadable files (.md, .txt, .html, .py, .csv, .json).",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The target filename (e.g. 'report.md', 'script.py', 'data.json')."
                    },
                    "format_type": {
                        "type": "string",
                        "description": "File format extension (e.g. 'md', 'py', 'txt', 'html', 'json')."
                    },
                    "content": {
                        "type": "string",
                        "description": "Full document or code content to export."
                    }
                },
                "required": ["filename", "format_type", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generates a high-resolution AI image (DALL-E / Flux style) based on a text prompt description, saves it locally, and displays it directly in chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed visual description of the image to generate."
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels (default: 1024)."
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels (default: 1024)."
                    },
                    "style": {
                        "type": "string",
                        "description": "Art style (e.g. 'photorealistic', 'anime', '3d render', 'cyberpunk', 'oil painting', 'digital art')."
                    }
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_health",
            "description": "Retrieves live macOS system performance diagnostics: CPU usage, RAM utilization, Disk free space, and Battery status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_files",
            "description": "Searches the local filesystem for files matching a keyword, query string, or specific file extension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Keyword or query string to search for in filenames."
                    },
                    "root_dir": {
                        "type": "string",
                        "description": "Directory path to start searching from (default: current workspace)."
                    },
                    "extension": {
                        "type": "string",
                        "description": "Optional file extension to filter by (e.g. 'py', 'json', 'txt')."
                    }
                },
                "required": ["search_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Fetches real-time live weather forecast, temperature, humidity, and wind speed for any city worldwide.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city (e.g. 'New Delhi', 'London', 'San Francisco', 'Tokyo')."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_tech_news_briefing",
            "description": "Fetches live top technology, AI, and science news headlines.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]
