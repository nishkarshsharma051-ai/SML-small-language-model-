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
    "control_spotify": control_spotify
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
    }
]
