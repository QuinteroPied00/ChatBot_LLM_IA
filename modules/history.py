import json
import os

HISTORY_FILE = "data/historial/search_history.json"

def ensure_history_file():
    """
    Verifica la existencia del archivo de historial y lo crea si no existe.
    """
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)

def save_search(query, response):
    """
    Guarda una consulta y su respuesta en el historial de búsquedas.
    """
    ensure_history_file()
    with open(HISTORY_FILE, "r+") as f:
        history = json.load(f)
        history.append({"query": query, "response": response})
        f.seek(0)
        json.dump(history, f, indent=4)