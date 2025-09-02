import sys
import os

# Añade el directorio actual al path de Python
sys.path.insert(0, os.path.dirname(__file__))

# Importa tu aplicación Flask desde app.py
from app import app as application