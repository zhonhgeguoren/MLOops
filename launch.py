import subprocess
import sys
import os

# Определяем путь к app.py в той же папке
script_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(script_dir, 'app.py')

# Запускаем Streamlit
subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path, '--server.headless', 'true'])