import os
from app.logger import LoggerWrapper

logger = LoggerWrapper

"""

"""

def check():
    os.makedirs(config['paths']['documents_dir'], exist_ok=True)
    if not os.path.exists("documents"):
        os.makedirs("documents")
        print("Создана папка 'documents'. Поместите туда .docx файлы.")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama сервер доступен")
        else:
            print("✗ Ollama сервер недоступен")
    except:
        print("✗ Ollama сервер недоступен. Убедитесь, что он запущен.")

    print("\n" + "=" * 60)


