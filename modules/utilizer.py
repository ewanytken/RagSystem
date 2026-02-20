
import os
import shutil
from typing import List, Optional, Any
from app.logger import LoggerWrapper
import yaml

log = LoggerWrapper()

class Utils:

    @staticmethod
    def get_config_file(config_path: str = "config.yaml") -> Any:
        try:
            return yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
        except FileNotFoundError as e:
            log(f"Config file not found: 40 {e}")

    @staticmethod
    def get_docx_files(directory: str) -> List[str]:
        """Получение списка DOCX файлов в директории"""
        docx_files = []
        for filename in os.listdir(directory):
            if filename.lower().endswith('.docx'):
                docx_files.append(os.path.join(directory, filename))
        return docx_files


    def copy_documents(source_dir: str, target_dir: str = "documents"):
        """Копирование документов в рабочую директорию"""
        os.makedirs(target_dir, exist_ok=True)

        docx_files = Utils.get_docx_files(source_dir)

        for file_path in docx_files:
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, filename)

            try:
                shutil.copy2(file_path, target_path)
                print(f"Скопирован: {filename}")
            except Exception as e:
                print(f"Ошибка копирования {filename}: {e}")

        print(f"\nВсего скопировано файлов: {len(docx_files)}")


    def clear_directory(directory: str):
        """Очистка директории"""
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Ошибка удаления {file_path}: {e}")
            print(f"Директория {directory} очищена")