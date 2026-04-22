import os
import datetime
import chromadb
from typing import Any
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    Settings,
    PromptTemplate
)
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 1. НАСТРОЙКИ ---
# Используем легкую модель gemma2:2b для экономии RAM (3.7 ГБ)
Settings.llm = Ollama(
    model="gemma2:2b", 
    request_timeout=600.0,  # Увеличиваем ожидание до 10 минут
    context_window=2048,    # Ограничиваем "память" модели, чтобы она не тормозила
    temperature=0.1         # Делаем ответы более четкими и быстрыми
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def create_rag_system():
    persist_dir = "./chroma_db"
    input_dirs = ["instructions/Labs", "instructions/Demos"]
    
    # Инициализируем клиент базы данных
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection("powerbi_exam_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Проверяем, есть ли уже данные в базе
    if chroma_collection.count() > 0:
        print(">>> Шаг 1: Найдена готовая база. Мгновенная загрузка...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    else:
        print(">>> Шаг 1: База пуста. Начинаю индексацию (это может занять время)...")
        documents = []
        for directory in input_dirs:
            if os.path.exists(directory):
                reader = SimpleDirectoryReader(input_dir=directory, recursive=True)
                documents.extend(reader.load_data())
        
        if not documents:
            print("ОШИБКА: Документы не найдены!")
            return None
            
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        print(">>> Индексация завершена.")
        
    return index

def get_query_engine(index):
    # Жесткая инструкция отвечать на русском
    template = (
        "ИНСТРУКЦИЯ: Ты эксперт по Power BI. Ты ВСЕГДА отвечаешь только на РУССКОМ языке.\n"
        "Используй только предоставленный текст для ответа:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Вопрос: {query_str}\n\n"
        "Твой ответ на РУССКОМ языке: "
    )
    # streaming=True позволяет видеть текст по мере генерации (ускоряет восприятие)
    return index.as_query_engine(
        text_qa_template=PromptTemplate(template),
        similarity_top_k=3,
        streaming=True 
    )

# --- ОБНОВЛЕННЫЙ БЛОК ЗАПУСКА ---
if __name__ == "__main__":
    pbi_index = create_rag_system()
    
    if pbi_index:
        engine = get_query_engine(pbi_index)
        print("\n" + "="*40)
        print("ЛОКАЛЬНЫЙ БОТ ГОТОВ! (Gemma 2 2B + Ollama)")
        print("Пиши 'exit' для выхода.")
        print("="*40)
        
        while True:
            query = input("\nТвой вопрос: ")
            if query.lower() in ['exit', 'quit', 'выход']:
                break
            
            if not query.strip():
                continue
                
            try:
                # Потоковый ответ для скорости
                print("\nОТВЕТ:")
                streaming_response = engine.query(query)
                streaming_response.print_response_stream()
                print("\n") # Перенос строки после завершения стриминга
                
                # Сохраняем историю в файл
                with open("query_history.txt", "a", encoding="utf-8") as f:
                    f.write(f"Дата: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Вопрос: {query}\n")
                    f.write(f"Ответ: {streaming_response.response_txt}\n") # Берем накопленный текст
                    f.write("=======================================================\n")
                
            except Exception as e:
                print(f"\nОшибка: {e}")