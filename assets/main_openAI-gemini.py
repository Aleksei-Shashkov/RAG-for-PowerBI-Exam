import os
import chromadb
from typing import Any
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    Settings,
    PromptTemplate,
    get_response_synthesizer
)
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
import google.generativeai as genai

# --- 1. КОНФИГУРАЦИЯ ---
API_KEY = "AIzaSyAKIlIw4gk6WPTWAVIZIbsLrkvIx_3ijbg"
genai.configure(api_key=API_KEY)

class GeminiLLM(CustomLLM):
    context_window: int = 1000000
    num_output: int = 2048
    model_name: str = "gemini-1.5-flash"

    @property
    def metadata(self) -> LLMMetadata:
        # Возвращаем фиксированные значения, чтобы никто не лез в OpenAI Utils
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=True
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return CompletionResponse(text=response.text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        yield CompletionResponse(text=response.text)

# Установка глобальных настроек
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = GeminiLLM()

def create_rag_system():
    input_dirs = ["instructions/Labs", "instructions/Demos"]
    documents = []
    
    print(">>> Шаг 1: Чтение документов...")
    for directory in input_dirs:
        if os.path.exists(directory):
            reader = SimpleDirectoryReader(input_dir=directory, recursive=True)
            documents.extend(reader.load_data())
    
    if not documents:
        print("ОШИБКА: Файлы не найдены!")
        return None

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("powerbi_exam_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(">>> Шаг 2: Индексация / Загрузка базы...")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

def get_query_engine(index):
    template = (
        "Ты эксперт по Power BI. Отвечай на русском языке, используя контекст:\n"
        "---------------------\n{context_str}\n---------------------\n"
        "Вопрос: {query_str}\nОтвет: "
    )
    
    # СОЗДАЕМ ДВИЖОК ВРУЧНУЮ, ОБХОДЯ АВТОМАТИКУ LLAMA-INDEX
    # Это предотвращает вызов openai_modelname_to_contextsize
    response_synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        text_qa_template=PromptTemplate(template),
        response_mode="compact"
    )
    
    return RetrieverQueryEngine(
        retriever=index.as_retriever(similarity_top_k=5),
        response_synthesizer=response_synthesizer
    )

if __name__ == "__main__":
    pbi_index = create_rag_system()
    if pbi_index:
        engine = get_query_engine(pbi_index)
        print("\n" + "="*30 + "\nСИСТЕМА ГОТОВА!\n" + "="*30)
        while True:
            q = input("\nТвой вопрос: ")
            if q.lower() in ['exit', 'quit', 'выход']: break
            try:
                response = engine.query(q)
                print(f"\nОТВЕТ:\n{response}")
            except Exception as e:
                print(f"Ошибка: {e}")