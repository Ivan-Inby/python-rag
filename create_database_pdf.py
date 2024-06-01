import os
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.document_loaders import PyPDFLoader

# Функция для разбивки текста на куски (chunks) заданного размера с перекрытием
def chunk_text_with_overlap(text, chunk_size, chunk_overlap):
    words = text.split()
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        yield ' '.join(words[start:end])
        start += chunk_size - chunk_overlap

# Функция для извлечения текста и номеров страниц из PDF файла с использованием PyPDFLoader
def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = []
    page_numbers = []
    for i, doc in enumerate(documents):
        text.append(doc.page_content)
        # Добавляем номер страницы (i+1, так как нумерация страниц начинается с 1)
        page_numbers.extend([i+1] * len(doc.page_content.split()))  # Каждое слово получает номер страницы
    return " ".join(text), page_numbers

# Указание пути к папке, содержащей PDF файлы
folder_path = "data"
chunk_size = 100  # Задание размера chunks
chunk_overlap = 20  # Задание перекрытия chunks (20 слов)
DB_path = "knowledge_base"  # Путь к базе данных
collection_name = "knowledge_base_collection"  # Имя коллекции

# Создание клиента Chroma для работы с постоянным хранилищем (PersistentClient).
chroma_client = chromadb.PersistentClient(path=DB_path)

# Получение или создание коллекции
collection = chroma_client.get_or_create_collection(name=collection_name)

# Инициализация модели для word embeddings
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Списки для хранения документов, их идентификаторов, эмбеддингов и метаданных
documents = []  # Список для хранения текстовых кусков
ids = []  # Список для хранения уникальных идентификаторов кусков
embeddings = []  # Список для хранения векторных представлений кусков
metadata = []  # Список для хранения метаданных
total_chunks = 0  # Переменная для подсчета общего количества чанков
total_documents = 0  # Переменная для подсчета общего количества документов

# Проход по всем файлам в указанной папке
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):  # Проверка, что файл имеет расширение .pdf
        print(f"  Filename: {filename}") # Выводим название обрабатываемого файла
        file_path = os.path.join(folder_path, filename)
        content, page_numbers = extract_text_from_pdf(file_path)  # Извлечение текста и номеров страниц из PDF файла
        chunks = list(chunk_text_with_overlap(content, chunk_size, chunk_overlap))  # Разбивка текста на chunks
        total_chunks += len(chunks)  # Увеличение общего количества чанков
        total_documents += 1  # Увеличение общего количества документов
        for i, chunk in enumerate(chunks):
            documents.append(chunk)  # Добавление текстового куска
            ids.append(f"{filename}_chunk_{i}")  # Создание уникального идентификатора для каждого куска
            embeddings.append(model.encode(chunk).tolist())  # Векторизация чанка и преобразование в список
            # Определение номера страницы для текущего чанка
            chunk_words = chunk.split()
            chunk_page_numbers = [page_numbers[j] for j in range(i * (chunk_size - chunk_overlap), i * (chunk_size - chunk_overlap) + len(chunk_words))]
            page_number = max(set(chunk_page_numbers), key=chunk_page_numbers.count)  # Определение самой часто встречающейся страницы в чанке
            metadata.append({"filename": filename, "chunk_id": f"{filename}_chunk_{i}", "page_number": page_number})  # Добавление метаданных
            print(f"  Page_number: {page_number}") # Выводим номер обрабатываемой страницы
            

# Обновление или вставка (upsert) документов в коллекцию с эмбеддингами и метаданными.
collection.upsert(
    embeddings=embeddings,
    documents=documents,
    ids=ids,
    metadatas=metadata
)

# Вывод сообщения о количестве документов и чанков
print(f"Разбили документов - {total_documents} на {total_chunks} чанков.")

# Печать десятого чанка с метаданными, если он существует
if total_chunks >= 10:
    index = 9  # Индекс десятого чанка (нумерация начинается с 0)
    print("\nДесятый чанк с метаданными:")
    print(f"Чанк 10:")
    print(f"  ID: {metadata[index]['chunk_id']}")
    print(f"  Filename: {metadata[index]['filename']}")
    print(f"  Page Number: {metadata[index]['page_number']}")
    print(f"  Content: {documents[index][:100]}...")  # Печать первых 100 символов содержимого чанка
    print("\n" + "-"*50 + "\n")
else:
    print("Всего чанков меньше 10, вывод десятого чанка невозможен.")
