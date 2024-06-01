import os
import argparse
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

# Установить переменную окружения для подавления предупреждения о параллелизме
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Указание пути к базе данных и имени коллекции
DB_path = "knowledge_base"
collection_name = "knowledge_base_collection"

# Создание клиента Chroma для работы с постоянным хранилищем (PersistentClient)
chroma_client = chromadb.PersistentClient(path=DB_path)

# Получение существующей коллекции или создание новой, если коллекции не существует
collection = chroma_client.get_or_create_collection(name=collection_name)

# Количество передаваемых чанков в промпт
number_results=3

# Ограничение дистанции найденных чанков от заданного вопроса
distance=0.4

# Инициализация модели для word embeddings (векторных представлений слов)
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Шаблон для генерации вывода на основе контекста и вопроса
PROMPT_TEMPLATE = """
Ответь на вопрос, базируясь только на этом контексте:

{context}

---

Ответь на вопрос, используя только контекст: {question}
"""

# Функция для поиска и генерации промпта на основе контекста
def search_and_generate_prompt(question, collection, model, n_results=number_results, distance_threshold=distance):
    # Оцифровка текста запроса и преобразование его в список
    query_embedding = model.encode([question])[0].tolist()

    # Запрос к коллекции для поиска документов, которые наиболее близки к запросу
    results = collection.query(
        query_embeddings=[query_embedding],  # Векторное представление текста запроса в виде списка
        n_results=n_results  # Количество возвращаемых результатов
    )

    # Фильтрация результатов по порогу расстояния (distance_threshold)
    filtered_results = [
        (doc, metadata, dist) for doc, metadata, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]) if dist <= distance_threshold
    ]

    # Проверка наличия релевантных результатов
    if not filtered_results:
        return "В базе знаний не нашлось необходимой информации.", ""

    # Формирование контекста из отфильтрованных результатов
    context = "\n\n".join([doc for doc, _, _ in filtered_results])

    # Формирование списка метаданных для вывода
    metadata_info = "\n".join([
        f"File: {metadata['filename']}, Page: {metadata.get('page_number', 'N/A')}" 
        for _, metadata, _ in filtered_results if metadata is not None
    ])

    # Подстановка контекста и вопроса в шаблон
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    return prompt, metadata_info

# Функция для отправки промпта языковой модели и получения ответа
def send_prompt_to_model(prompt):
    # Подключение к локальному серверу LM Studio
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # Отправка запроса к языковой модели
    completion = client.chat.completions.create(
        model="IlyaGusev/saiga_mistral_7b_gguf",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    # Возврат ответа от модели
    return completion.choices[0].message.content

# Основная функция для обработки аргументов и выполнения поиска
def main():
    # Создание парсера аргументов командной строки
    parser = argparse.ArgumentParser(description='Generate prompt based on context from ChromaDB collection.')
    parser.add_argument('question', type=str, help='The question to search for in the collection')
    args = parser.parse_args()
    
    question = args.question
    # Генерация промпта и получение метаданных
    prompt, metadata_info = search_and_generate_prompt(question, collection, model)
    if "В базе знаний не нашлось необходимой информации." in prompt:
        print(prompt)
    else:
        # Отправка промпта модели и получение ответа
        response = send_prompt_to_model(prompt)
        print(response)
        print("\nИспользованные источники:")
        print(metadata_info)

# Запуск основной функции, если скрипт выполняется напрямую
if __name__ == "__main__":
    main()
