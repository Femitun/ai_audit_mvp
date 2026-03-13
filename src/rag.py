import os
import re
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-2.0-flash")
RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "2"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))
MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "4"))
RETRY_BASE_SECONDS = float(os.getenv("RAG_RETRY_BASE_SECONDS", "2"))
TEST_MODE = os.getenv("RAG_TEST_MODE", "false").lower() == "true"

# The Prompt Template: This is crucial for "Advisory" tools. 
# It strictly instructs the AI not to hallucinate.
PROMPT_TEMPLATE = """
You are a professional advisory assistant. Answer the question based ONLY on the following context.
If the context does not contain the answer, say "I cannot answer this based on the provided documents." Do not guess.

Context:
{context}

---

Question: {question}
Answer:
"""

def _retry_wait_seconds(error_message: str, attempt: int) -> float:
    retry_match = re.search(r"retry in ([\d.]+)s", error_message, flags=re.IGNORECASE)
    if retry_match:
        return max(float(retry_match.group(1)), RETRY_BASE_SECONDS)
    return (2 ** attempt) * RETRY_BASE_SECONDS

def _run_with_quota_retries(label: str, operation):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return operation()
        except (GoogleGenerativeAIError, ChatGoogleGenerativeAIError) as error:
            message = str(error)
            
            # If we are out of retries, or it's not a quota issue, print the exact error and stop.
            if "RESOURCE_EXHAUSTED" not in message.upper() or attempt == MAX_RETRIES:
                print(f"\n[CRITICAL ERROR] The API returned: {message}\n")
                raise

            wait_seconds = _retry_wait_seconds(message, attempt)
            
            # Print the exact API message so we know WHICH limit we hit
            print(f"\n[DEBUG] API Error Message: {message}")
            print(f"{label} rate-limited. Waiting {wait_seconds:.1f}s before retry {attempt + 1}/{MAX_RETRIES}...")
            time.sleep(wait_seconds)

def main():
    # Loop to keep asking questions
    print("Welcome to the Advisory RAG System. Type 'exit' to quit.")
    while True:
        query_text = input("\nAsk a question about your documents: ")
        if query_text.lower() in ['exit', 'quit']:
            break
            
        if not query_text.strip():
            continue

        query_rag(query_text)

def query_rag(query_text: str):
    # 1. Connect to the existing database
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 2. Search the database for the top 3 most relevant chunks
    # k=3 means it retrieves the 3 best matching text blocks
    results = _run_with_quota_retries(
        "Retrieval",
        lambda: db.similarity_search_with_score(query_text, k=RETRIEVAL_K)
    )

    if not results:
        print("No relevant context found in the database.")
        return

    # 3. Combine the retrieved text chunks into one big context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    context_text = context_text[:MAX_CONTEXT_CHARS]

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    unique_sources = list(set(sources))

    if TEST_MODE:
        print("\n--- TEST MODE ---")
        print("Skipping chat model call. Showing retrieved context only.")
        print("\n--- CONTEXT PREVIEW ---")
        print(context_text[:1200])
        print("\n--- SOURCES ---")
        print(unique_sources)
        return

    # 4. Set up the Prompt and the LLM
    prompt_template = PromptTemplate(
        template=PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Use a configurable Gemini chat model available on your account.
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

    # 5. Get the answer
    print("\nThinking...")
    response_text = _run_with_quota_retries("Generation", lambda: model.invoke(prompt))
    if response_text is None:
        print("No response returned by the model.")
        return

    # 6. Print the answer and the sources used
    print("\n--- ANSWER ---")
    print(response_text.content)
    print("\n--- SOURCES ---")
    # Print unique sources to avoid spamming the console
    print(unique_sources)

if __name__ == "__main__":
    main()