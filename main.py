from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS index
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

print("✅ First Aid Assistant Ready. Ask your question.\n")

while True:
    query = input("You: ")

    if query.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    docs = retriever.invoke(query)

    print("\n🔍 Retrieved Information:\n")
    for doc in docs:
        print(doc.page_content)
        print("------")

    print("\n============================\n")