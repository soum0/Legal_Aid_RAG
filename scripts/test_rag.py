# scripts/test_rag.py

from src.lc_rag_chain import build_rag_chain

if __name__ == "__main__":

    rag_chain = build_rag_chain()

    while True:
        question = input("\nAsk a constitutional question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        result = rag_chain.invoke(question)

        # If it's AIMessage, extract only content
        if hasattr(result, "content"):
            print("\nAnswer:\n")
            print(result.content)
        else:
            print("\nAnswer:\n")
            print(result)

