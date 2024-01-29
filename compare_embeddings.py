from langchain.evaluation import load_evaluator
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.evaluation import load_evaluator

def main():
    # get embedding for a word.
    embedding_function = HuggingFaceEmbeddings(model_name='microsoft/mpnet-base')
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector[:4]}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embedding_function)
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()