from main import hybrid_search


chunks = hybrid_search( "What does H1−02−3605 say?", 10)


print("Top relevant chunks:\n")
for i, chunk in enumerate(chunks, start=1):
    print(f"{i}. {chunk}\n")

