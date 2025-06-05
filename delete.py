from main import index ,pc
index.delete(filter={"source": {"$eq": "my_pdf_source_01"}})
pc.delete_index('quickstart')