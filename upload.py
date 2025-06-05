# import requests

# def upload_pdf_to_api(pdf_path: str, source_id: str, api_url: str = "http://localhost:8000/upload_pdf/"):
#     with open(pdf_path, "rb") as f:
#         files = {
#             "file": (pdf_path, f, "application/pdf")
#         }
#         data = {
#             "source_id": source_id
#         }
#         response = requests.post(api_url, files=files, data=data)
    
#     if response.status_code == 200:
#         print("✅ Upload success:", response.json())
#     else:
#         print("❌ Upload failed:", response.status_code, response.text)

# if __name__ == "__main__":
#     pdf_file = "knowledge.pdf"  # change this to your PDF path
#     source_identifier = "my_pdf_source_01"  # change this to your source ID
    
#     upload_pdf_to_api(pdf_file, source_identifier)
from PyPDF2 import PdfReader

from main import chunk_text, clean_text, upsert_chunks
pdf_path = "knowledge_compressed.pdf"
source_id = "my_pdf_source_01"
with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        all_chunks = []

        for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if not page_text:
                    continue
                cleaned = clean_text(page_text)
                page_chunks = chunk_text(cleaned)
    
                for i, chunk in enumerate(page_chunks):
                    all_chunks.append({
            "text": chunk,
            "page": page_num,
            "index_in_page": i,
            "total_in_page": len(page_chunks)
        })

        upsert_chunks(all_chunks, source_id)
        print( f"PDF '{source_id}' processed and stored.")

