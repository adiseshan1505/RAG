import PyPDF2
import re
from typing import List
from io import BytesIO


class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:

        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"

            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def chunk_text(self, text: str, filename: str) -> List[dict]:
        """Split text into chunks with metadata"""
        clean_text = self.clean_text(text)
        chunks = []

        # Split by pages first
        pages = clean_text.split('--- Page')

        for page_idx, page_content in enumerate(pages):
            if not page_content.strip():
                continue

            page_num = page_idx
            if '---' in page_content:
                try:
                    page_num = int(page_content.split('---')[0].strip()) - 1
                    page_content = page_content.split('---', 1)[1] if '---' in page_content else page_content
                except:
                    pass

            # Further chunk if page is too long
            if len(page_content) <= self.chunk_size:
                chunks.append({
                    'content': page_content.strip(),
                    'metadata': {
                        'filename': filename,
                        'page': page_num,
                        'chunk_id': len(chunks)
                    }
                })
            else:
                # Split long pages into smaller chunks
                words = page_content.split()
                for i in range(0, len(words), self.chunk_size // 10):  # Rough word-based chunking
                    chunk_words = words[i:i + self.chunk_size // 10]
                    chunk_content = ' '.join(chunk_words)

                    if chunk_content.strip():
                        chunks.append({
                            'content': chunk_content.strip(),
                            'metadata': {
                                'filename': filename,
                                'page': page_num,
                                'chunk_id': len(chunks),
                                'sub_chunk': i // (self.chunk_size // 10)
                            }
                        })

        return chunks