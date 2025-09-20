import PyPDF2
import tiktoken
from typing import List, Dict, Optional
from pathlib import Path
import re


class PDFProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num}]\n{page_text}\n"
        except Exception as e:
            raise Exception(f"Error reading PDF {file_path}: {e}")

        return self.clean_text(text)

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])\s*', r'\1\n', text)
        text = text.strip()
        return text

    def chunk_text(self, text: str, source: str = "") -> List[Dict[str, str]]:
        tokens = self.encoding.encode(text)
        chunks = []

        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)

            page_ref = ""
            if "[Page " in chunk_text:
                page_match = re.search(r'\[Page (\d+)\]', chunk_text)
                if page_match:
                    page_ref = f", page {page_match.group(1)}"
                    chunk_text = re.sub(r'\[Page \d+\]', '', chunk_text)

            chunks.append({
                "text": chunk_text.strip(),
                "source": f"{source}{page_ref}" if source else f"Document{page_ref}"
            })

        return chunks

    def chunk_pdf(self, file_path: str) -> List[Dict[str, str]]:
        file_path = Path(file_path)
        text = self.extract_text_from_pdf(file_path)
        source_name = file_path.stem
        return self.chunk_text(text, source=source_name)

    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, str]]:
        all_chunks = []
        for pdf_path in pdf_paths:
            try:
                chunks = self.chunk_pdf(pdf_path)
                all_chunks.extend(chunks)
                print(f"Processed {pdf_path}: {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
        return all_chunks

    def estimate_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))