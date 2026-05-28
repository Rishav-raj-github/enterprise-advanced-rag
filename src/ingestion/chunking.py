import re
import uuid
import os

class ParentChildChunker:
    """
    Implements a premium Parent-Child chunking strategy.
    Splits text into larger Parent chunks, and then subdivides those 
    into overlapping Child chunks.
    """
    def __init__(self, parent_size: int = 1000, parent_overlap: int = 200, 
                 child_size: int = 250, child_overlap: int = 50):
        self.parent_size = parent_size
        self.parent_overlap = parent_overlap
        self.child_size = child_size
        self.child_overlap = child_overlap

    def split_text_overlapping(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """
        Splits a text string into chunks of standard size with overlapping borders.
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start += chunk_size - overlap
            
        return chunks

    def process_document(self, content: str, source_name: str) -> dict:
        """
        Processes a single document:
        - Extracts parent chunks.
        - Subdivides each parent into children.
        - Assigns unique parent IDs and associates child metadata.
        
        Returns:
            dict containing lists of 'parents' and 'children' dictionaries.
        """
        parents_data = []
        children_data = []
        
        # Split document into parent chunks
        raw_parents = self.split_text_overlapping(content, self.parent_size, self.parent_overlap)
        
        for p_idx, parent_text in enumerate(raw_parents):
            parent_id = f"parent_{uuid.uuid4().hex[:12]}_{p_idx}"
            
            # Save parent details
            parents_data.append({
                "parent_id": parent_id,
                "text": parent_text,
                "metadata": {
                    "source": source_name,
                    "type": "parent",
                    "chunk_index": p_idx
                }
            })
            
            # Split parent into child chunks
            raw_children = self.split_text_overlapping(parent_text, self.child_size, self.child_overlap)
            
            for c_idx, child_text in enumerate(raw_children):
                child_id = f"child_{uuid.uuid4().hex[:12]}_{c_idx}"
                
                # Save child details, carrying the parent_id inside metadata
                children_data.append({
                    "child_id": child_id,
                    "text": child_text,
                    "metadata": {
                        "source": source_name,
                        "type": "child",
                        "parent_id": parent_id,
                        "chunk_index": c_idx
                    }
                })
                
        return {
            "parents": parents_data,
            "children": children_data
        }

def load_and_chunk_corpus(corpus_dir: str, chunker: ParentChildChunker = None) -> dict:
    """
    Scans a directory of text/markdown files and applies Parent-Child chunking.
    """
    if chunker is None:
        chunker = ParentChildChunker()
        
    all_parents = []
    all_children = []
    
    if not os.path.exists(corpus_dir):
        print(f"[Chunker Warning] Corpus directory does not exist: {corpus_dir}")
        return {"parents": [], "children": []}
        
    for filename in os.listdir(corpus_dir):
        if filename.endswith((".md", ".txt")):
            file_path = os.path.join(corpus_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            docs = chunker.process_document(content, filename)
            all_parents.extend(docs["parents"])
            all_children.extend(docs["children"])
            
    print(f"[Chunker] Successfully chunked corpus. Generated {len(all_parents)} parent chunks and {len(all_children)} child chunks.")
    return {
        "parents": all_parents,
        "children": all_children
    }
