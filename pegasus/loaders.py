"""Document loading utilities for multiple source types."""

from pathlib import Path
from typing import List, Sequence, Union
from urllib.parse import urlparse

from langchain_core.documents import Document as LCDocument
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
)

from .models import PegasusDoc


def _is_url(s: str) -> bool:
    """Check if string is a URL."""
    p = urlparse(s)
    return p.scheme in ("http", "https") and bool(p.netloc)


def load_sources(
    sources: Union[str, Sequence[str]],
    *,
    recursive: bool = True,
    autodetect_encoding: bool = True,
    pdf_extract_images: bool = False,
) -> List[PegasusDoc]:
    """
    Load documents from mixed sources (URLs, directories, files).
    
    Supported:
    - URLs: WebBaseLoader (BeautifulSoup-based HTML extraction)
    - Directories: Recursive loading of .txt/.md/.mdx/.pdf files
    - Files: Direct loading based on extension
    
    Args:
        sources: Single source or list of sources (URLs, paths, directories)
        recursive: Recursively scan directories
        autodetect_encoding: Auto-detect text file encoding
        pdf_extract_images: Extract images from PDFs (requires extra deps)
    
    Returns:
        List of PegasusDoc objects
    """
    if isinstance(sources, str):
        sources = [sources]

    lc_docs: List[LCDocument] = []

    for src in sources:
        if _is_url(src):
            try:
                lc_docs.extend(WebBaseLoader(web_paths=[src]).load())
            except Exception as e:
                print(f"Warning: Failed to load URL {src}: {e}")
            continue

        path = Path(src)
        if path.is_dir():
            # Text-like files
            for pattern in ("**/*.txt", "**/*.md", "**/*.mdx"):
                try:
                    lc_docs.extend(
                        DirectoryLoader(
                            str(path),
                            glob=pattern,
                            recursive=recursive,
                            loader_cls=TextLoader,
                            loader_kwargs={"autodetect_encoding": autodetect_encoding},
                            silent_errors=True,
                        ).load()
                    )
                except Exception as e:
                    print(f"Warning: Failed to load {pattern} from {src}: {e}")

            # PDFs
            try:
                lc_docs.extend(
                    DirectoryLoader(
                        str(path),
                        glob="**/*.pdf",
                        recursive=recursive,
                        loader_cls=PyMuPDFLoader,
                        loader_kwargs={"extract_images": pdf_extract_images},
                        silent_errors=True,
                    ).load()
                )
            except Exception as e:
                print(f"Warning: Failed to load PDFs from {src}: {e}")
            continue

        # Single file
        if not path.exists():
            print(f"Warning: File not found: {src}")
            continue
            
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                lc_docs.extend(PyMuPDFLoader(str(path), extract_images=pdf_extract_images).load())
            else:
                lc_docs.extend(TextLoader(str(path), autodetect_encoding=autodetect_encoding).load())
        except Exception as e:
            print(f"Warning: Failed to load {src}: {e}")

    # Normalize to PegasusDoc
    out: List[PegasusDoc] = []
    for d in lc_docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        out.append(PegasusDoc(text=text, metadata=dict(d.metadata)))
    return out
