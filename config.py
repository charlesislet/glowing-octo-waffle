import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration 
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("LLM_API_BASE").rstrip("/")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

# Embedding Model Configuration
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE").rstrip("/")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
VECTOR_DIM = 1024  # BGE-M3 output dimension

# Milvus Configuration
MILVUS_BASE = os.getenv("MILVUS_BASE").rstrip("/")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME")
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
COLLECTION_NAME = "pdf_manual_collection"

# RAG Configuration
TOP_K = 3  # Number of documents to retrieve
MAX_RETRIES = 1  # Max retries for API calls

# Paths
BASE_DIR = Path(__file__).parent
TEMP_IMAGES_DIR = BASE_DIR / "temp_images"
PDFS_DIR = BASE_DIR / "pdfs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Static Files Serving
# Base URL used to construct image links for the frontend/LLM.
# If not provided via env `BASE_IMAGE_URL`, default to the app server.
# Examples:
#   http://10.12.100.164:8002/static
#   http://10.12.100.164:8002/static/pages
BASE_IMAGE_URL = os.getenv("BASE_IMAGE_URL", "http://10.12.100.164:8002/static").rstrip("/")

# PDF Processing
# PDF_DPI = 300  # Resolution for PDF to image conversion

# Prompts (from project.md)
SUMMARY_PROMPT = """你是一個專業的文件分析助手。請仔細閱讀這個文件頁面，並提供完整的結構化摘要。

請按以下格式輸出：

【頁面主題】
簡述這一頁的主要內容主題

【詳細內容】
- 列出所有重要資訊、數據、定義
- 如有表格，請完整轉錄表格內容（用 markdown 表格格式）
- 如有流程圖或圖表，請描述其內容和關係

【關鍵術語】
列出頁面中出現的專業術語或關鍵字（用逗號分隔）

【備註】
-任何需要特別注意的事項
-不要有前面的罐頭回復"""

QUERY_PROMPT_TEMPLATE = """你是一個專業的文件問答助手。根據提供的文件頁面圖像，回答使用者的問題。

使用者問題：{query}

請注意：
- 只根據圖像中的內容回答，不要編造資訊
- 如果圖像中沒有相關資訊，請明確說明
- 如果答案涉及表格或數據，請準確引用
- 用繁體中文回答"""