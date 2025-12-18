"""
LangRAG Integration - Advanced Retrieval Augmented Generation for Python Backend
Provides intelligent knowledge retrieval and context management for AI agents
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
from .config import get_config


class VectorStoreManager:
    """Manages vector stores for knowledge bases"""
    
    def __init__(self):
        config = get_config()
        
        self.embeddings = AzureOpenAIEmbeddings(
            openai_api_key=config.azure.api_key,
            azure_endpoint=config.azure.endpoint,
            deployment=config.azure.embedding_deployment,
            openai_api_version=config.azure.api_version
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            length_function=len
        )
        
        self.vector_stores: Dict[str, Any] = {}
        self.db_type = config.rag.vector_db_type
    
    async def create_vector_store(
        self,
        store_id: str,
        documents: List[Document]
    ) -> Any:
        """Create a new vector store from documents"""
        print(f"📚 Creating vector store '{store_id}' with {len(documents)} documents...")
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        print(f"✂️  Split into {len(split_docs)} chunks")
        
        # Create vector store based on configured type
        if self.db_type == "chromadb":
            vector_store = await asyncio.to_thread(
                Chroma.from_documents,
                documents=split_docs,
                embedding=self.embeddings,
                collection_name=store_id
            )
        elif self.db_type == "faiss":
            vector_store = await asyncio.to_thread(
                FAISS.from_documents,
                documents=split_docs,
                embedding=self.embeddings
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.db_type}")
        
        self.vector_stores[store_id] = vector_store
        print(f"✅ Vector store '{store_id}' created successfully")
        
        return vector_store
    
    async def add_documents(
        self,
        store_id: str,
        documents: List[Document]
    ) -> None:
        """Add documents to existing vector store"""
        vector_store = self.vector_stores.get(store_id)
        
        if not vector_store:
            await self.create_vector_store(store_id, documents)
            return
        
        split_docs = self.text_splitter.split_documents(documents)
        await asyncio.to_thread(vector_store.add_documents, split_docs)
        print(f"➕ Added {len(split_docs)} chunks to vector store '{store_id}'")
    
    async def similarity_search(
        self,
        store_id: str,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """Perform similarity search"""
        vector_store = self.vector_stores.get(store_id)
        
        if not vector_store:
            raise ValueError(f"Vector store '{store_id}' not found")
        
        results = await asyncio.to_thread(
            vector_store.similarity_search,
            query,
            k=k
        )
        
        return results
    
    async def similarity_search_with_score(
        self,
        store_id: str,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with relevance scores"""
        vector_store = self.vector_stores.get(store_id)
        
        if not vector_store:
            raise ValueError(f"Vector store '{store_id}' not found")
        
        results = await asyncio.to_thread(
            vector_store.similarity_search_with_score,
            query,
            k=k
        )
        
        return results
    
    def get_vector_store(self, store_id: str) -> Optional[Any]:
        """Get vector store by ID"""
        return self.vector_stores.get(store_id)


class RAGChain:
    """Retrieval Augmented Generation pipeline"""
    
    def __init__(self, vector_store_manager: VectorStoreManager, temperature: float = 0.7):
        self.vector_store_manager = vector_store_manager
        
        config = get_config()
        self.llm = AzureChatOpenAI(
            openai_api_key=config.azure.api_key,
            azure_endpoint=config.azure.endpoint,
            deployment_name=config.azure.deployment_name,
            openai_api_version=config.azure.api_version,
            temperature=temperature,
            max_tokens=4000
        )
        
        self.config = config
    
    async def query(
        self,
        store_id: str,
        question: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query knowledge base and generate answer"""
        # Retrieve relevant documents
        relevant_docs = await self.vector_store_manager.similarity_search(
            store_id,
            question,
            k=self.config.rag.top_k
        )
        
        # Build context from documents
        context = "\n\n".join(
            f"[Source {i+1}] {doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        )
        
        # Create prompt
        default_prompt = "You are a helpful AI assistant that answers questions based on the provided context."
        prompt = f"""{system_prompt or default_prompt}

Context:
{context}

Question: {question}

Answer based on the context provided above. If the answer is not in the context, say so clearly."""
        
        # Generate response
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        
        return {
            "answer": response.content,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in relevant_docs
            ],
            "num_sources": len(relevant_docs)
        }
    
    async def query_with_score(
        self,
        store_id: str,
        question: str,
        system_prompt: Optional[str] = None,
        score_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Query with relevance score filtering"""
        # Retrieve documents with scores
        docs_with_scores = await self.vector_store_manager.similarity_search_with_score(
            store_id,
            question,
            k=self.config.rag.top_k
        )
        
        # Filter by threshold
        filtered_docs = [
            (doc, score) for doc, score in docs_with_scores
            if score >= score_threshold
        ]
        
        if not filtered_docs:
            return {
                "answer": "No relevant information found in the knowledge base with sufficient confidence.",
                "sources": [],
                "num_sources": 0
            }
        
        # Build context with relevance scores
        context = "\n\n".join(
            f"[Source {i+1}] (Relevance: {score*100:.1f}%) {doc.page_content}"
            for i, (doc, score) in enumerate(filtered_docs)
        )
        
        default_prompt = "You are a helpful AI assistant. Answer based on the provided context and cite sources using [Source numbers]."
        prompt = f"""{system_prompt or default_prompt}

Context:
{context}

Question: {question}

Answer based on the context above. Cite sources using [Source N] format."""
        
        response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        
        return {
            "answer": response.content,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score)
                }
                for doc, score in filtered_docs
            ],
            "num_sources": len(filtered_docs)
        }


class ExamKnowledgeBaseManager:
    """Manages knowledge bases for government examinations"""
    
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.rag_chain = RAGChain(self.vector_store_manager)
    
    async def load_exam_knowledge_base(
        self,
        examination_id: str,
        syllabus_docs: Optional[List[Dict]] = None,
        reference_materials: Optional[List[Dict]] = None,
        previous_papers: Optional[List[Dict]] = None,
        current_affairs: Optional[List[Dict]] = None
    ) -> None:
        """Load comprehensive knowledge base for an examination"""
        print(f"📖 Loading knowledge base for examination: {examination_id}")
        
        documents: List[Document] = []
        
        # Add syllabus documents
        if syllabus_docs:
            for item in syllabus_docs:
                documents.append(Document(
                    page_content=str(item.get('content', '')),
                    metadata={
                        'type': 'syllabus',
                        'examination_id': examination_id,
                        'subject': item.get('subject', 'general')
                    }
                ))
        
        # Add reference materials
        if reference_materials:
            for item in reference_materials:
                documents.append(Document(
                    page_content=str(item.get('content', '')),
                    metadata={
                        'type': 'reference',
                        'examination_id': examination_id,
                        'title': item.get('title', 'untitled')
                    }
                ))
        
        # Add previous year papers
        if previous_papers:
            for item in previous_papers:
                documents.append(Document(
                    page_content=str(item.get('content', '')),
                    metadata={
                        'type': 'previous_papers',
                        'examination_id': examination_id,
                        'year': item.get('year', 'unknown')
                    }
                ))
        
        # Add current affairs (only 1+ day old)
        if current_affairs:
            cutoff_date = datetime.now() - timedelta(days=1)
            for item in current_affairs:
                published_date = item.get('published_date')
                if published_date and published_date < cutoff_date:
                    documents.append(Document(
                        page_content=f"{item.get('title', '')}\n\n{item.get('content', '')}",
                        metadata={
                            'type': 'current_affairs',
                            'examination_id': examination_id,
                            'published_date': str(published_date),
                            'category': item.get('category', 'general')
                        }
                    ))
        
        if not documents:
            print(f"⚠️  No documents found for examination {examination_id}")
            return
        
        # Create vector store
        await self.vector_store_manager.create_vector_store(
            examination_id,
            documents
        )
        
        print(f"✅ Loaded {len(documents)} documents into knowledge base")
    
    async def query_knowledge_base(
        self,
        examination_id: str,
        question: str,
        include_current_affairs: bool = True
    ) -> Dict[str, Any]:
        """Query exam knowledge base"""
        system_prompt = (
            "You are an expert in government examinations. "
            "Answer based on syllabus, reference materials, and current affairs. "
            "Include relevant current affairs when appropriate."
            if include_current_affairs else
            "You are an expert in government examinations. "
            "Answer based on syllabus and reference materials only."
        )
        
        return await self.rag_chain.query_with_score(
            examination_id,
            question,
            system_prompt,
            score_threshold=self.rag_chain.config.rag.similarity_threshold
        )
    
    async def generate_question_context(
        self,
        examination_id: str,
        subject: str,
        include_current_affairs: bool = True
    ) -> str:
        """Generate context for question generation"""
        query = f"Provide key concepts and topics for {subject} suitable for examination questions"
        
        result = await self.query_knowledge_base(
            examination_id,
            query,
            include_current_affairs
        )
        
        return result["answer"]
    
    async def update_knowledge_base(
        self,
        examination_id: str,
        new_documents: List[Document]
    ) -> None:
        """Update existing knowledge base with new documents"""
        print(f"🔄 Updating knowledge base for examination: {examination_id}")
        
        await self.vector_store_manager.add_documents(
            examination_id,
            new_documents
        )
        
        print(f"✅ Added {len(new_documents)} new documents")


# Singleton instances
_vector_store_manager_instance: Optional[VectorStoreManager] = None
_exam_kb_manager_instance: Optional[ExamKnowledgeBaseManager] = None


def get_vector_store_manager() -> VectorStoreManager:
    """Get singleton vector store manager"""
    global _vector_store_manager_instance
    if _vector_store_manager_instance is None:
        _vector_store_manager_instance = VectorStoreManager()
    return _vector_store_manager_instance


def get_exam_knowledge_base_manager() -> ExamKnowledgeBaseManager:
    """Get singleton exam knowledge base manager"""
    global _exam_kb_manager_instance
    if _exam_kb_manager_instance is None:
        _exam_kb_manager_instance = ExamKnowledgeBaseManager()
    return _exam_kb_manager_instance