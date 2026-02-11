"""Main RAG pipeline for question answering using Iliad API."""

from typing import List, Dict, Any, Optional
import requests
from loguru import logger
from src.rag.vectorstore import VectorStore
from src.rag.embeddings import EmbeddingManager


class RAGPipeline:
    """
    Retrieval Augmented Generation pipeline for answering questions.

    This class combines document retrieval from the vector store with
    generation using AbbVie's Iliad API.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        iliad_api_key: str,
        iliad_api_url: str,
        top_k: int = 5,
    ) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: Initialized VectorStore instance.
            embedding_manager: Initialized EmbeddingManager instance.
            iliad_api_key: API key for Iliad authentication.
            iliad_api_url: URL endpoint for Iliad API.
            top_k: Number of relevant documents to retrieve. Defaults to 5.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.iliad_api_key = iliad_api_key
        self.iliad_api_url = iliad_api_url
        self.top_k = top_k
        logger.info("Initialized RAG pipeline")

    def retrieve_relevant_documents(
        self, query: str, n_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents from the vector store.

        Args:
            query: User's question or query.
            n_results: Number of documents to retrieve. If None, uses self.top_k.

        Returns:
            Dictionary containing retrieved documents, metadatas, and distances.
        """
        n_results = n_results or self.top_k
        logger.info(f"Retrieving {n_results} relevant documents for query")

        try:
            results = self.vector_store.query(query_text=query, n_results=n_results)
            logger.info(
                f"Retrieved {len(results['documents'])} documents with "
                f"distances: {results['distances']}"
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    def format_context(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context for the LLM.

        Args:
            documents: List of retrieved document texts.
            metadatas: List of metadata dictionaries for each document.

        Returns:
            Formatted context string.
        """
        context_parts = []

        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            source_info = []

            if meta.get("title"):
                source_info.append(f"Title: {meta['title']}")
            if meta.get("url"):
                source_info.append(f"URL: {meta['url']}")
            if meta.get("source_type"):
                source_info.append(f"Type: {meta['source_type']}")

            context_parts.append(f"\n--- Document {i} ---")
            if source_info:
                context_parts.append("\n".join(source_info))
            context_parts.append(f"\nContent:\n{doc}\n")

        context = "\n".join(context_parts)
        logger.debug(f"Formatted context with {len(documents)} documents")
        return context

    def call_iliad_api(self, instructions: str, context: str) -> Dict[str, Any]:
        """
        Make an API call to the Iliad service.

        Args:
            instructions: Instructions/prompt for the AI model.
            context: Context information from retrieved documents.

        Returns:
            API response dictionary.
        """
        logger.info("Calling Iliad API")

        headers = {"X-API-Key": self.iliad_api_key}

        messages = [
            {"role": "user", "content": instructions},
            {"role": "user", "content": context},
        ]

        payload = {"messages": messages}

        try:
            response = requests.post(self.iliad_api_url, json=payload, headers=headers)
            response.raise_for_status()

            logger.info("Successfully received response from Iliad API")
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Iliad API: {str(e)}")
            raise

    def generate_answer(
        self, query: str, context: str, return_sources: bool = True
    ) -> str:
        """
        Generate an answer using Iliad API with retrieved context.

        Args:
            query: User's question.
            context: Formatted context from retrieved documents.
            return_sources: Whether to include source references. Defaults to True.

        Returns:
            Generated answer string.
        """
        source_instruction = (
            "Include references to the source documents in your answer."
            if return_sources
            else ""
        )

        instructions = f"""You are a helpful assistant answering questions about data science
projects at AbbVie. Use the provided context to answer the question accurately.

Rules:
- Only use information from the provided context
- Do not make up or infer information not present in the context
- If the context doesn't contain enough information, say so
- Be specific and cite which documents you're referencing
- {source_instruction}

Question: {query}

Please provide a clear, accurate answer based on the context provided."""

        try:
            response = self.call_iliad_api(instructions, context)

            # Extract answer from response
            # Adjust this based on actual Iliad API response format
            if isinstance(response, dict):
                answer = response.get("content", str(response))
            else:
                answer = str(response)

            logger.info("Successfully generated answer")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    def query(
        self, question: str, n_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents and generate answer.

        Args:
            question: User's question.
            n_results: Number of documents to retrieve. If None, uses self.top_k.

        Returns:
            Dictionary containing answer, source documents, and metadata.
        """
        logger.info(f"Processing query: {question}")

        try:
            # Step 1: Retrieve relevant documents
            retrieved = self.retrieve_relevant_documents(question, n_results)

            # Step 2: Format context
            context = self.format_context(
                retrieved["documents"], retrieved["metadatas"]
            )

            # Step 3: Generate answer
            answer = self.generate_answer(question, context)

            # Step 4: Prepare response with sources
            result = {
                "question": question,
                "answer": answer,
                "sources": self._format_sources(retrieved["metadatas"]),
                "retrieved_documents": retrieved["documents"],
                "distances": retrieved["distances"],
            }

            logger.info("Successfully completed RAG query")
            return result

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            raise

    def _format_sources(self, metadatas: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format source information for response.

        Args:
            metadatas: List of metadata dictionaries.

        Returns:
            List of formatted source dictionaries.
        """
        sources = []
        seen_urls = set()

        for meta in metadatas:
            url = meta.get("url")
            if url and url not in seen_urls:
                sources.append(
                    {
                        "title": meta.get("title", "Unknown"),
                        "url": url,
                        "type": meta.get("source_type", "Unknown"),
                    }
                )
                seen_urls.add(url)

        return sources

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.

        Args:
            questions: List of questions to answer.

        Returns:
            List of result dictionaries for each question.
        """
        logger.info(f"Processing batch of {len(questions)} questions")
        results = []

        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            try:
                result = self.query(question)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing question {i}: {str(e)}")
                results.append(
                    {"question": question, "error": str(e), "answer": None}
                )

        logger.info(f"Completed batch processing: {len(results)} results")
        return results
