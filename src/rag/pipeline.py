"""Main RAG pipeline for question answering using Iliad API."""

from typing import List, Dict, Any, Optional
import requests
from loguru import logger
from rag.vectorstore import VectorStore
from rag.embeddings import EmbeddingManager
from rag.query_processor import QueryProcessor, ProcessedQuery
from rag.reranker import DocumentReranker, RankingWeights
from rag.project_vectorstore import ProjectVectorStore


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
        use_reranking: bool = True,
        ranking_weights: Optional[RankingWeights] = None,
        project_store: Optional[ProjectVectorStore] = None,
        enable_two_stage_rag: bool = True,
        project_retrieval_top_k: int = 3,
    ) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: Initialized VectorStore instance.
            embedding_manager: Initialized EmbeddingManager instance.
            iliad_api_key: API key for Iliad authentication.
            iliad_api_url: URL endpoint for Iliad API.
            top_k: Number of relevant documents to retrieve. Defaults to 5.
            use_reranking: Whether to use composite scoring re-ranking. Defaults to True.
            ranking_weights: Custom weights for re-ranking. Uses defaults if None.
            project_store: Optional ProjectVectorStore for two-stage retrieval.
            enable_two_stage_rag: Whether to use two-stage project-filtered retrieval.
            project_retrieval_top_k: Number of projects to identify in stage 1.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.iliad_api_key = iliad_api_key
        self.iliad_api_url = iliad_api_url
        self.top_k = top_k
        self.use_reranking = use_reranking

        # Two-stage RAG configuration
        self.project_store = project_store
        self.enable_two_stage_rag = enable_two_stage_rag and project_store is not None
        self.project_retrieval_top_k = project_retrieval_top_k

        # Initialize query processor for keyword extraction
        self.query_processor = QueryProcessor()

        # Initialize re-ranker if enabled
        if use_reranking:
            self.reranker = DocumentReranker(
                weights=ranking_weights,
                embedding_manager=embedding_manager,
            )
        else:
            self.reranker = None

        logger.info(
            f"Initialized RAG pipeline (reranking: {use_reranking}, "
            f"two_stage: {self.enable_two_stage_rag})"
        )

    def identify_relevant_projects(
        self, query: str, n_results: Optional[int] = None
    ) -> List[str]:
        """
        Stage 1: Identify relevant main projects for the query.

        Args:
            query: User's question or query.
            n_results: Number of projects to retrieve. Defaults to project_retrieval_top_k.

        Returns:
            List of relevant main_project names.
        """
        if not self.project_store:
            logger.warning("Project store not available for two-stage retrieval")
            return []

        n_results = n_results or self.project_retrieval_top_k

        logger.info(f"Stage 1: Identifying top {n_results} relevant projects")

        try:
            project_results = self.project_store.query_projects(query, n_results=n_results)

            if not project_results:
                logger.warning("No projects found in stage 1")
                return []

            # DEBUG: Log detailed project results with similarity scores
            logger.info("--- Stage 1 Project Retrieval Results ---")
            for i, p in enumerate(project_results, 1):
                logger.info(
                    f"  {i}. {p.get('main_project', 'Unknown')}: "
                    f"similarity={p.get('similarity', 0):.4f}, "
                    f"pages={p.get('page_count', 0)}"
                )
            logger.info("-" * 40)

            project_names = [p.get("main_project", "") for p in project_results]
            project_names = [p for p in project_names if p]

            logger.info(f"Stage 1 identified projects: {project_names}")
            return project_names

        except Exception as e:
            logger.error(f"Error in stage 1 project identification: {str(e)}")
            return []

    def retrieve_filtered_by_project(
        self,
        query: str,
        project_names: List[str],
        n_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Stage 2: Retrieve documents filtered to specific projects.

        Args:
            query: User's question or query.
            project_names: List of project names to filter by.
            n_results: Number of documents to retrieve per project.

        Returns:
            Dictionary containing retrieved documents, metadatas, and distances.
        """
        n_results = n_results or self.top_k

        logger.info(f"Stage 2: Retrieving from projects: {project_names}")

        try:
            # Use the vector store's filter capability
            results = self.vector_store.query_with_filter(
                query_text=query,
                n_results=n_results,
                filter_field="main_project",
                filter_values=project_names,
            )

            logger.info(f"Stage 2 retrieved {len(results['documents'])} documents")
            return results

        except Exception as e:
            logger.error(f"Error in stage 2 filtered retrieval: {str(e)}")
            # Fallback to unfiltered query
            logger.warning("Falling back to unfiltered query")
            return self.vector_store.query(query_text=query, n_results=n_results)

    def retrieve_relevant_documents(
        self, query: str, n_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents from the vector store with optional re-ranking.

        Uses two-stage retrieval if enabled:
        - Stage 1: Identify relevant projects using project vector store
        - Stage 2: Retrieve chunks filtered to those projects

        Args:
            query: User's question or query.
            n_results: Number of documents to retrieve. If None, uses self.top_k.

        Returns:
            Dictionary containing retrieved documents, metadatas, and distances.
        """
        n_results = n_results or self.top_k

        # Process query to extract keywords
        processed_query = self.query_processor.process_query(query)
        logger.info(
            f"Processed query - Keywords: {processed_query.keywords}, "
            f"Projects: {processed_query.potential_project_names}, "
            f"People: {processed_query.potential_person_names}, "
            f"Comparative: {processed_query.is_comparative}"
        )

        # Retrieve more documents initially if re-ranking is enabled
        # This allows re-ranking to potentially surface better matches
        initial_retrieve = n_results * 3 if self.use_reranking else n_results

        try:
            # Use two-stage retrieval if enabled
            if self.enable_two_stage_rag:
                logger.info("Using two-stage RAG retrieval")

                # Stage 1: Identify relevant projects
                relevant_projects = self.identify_relevant_projects(query)

                if relevant_projects:
                    # Stage 2: Retrieve filtered by projects
                    results = self.retrieve_filtered_by_project(
                        query, relevant_projects, n_results=initial_retrieve
                    )
                    results['identified_projects'] = relevant_projects
                else:
                    # Fallback to unfiltered query if no projects identified
                    logger.warning("No projects identified, using unfiltered query")
                    results = self.vector_store.query(
                        query_text=query, n_results=initial_retrieve
                    )
                    results['identified_projects'] = []
            else:
                logger.info(
                    f"Retrieving {initial_retrieve} documents for query "
                    f"(reranking: {self.use_reranking})"
                )
                results = self.vector_store.query(
                    query_text=query, n_results=initial_retrieve
                )

            logger.info(
                f"Retrieved {len(results['documents'])} documents with "
                f"distances: {results['distances'][:5]}..."  # Log first 5 distances
            )

            # DEBUG: Log detailed top 5 document info
            logger.info("--- Top 5 Retrieved Documents ---")
            for i in range(min(5, len(results['documents']))):
                meta = results['metadatas'][i] if i < len(results['metadatas']) else {}
                dist = results['distances'][i] if i < len(results['distances']) else 0
                logger.info(
                    f"  {i+1}. Title: {meta.get('title', 'Unknown')[:50]}"
                )
                logger.info(
                    f"      Project: {meta.get('main_project', 'N/A')}, "
                    f"Distance: {dist:.4f}, "
                    f"Chunk: {meta.get('chunk_index', 'N/A')}"
                )
            logger.info("-" * 40)

            # Apply re-ranking if enabled
            if self.use_reranking and self.reranker and len(results['documents']) > 0:
                logger.info("Applying composite scoring re-ranking")

                # Generate query embedding for title similarity
                query_embedding = self.embedding_manager.generate_embedding(query)

                # Re-rank documents
                scored_docs = self.reranker.rerank(
                    documents=results['documents'],
                    metadatas=results['metadatas'],
                    ids=results['ids'],
                    distances=results['distances'],
                    processed_query=processed_query,
                    query_embedding=query_embedding,
                )

                # Extract top results after re-ranking
                results = self.reranker.extract_reranked_results(scored_docs, n_results)

                logger.info(
                    f"Re-ranked results. Top composite scores: "
                    f"{results.get('composite_scores', [])[:3]}"
                )
            else:
                # Trim to requested number if not re-ranking
                results = {
                    'documents': results['documents'][:n_results],
                    'metadatas': results['metadatas'][:n_results],
                    'distances': results['distances'][:n_results],
                    'ids': results['ids'][:n_results],
                }

            # Store processed query for potential use in generation
            results['processed_query'] = processed_query

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
            API response dictionary with structure:
            {
                "completion": {
                    "content": "The AI response text"
                }
            }
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

            # Log the response structure for debugging
            logger.debug(f"Iliad API response type: {type(response)}")
            if isinstance(response, dict):
                logger.debug(f"Iliad API response keys: {response.keys()}")

            # Extract answer from Iliad API response structure
            # Response format: {"response_id": "...", "completion": {"role": "assistant", "content": "answer text"}}
            answer = ""

            if isinstance(response, dict):
                if "completion" in response:
                    completion = response.get("completion", {})
                    if isinstance(completion, dict):
                        answer = completion.get("content", "")
                        logger.debug(f"Extracted answer length: {len(answer)}")
                    else:
                        logger.warning(f"completion is not a dict: {type(completion)}")
                        answer = str(completion)
                else:
                    # Fallback: try to get content directly
                    logger.warning(f"No 'completion' key in response. Keys: {response.keys()}")
                    answer = response.get("content", "")
            else:
                logger.warning(f"Response is not a dict: {type(response)}")
                answer = str(response)

            if not answer:
                logger.error(f"Empty answer received. Full response: {response}")
                raise ValueError("Empty answer received from Iliad API")

            logger.info(f"Successfully generated answer ({len(answer)} characters)")
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
        # === DEBUG SEPARATOR ===
        logger.info("=" * 80)
        logger.info(f"NEW QUERY: {question}")
        logger.info(f"Settings: two_stage={self.enable_two_stage_rag}, reranking={self.use_reranking}, top_k={self.top_k}")
        logger.info("=" * 80)

        try:
            # Step 1: Retrieve relevant documents (with re-ranking if enabled)
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

            # Add re-ranking information if available
            if "composite_scores" in retrieved:
                result["composite_scores"] = retrieved["composite_scores"]
            if "score_breakdown" in retrieved:
                result["score_breakdown"] = retrieved["score_breakdown"]
            if "processed_query" in retrieved:
                processed = retrieved["processed_query"]
                result["query_analysis"] = {
                    "keywords": processed.keywords,
                    "lemmatized_keywords": processed.lemmatized_keywords,
                    "potential_projects": processed.potential_project_names,
                    "potential_people": processed.potential_person_names,
                }

            logger.info("Successfully completed RAG query")
            return result

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            raise

    def _format_sources(self, metadatas: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format source information for response.

        Sources are kept in order matching the documents sent to the LLM
        (Document 1, Document 2, etc.) without deduplication to ensure
        the UI sources align with LLM references.

        Args:
            metadatas: List of metadata dictionaries.

        Returns:
            List of formatted source dictionaries with document_index.
        """
        sources = []

        for i, meta in enumerate(metadatas, 1):
            sources.append(
                {
                    "document_index": i,
                    "title": meta.get("title", "Unknown"),
                    "url": meta.get("url", ""),
                    "type": meta.get("source_type", "Unknown"),
                }
            )

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
