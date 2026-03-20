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
        Identify relevant main projects using ProjectVectorStore similarity.

        This is the fallback method when LLM entity extraction finds no projects.

        Args:
            query: User's question or query.
            n_results: Number of projects to retrieve. Defaults to project_retrieval_top_k.

        Returns:
            List of relevant main_project names.
        """
        if not self.project_store:
            logger.warning("Project store not available for fallback retrieval")
            return []

        n_results = n_results or self.project_retrieval_top_k

        logger.info(f"Fallback: Identifying top {n_results} projects via similarity")

        try:
            project_results = self.project_store.query_projects(query, n_results=n_results)

            if not project_results:
                logger.warning("No projects found via similarity fallback")
                return []

            # DEBUG: Log detailed project results with similarity scores
            logger.info("--- Fallback Project Similarity Results ---")
            for i, p in enumerate(project_results, 1):
                logger.info(
                    f"  {i}. {p.get('main_project', 'Unknown')}: "
                    f"similarity={p.get('similarity', 0):.4f}, "
                    f"pages={p.get('page_count', 0)}"
                )
            logger.info("-" * 40)

            project_names = [p.get("main_project", "") for p in project_results]
            project_names = [p for p in project_names if p]

            logger.info(f"Fallback identified projects: {project_names}")
            return project_names

        except Exception as e:
            logger.error(f"Error in fallback project identification: {str(e)}")
            return []

    def _determine_filter_logic(self, processed_query: "ProcessedQuery") -> str:
        """
        Determine the filter logic (AND/OR) based on query intent.

        Args:
            processed_query: The processed query with extracted entities and intent.

        Returns:
            "OR" or "AND" based on query characteristics.
        """
        intent = processed_query.query_intent.lower()

        # Listing queries: "What projects did X work on?" → OR (any match)
        if intent in ["listing", "aggregation"]:
            return "OR"

        # Comparison queries: "Compare A to B" → OR (need results from both)
        if intent == "comparison" or processed_query.is_comparative:
            return "OR"

        # How-to with specific project: "How to use X in project Y?" → AND
        if intent == "how-to" and processed_query.potential_project_names:
            return "AND"

        # Informational with single entity type → OR
        # Informational with multiple entity types → context-dependent
        has_projects = len(processed_query.potential_project_names) > 0
        has_people = len(processed_query.potential_person_names) > 0

        if has_projects and has_people:
            # "What did John work on in ATLAS?" → AND (both conditions)
            return "AND"

        # Default to OR for broader results
        return "OR"

    def _build_entity_filters(
        self,
        processed_query: "ProcessedQuery",
        matched_projects: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build filter list from extracted entities.

        Args:
            processed_query: The processed query with extracted entities.
            matched_projects: Optional list of matched main_project values
                (from similarity matching). If provided, uses these instead
                of the raw extracted project names.

        Returns:
            List of filter dictionaries for query_with_multi_filter.
        """
        filters = []

        # Add project filter if projects were extracted or matched
        project_values = matched_projects or processed_query.potential_project_names
        if project_values:
            filters.append({
                "field": "main_project",
                "values": project_values,
            })
            logger.info(f"Entity filter - projects: {project_values}")

        # Add author filter if people were extracted
        if processed_query.potential_person_names:
            filters.append({
                "field": "author",
                "values": processed_query.potential_person_names,
            })
            logger.info(f"Entity filter - authors: {processed_query.potential_person_names}")

        return filters

    def retrieve_by_title_and_children(
        self,
        query: str,
        project_names: List[str],
        n_results: int,
    ) -> Dict[str, Any]:
        """
        Retrieve documents by finding pages with similar titles and their children.

        Used when main_project matching fails. Finds pages with titles similar to
        the extracted project names, then retrieves content from those pages
        and all their descendants.

        Args:
            query: User's question or query.
            project_names: Project names extracted from the query.
            n_results: Number of documents to retrieve.

        Returns:
            Dictionary containing retrieved documents, metadatas, and distances.
        """
        logger.info(f"Attempting title+children retrieval for: {project_names}")

        # Find pages with titles similar to project names
        # Use max_depth=8 to cover most content (pages go up to depth 10)
        title_matches = self.vector_store.find_pages_by_title_similarity(
            search_terms=project_names,
            similarity_threshold=0.6,
            max_depth=8,
        )

        if not title_matches:
            logger.warning("No title matches found")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
                "filter_method": "title_children_empty",
            }

        # Collect all page IDs (matched pages + their descendants)
        all_page_ids = set()
        for match in title_matches:
            page_id = match['page_id']
            # Get this page and all its descendants
            descendants = self.vector_store.get_descendant_page_ids(page_id)
            all_page_ids.update(descendants)

        matched_titles = [m['title'] for m in title_matches]
        logger.info(
            f"Title matching found {len(title_matches)} pages: {matched_titles}, "
            f"expanding to {len(all_page_ids)} total pages (including children)"
        )

        # Query filtered by these page IDs
        results = self.vector_store.query_with_page_ids(
            query_text=query,
            page_ids=list(all_page_ids),
            n_results=n_results,
        )

        results['filter_method'] = 'title_children'
        results['matched_titles'] = [m['title'] for m in title_matches]
        results['title_matches'] = title_matches

        return results

    def retrieve_with_entity_filter(
        self,
        query: str,
        processed_query: "ProcessedQuery",
        n_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve documents filtered by LLM-extracted entities.

        Implements a cascading fallback strategy:
        1. First, try main_project similarity matching (strict, 0.8 threshold)
        2. If no results, try title similarity + children pages
        3. If still no results, fall back to ProjectVectorStore similarity

        Args:
            query: User's question or query.
            processed_query: Processed query with extracted entities.
            n_results: Number of documents to retrieve.

        Returns:
            Dictionary containing retrieved documents, metadatas, and distances.
        """
        n_results = n_results or self.top_k
        filter_logic = self._determine_filter_logic(processed_query)

        logger.info("=" * 60)
        logger.info("Entity-based retrieval with cascading fallback")
        logger.info(f"  Projects: {processed_query.potential_project_names}")
        logger.info(f"  People: {processed_query.potential_person_names}")
        logger.info("=" * 60)

        try:
            # === STAGE 1: main_project similarity matching ===
            if processed_query.potential_project_names:
                logger.info("Stage 1: Attempting main_project similarity matching")

                # Find main_projects that match the extracted names using similarity
                matched_projects = self.vector_store.find_main_projects_by_similarity(
                    project_names=processed_query.potential_project_names,
                    similarity_threshold=0.8,  # Strict matching
                )

                if matched_projects:
                    logger.info(f"Stage 1: Found matching main_projects: {matched_projects}")

                    # Build filters with the matched projects
                    filters = self._build_entity_filters(processed_query, matched_projects)

                    results = self.vector_store.query_with_multi_filter(
                        query_text=query,
                        n_results=n_results,
                        filters=filters,
                        filter_logic=filter_logic,
                    )

                    if len(results['documents']) >= n_results // 2:
                        results['filter_method'] = 'main_project_similarity'
                        results['matched_projects'] = matched_projects
                        results['extracted_projects'] = processed_query.potential_project_names
                        results['extracted_people'] = processed_query.potential_person_names
                        results['filter_logic'] = filter_logic
                        logger.info(
                            f"Stage 1 SUCCESS: Retrieved {len(results['documents'])} documents"
                        )
                        return results
                    else:
                        logger.warning(
                            f"Stage 1: Only {len(results['documents'])} docs, trying Stage 2"
                        )
                else:
                    logger.warning("Stage 1: No main_project matches found")

                # === STAGE 2: Title similarity + children ===
                logger.info("Stage 2: Attempting title similarity + children")

                results = self.retrieve_by_title_and_children(
                    query=query,
                    project_names=processed_query.potential_project_names,
                    n_results=n_results,
                )

                if len(results['documents']) >= n_results // 2:
                    results['extracted_projects'] = processed_query.potential_project_names
                    results['extracted_people'] = processed_query.potential_person_names
                    logger.info(
                        f"Stage 2 SUCCESS: Retrieved {len(results['documents'])} documents"
                    )
                    return results
                else:
                    logger.warning(
                        f"Stage 2: Only {len(results['documents'])} docs, trying Stage 3"
                    )

            # === STAGE 3: People-only filter or classic fallback ===
            # If we only have people (no projects), or previous stages failed
            if processed_query.potential_person_names and not processed_query.potential_project_names:
                logger.info("Stage 3a: People-only filter")
                filters = self._build_entity_filters(processed_query)

                results = self.vector_store.query_with_multi_filter(
                    query_text=query,
                    n_results=n_results,
                    filters=filters,
                    filter_logic=filter_logic,
                )

                results['filter_method'] = 'author_filter'
                results['extracted_people'] = processed_query.potential_person_names
                results['filter_logic'] = filter_logic
                return results

            # === STAGE 3b: ProjectVectorStore similarity fallback ===
            logger.info("Stage 3b: Falling back to ProjectVectorStore similarity")

            fallback_projects = self.identify_relevant_projects(query)
            if fallback_projects:
                results = self.retrieve_filtered_by_project(
                    query, fallback_projects, n_results=n_results
                )
                results['filter_method'] = 'similarity_fallback'
                results['identified_projects'] = fallback_projects
            else:
                # Ultimate fallback: unfiltered query
                logger.warning("Stage 3b: No projects identified, using unfiltered query")
                results = self.vector_store.query(
                    query_text=query, n_results=n_results
                )
                results['filter_method'] = 'unfiltered'

            results['extracted_projects'] = processed_query.potential_project_names
            results['extracted_people'] = processed_query.potential_person_names
            return results

        except Exception as e:
            logger.error(f"Error in entity-based retrieval: {str(e)}")
            raise

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

        Uses entity-based filtering with fallback:
        1. First, extract entities (projects, people) from query via LLM
        2. If entities found: filter vector store by those entities
        3. If no entities found: fallback to ProjectVectorStore similarity
        4. Apply re-ranking if enabled

        Args:
            query: User's question or query.
            n_results: Number of documents to retrieve. If None, uses self.top_k.

        Returns:
            Dictionary containing retrieved documents, metadatas, and distances.
        """
        n_results = n_results or self.top_k

        # Process query to extract keywords and entities
        processed_query = self.query_processor.process_query(query)
        logger.info(
            f"Processed query - Keywords: {processed_query.keywords}, "
            f"Projects: {processed_query.potential_project_names}, "
            f"People: {processed_query.potential_person_names}, "
            f"Intent: {processed_query.query_intent}, "
            f"Comparative: {processed_query.is_comparative}"
        )

        # Retrieve more documents initially if re-ranking is enabled
        # This allows re-ranking to potentially surface better matches
        initial_retrieve = n_results * 3 if self.use_reranking else n_results

        try:
            # Check if LLM extracted any filterable entities
            has_extracted_entities = (
                len(processed_query.potential_project_names) > 0 or
                len(processed_query.potential_person_names) > 0
            )

            if self.enable_two_stage_rag:
                if has_extracted_entities:
                    # Primary path: Use entity-based retrieval with cascading fallback
                    # (main_project similarity → title+children → ProjectVectorStore)
                    logger.info("Using entity-based filtering with cascading fallback")
                    results = self.retrieve_with_entity_filter(
                        query, processed_query, n_results=initial_retrieve
                    )
                else:
                    # No entities extracted, go directly to ProjectVectorStore similarity
                    logger.info("No entities extracted, using ProjectVectorStore similarity")
                    relevant_projects = self.identify_relevant_projects(query)

                    if relevant_projects:
                        results = self.retrieve_filtered_by_project(
                            query, relevant_projects, n_results=initial_retrieve
                        )
                        results['filter_method'] = 'similarity_fallback'
                        results['identified_projects'] = relevant_projects
                    else:
                        # Ultimate fallback: unfiltered query
                        logger.warning("No projects identified, using unfiltered query")
                        results = self.vector_store.query(
                            query_text=query, n_results=initial_retrieve
                        )
                        results['filter_method'] = 'unfiltered'
                        results['identified_projects'] = []
            else:
                # Two-stage RAG disabled, use simple query
                logger.info(
                    f"Retrieving {initial_retrieve} documents for query "
                    f"(reranking: {self.use_reranking})"
                )
                results = self.vector_store.query(
                    query_text=query, n_results=initial_retrieve
                )
                results['filter_method'] = 'disabled'

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
                    "type": meta.get("source_type") or meta.get("space_name", "Confluence Page"),
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
