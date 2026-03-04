from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class FullContext:
    query: str
    chunks: List[Dict[str, Any]]
    triplets: List[Dict[str, str]]
    entities: List[Dict]

class PromptAssembler:
    def __init__(self):
        self.config: Optional[Dict] = None

    def assemble_final_prompt(self, context: FullContext) -> str:
        sections = []
        sections.append(self._get_system_instruction())
        if context.entities:
            sections.append(self._format_entity_labels(context.entities))
        if context.chunks:
            sections.append(self._format_document_chunks(context.chunks))
        if context.triplets:
            sections.append(self._format_triplets(context.triplets))
        sections.append(self._format_query(context.query))
        sections.append(self._get_response_format())

        return "\n\n".join(sections)

    def _get_system_instruction(self) -> str:
        return """You are an advanced AI assistant with access to multiple knowledge sources. Your task is to provide accurate, comprehensive answers by synthesizing information from:
                1. Retrieved document passages (primary source)
                2. Knowledge graph triplets (structured relationships)
                3. Domain-specific entity labels (contextual categorization)
                
                Follow these guidelines:
                - Prioritize information from document chunks for factual accuracy
                - Use triplets to understand relationships between entities
                - Leverage entity labels for domain-specific context
                - Cite sources when possible (document chunk IDs)
                - If information conflicts, acknowledge the discrepancy
                - If information is insufficient, clearly state what's missing
                - Maintain scientific precision in responses"""

    def _format_document_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        formatted = "=== RETRIEVED DOCUMENT PASSAGES ===\n"

        for i, chunk in enumerate(chunks, 1):
            relevance = chunk.get('score', 'N/A')
            if isinstance(relevance, float):
                relevance = f"{relevance:.2f}"
            formatted += f"\n[Passage {i}] (Relevance: {relevance})\n"
            formatted += f"{chunk.get('text', '')}\n"
            formatted += "-" * 50

        return formatted

    def _format_triplets(self, triplets: List[Dict[str, str]]) -> str:
        if not triplets:
            return ""
        formatted = "=== KNOWLEDGE GRAPH RELATIONSHIPS ===\n"
        formatted += "Structured facts extracted from documents:\n"

        documents_extracted_from_triplet = set()
        triplet_context = "Triplet extracted from documents:\n"
        for i, triplet in enumerate(triplets, 1):
            triplet_context += f"{i}. {triplet['subject']} --> [{triplet['predicate']}]--> {triplet['object']} \n"
            documents_extracted_from_triplet.add(triplet['document'])
        formatted = triplet_context + "\n".join(documents_extracted_from_triplet)

        return formatted

    def _format_entity_labels(self, entities: List[Dict]) -> str:
        formatted = "=== DOMAIN-SPECIFIC ENTITY CLASSIFICATIONS ===\n"
        formatted += "Entity types and categories for context:\n"

        if entities is not None:
            for i, entity in enumerate(entities, 1):
                formatted += f"{i}. Entity: {entity['entity']} is label: {entity['label']} \n"

        formatted += "\nUse these classifications to understand the domain context of entities mentioned."
        return formatted

    def _format_query(self, query: str) -> str:
        return f"""=== USER QUERY ===
                {query}
                Based on ALL the information provided above (document passages, knowledge graph relationships, and entity classifications), 
                please provide a comprehensive answer. If the answer requires combining information from multiple sources, 
                explicitly show how you synthesized it.
                """
    def _get_response_format(self) -> str:
        return """=== RESPONSE FORMAT ===
                Please structure your response as follows:
                1. **Direct Answer**: Brief, direct response to the query
                2. **Supporting Evidence**: Key facts from sources
                3. **Synthesis**: How information from different sources connects
                4. **Limitations**: Any gaps or uncertainties in the available information
                
                Begin your response now:"""

    def set_config(self, config: Dict):
        self.config = config

class AdaptivePromptAssembler(PromptAssembler):
    def __init__(self):
        super().__init__()

        self.query_templates = {
            "factual": self._factual_template,
            "explanatory": self._explanatory_template,
            "comparative": self._comparative_template,
            "procedural": self._procedural_template
        }

    def assemble_final_prompt(self, context: FullContext) -> str:
        query_type = self._detect_query_type(context.query)
        specific_instructions = self.query_templates[query_type]
        base_prompt = super().assemble_final_prompt(context)
        return base_prompt + "\n\n" + specific_instructions

    def _detect_query_type(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ["what is", "who is", "when did", "where is"]):
            return "factual"
        elif any(word in query_lower for word in ["why", "how does", "explain", "describe"]):
            return "explanatory"
        elif any(word in query_lower for word in ["compare", "versus", "vs", "difference"]):
            return "comparative"
        elif any(word in query_lower for word in ["how to", "steps", "process", "method"]):
            return "procedural"
        else:
            return "factual"

    def _factual_template(self) -> str:
        return """Additional Instructions for Factual Query:
                - Extract specific facts from document chunks first
                - Use triplets to verify relationships between entities
                - Provide exact quotes when relevant
                - Include temporal information (dates, periods) if available
                - Cite specific passage numbers for key facts"""

    def _explanatory_template(self) -> str:
        return """Additional Instructions for Explanatory Query:
                - Synthesize information from multiple sources
                - Use entity labels to understand domain context
                - Explain cause-effect relationships found in triplets
                - Connect concepts across different document chunks
                - Identify underlying principles or mechanisms"""

    def _comparative_template(self) -> str:
        return """Additional Instructions for Comparative Query:
                - Create structured comparison using available information
                - Use triplets to highlight relationship differences
                - Note when information for comparison is incomplete
                - Identify shared attributes from entity labels
                - Present comparisons in a clear format (tables if helpful)"""

    def _procedural_template(self) -> str:
        return """Additional Instructions for Procedural Query:
                - Sequence steps in logical order
                - Note prerequisites from entity classifications
                - Highlight critical points from document chunks
                - Use triplets to identify tools or materials needed
                - Include warnings or important considerations"""

class FinalAssembler:
    def __init__(self):
        self.assembler = AdaptivePromptAssembler()
        self.chunk: Optional[List[Dict]] = []
        self.triplets: Optional[List[Dict]] = []
        self.entities: Optional[List[Dict]] = []
        self.final_prompt: Optional[str] = ""
        self.query: Optional[str] = ""

    def answer_query(self) -> None:
        context = FullContext(
            query=self.query,
            chunks=self.chunk,
            triplets=self.triplets,
            entities=self.entities,
        )
        self.final_prompt = self.assembler.assemble_final_prompt(context)

    def get_final_prompt(self) -> str:
        return self.final_prompt

    def set_chunk(self, chunk: List) -> None:
        self.chunk = chunk

    def set_triplet(self, triplet: List) -> None:
        self.triplets = triplet

    def set_entities(self, entities: List) -> None:
        self.entities = entities

    def set_query(self, query: str) -> None:
        self.query = query
