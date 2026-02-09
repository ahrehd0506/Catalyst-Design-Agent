from .schemas import ReviewInputs, DesignInputs
from pydantic_ai import RunContext
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def modification_report_base(file_path: str) -> FAISS:
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    
    return vector_store

def modification_report_for_design_agent(ctx: RunContext[DesignInputs], query: str) -> str:
    """
    Search for results of modification from modification history.
    Use this to predict how modification will affect the binding energies of catalyst.
    
    Args:
        query: Keywords or questions about modifications.
    """

    print(f"[Designer] RAG Searching for: '{query}'")
    
    vector_store = ctx.deps.vector_store
    results = vector_store.similarity_search(query, k=3)
    
    if not results:
        return "No information found in the knowledge base."
    
    context_text = "\n\n".join([f"- {doc.page_content}" for doc in results])
    
    return context_text


def _consult_knowledge_base_impl(query: str) -> str:
    knowledge_db = {
        "single atom catalyst": "A catalyst featuring isolated metal atoms anchored on a support matrix (e.g., Fe-N-C).",
        "support metal": "The central active metal atom (e.g., Fe, Co, Pt) that directly binds with reactants and acts as the primary catalytic center.",
        "coordination atoms": "The atoms (typically N, C, O, or S) in the first coordination shell that are directly bonded to the support metal, defining its electronic state and geometry.",
        "axial ligand": "An atom or group (e.g., O, OH) adsorbed perpendicular to the metal plane (axial position) that modulates the metal's reactivity via electronic effects.",
        "functional group": "Chemical groups (e.g., COOH, OH) attached to the surrounding carbon support lattice (2nd shell or periphery) that tune the local environment and electronic density.",
    }
    
    results = []
    for key, value in knowledge_db.items():
        if key in query.lower() or query.lower() in key:
            results.append(f"- {value}")
            
    if not results:
        return "No specific information found in the local knowledge base. Rely on general chemical principles."
    
    return "\n".join(results)


def knowledge_for_review_agent(ctx: RunContext[ReviewInputs],  query: str) -> str:
    """
    Search for scientific principles, chemical properties, or synthesis rules from the knowledge base.
    Use this to verify the Designer's hypothesis.
    
    Args:
        query: Keywords or questions like "stability of Fe-N4", "electronegativity of Co vs Fe", "Sabatier principle for ORR".
    """
    print(f"[Master] Searching Knowledge Base for: '{query}'")
    return _consult_knowledge_base_impl(query)

def knowledge_for_design_agent(ctx: RunContext[DesignInputs], query: str) -> str:
    """
    Search for scientific principles, chemical properties, or rules from the knowledge base.
    Use this to make your reasoning.
    
    Args:
        query: Keywords or questions like "stability of Fe-N4", "electronegativity of Co vs Fe", "Sabatier principle for ORR".
    """
    print(f"[Designer] Searching Knowledge Base for: '{query}'")
    return _consult_knowledge_base_impl(query)

def knowledge_for_reflect_agent(ctx: RunContext[None], query: str) -> str:
    """
    Search for scientific principles, chemical properties, or rules from the knowledge base.
    Use this to make your reflection.
    
    Args:
        query: Keywords or questions like "stability of Fe-N4", "electronegativity of Co vs Fe", "Sabatier principle for ORR".
    """
    print(f"[Designer] Searching Knowledge Base for: '{query}'")
    return _consult_knowledge_base_impl(query)