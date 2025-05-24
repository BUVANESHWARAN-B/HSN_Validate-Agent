import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
import numpy as np
from sklearn.preprocessing import normalize
import os
# import re # Removed as explicit commands are back

# --- Re-using the existing HSN data loading, Trie, and SemanticSearcher classes ---
# (These classes would typically be defined in separate modules and imported in a real ADK setup)

# --- 1. Data Loading ---
def load_hsn_data(file_path, sheet_name='HSN_MSTR'):
    """
    Loads HSN master data from an Excel file.
    Args:
        file_path (str): The path to the HSN_SAC.xlsx file.
        sheet_name (str): The name of the sheet containing HSN master data.
    Returns:
        pd.DataFrame: DataFrame containing HSNCode and Description.
    """
    try:
        # Use pd.read_excel to read directly from the .xlsx file
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        # Ensure HSNCode is treated as string to preserve leading zeros
        df['HSNCode'] = df['HSNCode'].astype(str)
        
        # --- FIX: Ensure 'Description' column is all strings ---
        # Fill any NaN values with an empty string and then convert to string type
        df['Description'] = df['Description'].fillna('').astype(str)
        # --- END FIX ---

        print(f"Successfully loaded {len(df)} HSN records from sheet '{sheet_name}' in '{file_path}'")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the Excel file: {e}")
        return None

# --- 2. Trie (Prefix Tree) Implementation ---
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_hsn = False
        self.description = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, hsn_code: str, description: str):
        node = self.root
        for digit in hsn_code:
            if digit not in node.children:
                node.children[digit] = TrieNode()
            node = node.children[digit]
        node.is_end_of_hsn = True
        node.description = description

    def search(self, hsn_code: str) -> tuple[bool, str | None]:
        node = self.root
        for digit in hsn_code:
            if digit not in node.children:
                return False, None
            node = node.children[digit]
        if node.is_end_of_hsn:
            return True, node.description
        return False, None

    def starts_with(self, prefix: str) -> list[tuple[str, str]]:
        node = self.root
        for digit in prefix:
            if digit not in node.children:
                return []
            node = node.children[digit]

        results = []
        def _collect_all_hsn(current_node, current_hsn_str):
            if current_node.is_end_of_hsn:
                results.append((current_hsn_str, current_node.description))
            for digit, child_node in current_node.children.items():
                _collect_all_hsn(child_node, current_hsn_str + digit)

        _collect_all_hsn(node, prefix)
        return results

# --- 3. Vector Embeddings and FAISS for Semantic Retrieval ---
class SemanticSearcher:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print(f"Loading Sentence Transformer model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.faiss_index = None
        self.hsn_data_map = []

    def _get_embedding(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        embeddings = normalize(embeddings.cpu().numpy(), axis=1, norm='l2')
        return embeddings

    def build_index(self, hsn_df: pd.DataFrame):
        print("Generating embeddings for HSN descriptions...")
        descriptions = hsn_df['Description'].tolist()
        hsn_codes = hsn_df['HSNCode'].tolist()

        batch_size = 32
        all_embeddings = []
        for i in range(0, len(descriptions), batch_size):
            batch_descriptions = descriptions[i:i + batch_size]
            batch_embeddings = self._get_embedding(batch_descriptions)
            all_embeddings.append(batch_embeddings)
            for j in range(len(batch_descriptions)):
                self.hsn_data_map.append({
                    "hsn_code": hsn_codes[i+j],
                    "description": batch_descriptions[j]
                })

        embeddings_array = np.vstack(all_embeddings).astype('float32')
        dimension = embeddings_array.shape[1]

        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings_array)
        print(f"FAISS index built with {self.faiss_index.ntotal} embeddings.")

    def search_semantic(self, query_text: str, k: int = 5) -> list[dict]:
        if self.faiss_index is None:
            print("FAISS index not built. Please call build_index() first.")
            return []

        query_embedding = self._get_embedding([query_text]).astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            l2_distance = distances[0][i]
            similarity_score = 1 - (l2_distance**2 / 2)

            if idx < len(self.hsn_data_map):
                results.append({
                    "hsn_code": self.hsn_data_map[idx]["hsn_code"],
                    "description": self.hsn_data_map[idx]["description"],
                    "similarity_score": similarity_score
                })
        return results

# --- Conceptual Google ADK Agent Framework ---

# Simulate ADK components
class ADKState:
    """Represents the agent's internal state."""
    def __init__(self):
        self.current_hsn_input = None
        self.validation_result = None
        self.semantic_results = None
        self.conversation_history = []

class ADKTool:
    """Base class for ADK tools."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Execute method must be implemented by subclasses.")

class HSNValidationTool(ADKTool):
    """ADK Tool for HSN code exact validation and prefix search using Trie."""
    def __init__(self, trie_instance: Trie):
        super().__init__(
            name="HSN_Validation_Tool",
            description="Validates HSN codes for exact existence and provides prefix-based suggestions."
        )
        self.trie = trie_instance

    def execute(self, hsn_code: str, mode: str = "exact") -> dict:
        """
        Executes HSN validation or prefix search.
        Args:
            hsn_code (str): The HSN code or prefix to check.
            mode (str): "exact" for exact validation, "prefix" for suggestions.
        Returns:
            dict: Results of the validation or suggestions.
        """
        if mode == "exact":
            found, description = self.trie.search(hsn_code)
            if found:
                return {"status": "success", "type": "exact_match", "hsn_code": hsn_code, "description": description}
            else:
                return {"status": "not_found", "type": "exact_match", "hsn_code": hsn_code, "message": "HSN code not found."}
        elif mode == "prefix":
            suggestions = self.trie.starts_with(hsn_code)
            if suggestions:
                return {"status": "success", "type": "prefix_suggestions", "prefix": hsn_code, "suggestions": suggestions}
            else:
                return {"status": "no_suggestions", "type": "prefix_suggestions", "prefix": hsn_code, "message": "No HSN codes found for this prefix."}
        else:
            return {"status": "error", "message": "Invalid mode for HSNValidationTool. Use 'exact' or 'prefix'."}

class HSNSemanticSearchTool(ADKTool):
    """ADK Tool for semantic search of HSN descriptions using FAISS."""
    def __init__(self, semantic_searcher_instance: SemanticSearcher):
        super().__init__(
            name="HSN_Semantic_Search_Tool",
            description="Performs semantic search on HSN descriptions using natural language queries."
        )
        self.semantic_searcher = semantic_searcher_instance

    def execute(self, query_text: str, k: int = 5) -> dict:
        """
        Executes a semantic search.
        Args:
            query_text (str): The natural language query for HSN descriptions.
            k (int): Number of top results to return.
        Returns:
            dict: Results of the semantic search.
        """
        results = self.semantic_searcher.search_semantic(query_text, k)
        if results:
            return {"status": "success", "results": results}
        else:
            return {"status": "no_results", "message": "No semantic results found for the query."}

class ADKAgent:
    """
    Conceptual ADK Agent for HSN Code Validation.
    This simulates the agent's main loop and interaction with tools.
    """
    def __init__(self, hsn_excel_path: str, hsn_sheet_name: str):
        self.state = ADKState()
        self.hsn_df = load_hsn_data(hsn_excel_path, hsn_sheet_name)

        if self.hsn_df is None:
            raise Exception("Failed to load HSN data. Agent cannot initialize.")

        # Initialize core logic components
        self.hsn_trie = Trie()
        for index, row in self.hsn_df.iterrows():
            # Ensure description is passed as string for Trie insertion
            self.hsn_trie.insert(row['HSNCode'], str(row['Description']))

        self.semantic_searcher = SemanticSearcher()
        self.semantic_searcher.build_index(self.hsn_df)

        # Register tools with the agent
        self.tools = {
            "hsn_validation": HSNValidationTool(self.hsn_trie),
            "hsn_semantic_search": HSNSemanticSearchTool(self.semantic_searcher)
        }
        print("ADK HSN Validation Agent initialized.")

    def process_input(self, user_input: str) -> str:
        """
        Simulates the agent processing a user input.
        This version relies on explicit command prefixes for accurate intent detection.
        """
        self.state.conversation_history.append(f"User: {user_input}")
        response = ""

        # Explicit intent recognition based on command prefixes
        if user_input.lower().startswith("validate hsn"):
            parts = user_input.split(" ", 2)
            if len(parts) > 2:
                hsn_code_to_validate = parts[2].strip()
                print(f"Agent: Attempting to validate HSN code: {hsn_code_to_validate}")
                result = self.tools["hsn_validation"].execute(hsn_code_to_validate, mode="exact")
                self.state.validation_result = result
                if result["status"] == "success":
                    response = f"HSN Code '{result['hsn_code']}' is VALID. Description: '{result['description']}'"
                else:
                    response = f"HSN Code '{result['hsn_code']}' is NOT FOUND in master data."
            else:
                response = "Please provide an HSN code to validate (e.g., 'validate hsn 01012100')."
        elif user_input.lower().startswith("find hsn for"):
            query_text = user_input[len("find hsn for"):].strip()
            if query_text:
                print(f"Agent: Performing semantic search for: '{query_text}'")
                result = self.tools["hsn_semantic_search"].execute(query_text, k=3)
                self.state.semantic_results = result
                if result["status"] == "success":
                    response = f"Here are the top semantic matches for '{query_text}':\n"
                    for res in result['results']:
                        response += f"  - HSN: {res['hsn_code']}, Description: '{res['description']}', Similarity: {res['similarity_score']:.4f}\n"
                else:
                    response = f"Could not find semantic matches for '{query_text}'."
            else:
                response = "Please provide a description for semantic search (e.g., 'find hsn for live animals')."
        elif user_input.lower().startswith("suggest hsn for"):
            prefix_to_suggest = user_input[len("suggest hsn for"):].strip()
            if prefix_to_suggest:
                print(f"Agent: Getting HSN suggestions for prefix: '{prefix_to_suggest}'")
                result = self.tools["hsn_validation"].execute(prefix_to_suggest, mode="prefix")
                if result["status"] == "success":
                    response = f"Here are HSN codes starting with '{result['prefix']}':\n"
                    for hsn, desc in result['suggestions'][:5]:
                        response += f"  - {hsn}: {desc}\n"
                    if len(result['suggestions']) > 5:
                        response += f"  ...and {len(result['suggestions']) - 5} more.\n"
                else:
                    response = f"No suggestions found for prefix '{result['prefix']}'."
            else:
                response = "Please provide a prefix for HSN suggestions (e.g., 'suggest hsn for 010')."
        else:
            response = "I can validate HSN codes (e.g., 'validate hsn 01012100'), find HSNs by description (e.g., 'find hsn for parts of aircraft'), or suggest HSNs by prefix (e.g., 'suggest hsn for 010')."

        self.state.conversation_history.append(f"Agent: {response}")
        return response

# --- Main Execution (Simulating ADK Agent Lifecycle) ---
if __name__ == "__main__":
    hsn_excel_path = 'HSN_SAC.xlsx'
    hsn_sheet_name = 'HSN_MSTR'

    try:
        # Initialize the ADK Agent (which loads data and builds tools)
        hsn_agent = ADKAgent(hsn_excel_path, hsn_sheet_name)

        print("\n--- ADK Agent Interaction Simulation ---")
        print("Type 'exit' to quit.")

        while True:
            user_query = input("\nYour query: ")
            if user_query.lower() == 'exit':
                print("Exiting ADK Agent simulation.")
                break
            
            agent_response = hsn_agent.process_input(user_query)
            print(f"Agent Response: {agent_response}")

    except Exception as e:
        print(f"An error occurred during agent initialization or operation: {e}")
