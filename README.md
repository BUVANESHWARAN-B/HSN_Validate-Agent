# HSN Validation Agent

This project implements a sophisticated HSN (Harmonized System of Nomenclature) Validation Agent in Python. It provides functionalities for exact HSN code validation, prefix-based suggestions, and semantic search of HSN descriptions using natural language queries. The agent is designed to simulate the core components of an Agent Development Kit (ADK) framework, demonstrating how different tools can be integrated to provide intelligent responses.

## Table of Contents

1.  [Features](#features)
2.  [How it Works (Flow Explanation)](#how-it-works-flow-explanation)
3.  [Setup](#setup)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
4.  [Usage](#usage)
5.  [Code Structure](#code-structure)
6.  [Future Enhancements](#future-enhancements)

---

## Features

* **Exact HSN Code Validation:** Quickly checks if a given HSN code exists in the master data and retrieves its full description.
* **Prefix-based HSN Suggestions:** Provides a list of HSN codes and their descriptions that start with a given numerical prefix, useful for auto-completion or narrowing down searches.
* **Semantic Search:** Allows users to find relevant HSN codes by describing goods or services in natural language. It leverages advanced NLP models to understand the query's meaning and return the most similar HSN descriptions.
* **Modular Design:** Built with a conceptual ADK-like structure, separating data loading, core logic (Trie, Semantic Searcher), and agent interaction into distinct components.

---

## How it Works (Flow Explanation)

The HSN Validation Agent operates in a simulated ADK environment, processing user queries through a defined flow:

1.  **Data Loading (`load_hsn_data`):**
    * Upon initialization, the agent first loads the HSN master data from an Excel file (`HSN_SAC.xlsx`).
    * It ensures that HSN codes are treated as strings (to preserve leading zeros) and descriptions are robustly handled.

2.  **Core Data Structures & Indexing:**
    * **Trie (`Trie` class):** All HSN codes from the loaded data are inserted into a Trie (prefix tree). This data structure enables extremely fast and efficient exact lookups and prefix-based searches.
    * **Semantic Searcher (`SemanticSearcher` class):**
        * A pre-trained Sentence Transformer model (`sentence-transformers/all-MiniLM-L6-v2`) is loaded to generate numerical vector embeddings for all HSN descriptions.
        * These embeddings are then indexed using FAISS (`faiss.IndexFlatL2`), a library for efficient similarity search. This index allows the agent to quickly find descriptions semantically similar to a user's natural language query.

3.  **Tool Registration:**
    * The agent registers two specialized "tools":
        * `HSNValidationTool`: Handles exact HSN validation and prefix suggestions by interacting with the Trie.
        * `HSNSemanticSearchTool`: Manages natural language semantic searches by querying the FAISS index.

4.  **User Interaction (`ADKAgent.process_input`):**
    * The agent listens for user input.
    * It uses **explicit command prefixes** to understand the user's intent:
        * `validate hsn [HSN_CODE]`
        * `find hsn for [DESCRIPTION_QUERY]`
        * `suggest hsn for [PREFIX]`
    * Based on the detected command, the agent delegates the task to the appropriate internal tool (`HSNValidationTool` or `HSNSemanticSearchTool`).
    * The tool executes its specific logic (e.g., searching the Trie or FAISS index) and returns a structured result.
    * The agent then formats this result into a human-readable response for the user.

This explicit command-based approach ensures reliable and predictable behavior, as the agent doesn't rely on a large language model (LLM) for interpreting the specific tool to use, but rather on the user's direct instruction.

---

## Setup

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository (or download the script):**
    ```bash
    git clone [https://github.com/BUVANESHWARAN-B/HSN_Validate-Agent-.git](https://github.com/BUVANESHWARAN-B/HSN_Validate-Agent-.git)
    cd HSN_Validate-Agent-
    ```

2.  **Install required Python packages:**
    ```bash
    pip install requirements.txt
    ```

## Usage

To run the HSN Validation Agent, navigate to the project directory in your terminal and execute the script:

```bash
python hsnagent.py
```

## Code Structure

* `load_hsn_data(file_path, sheet_name)`: Function to **load the HSN data** from an Excel file.
* `TrieNode`, `Trie`: Classes implementing a **Trie data structure** for efficient prefix and exact HSN code searches.
* `SemanticSearcher`: Class responsible for **loading the Sentence Transformer model**, generating embeddings, building a FAISS index, and performing **semantic similarity searches**.
* `ADKState`: A simple class to simulate the **agent's internal state**.
* `ADKTool`: Base class for **agent tools**.
* `HSNValidationTool`: An ADK tool that wraps the `Trie` functionalities for **validation and suggestions**.
* `HSNSemanticSearchTool`: An ADK tool that wraps the `SemanticSearcher` functionalities for **natural language queries**.
* `ADKAgent`: The main agent class, which initializes the data, builds the tools, and contains the `process_input` method to handle user queries and delegate to the appropriate tools.

---

## Future Enhancements

* **Integration with a real ADK/LLM Framework:** Replace the conceptual `ADKAgent` and `ADKTool` classes with actual `google.adk` components and integrate a large language model (LLM) to enable more natural language understanding and dynamic tool selection.
* **Error Handling and User Feedback:** Improve robustness with more detailed error messages and guidance for the user.
* **Batch Processing:** Allow validation or search for multiple HSN codes in a single query.
* **Web Interface:** Develop a simple web application (e.g., using Flask or FastAPI) to provide a more user-friendly interface.
* **Dynamic Model Loading:** Allow specifying the Sentence Transformer model name as a command-line argument or configuration.
* **Caching:** Implement caching for embeddings or search results to improve performance on repeated queries.
