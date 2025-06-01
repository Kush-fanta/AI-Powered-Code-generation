# AI-Powered-Code-generation

Super AI is a smart coding assistant that takes your project idea and automatically creates both a detailed design and working code. It's like having a team of expert programmers working for you.

## How the Code Works

### 1. Setting Up the AI Models

**Primary AI (Groq)**
```python
groq_api_key = "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"
primary_llm = ChatGroq(model='llama-3.3-70b-versatile', groq_api_key=groq_api_key)
```
This sets up the main AI brain using Groq's Llama model. Think of this as hiring your main programmer who will do most of the work.

**Backup AI System (AzureGitHubAILLM Class)**
```python
class AzureGitHubAILLM(BaseLanguageModel):
```
This creates a custom wrapper that connects to Microsoft's Azure AI service through GitHub. It's like having backup programmers ready in case your main programmer gets sick or overloaded.

The class includes several important methods:
- `_generate()`: Sends requests to the AI and gets responses back
- `invoke()`: The main way to ask the AI to do work
- `_call()`: Actually talks to the AI service
- Error handling to catch problems and report them clearly

**Multiple Backup Models**
```python
guard_models = [
    "openai/gpt-4o",
    "meta/Meta-Llama-3.1-405B-Instruct", 
    "microsoft/Phi-4",
    "openai/gpt-4.1",
    "xai/grok-3"
]
```
The app tries to connect to 5 different backup AI models. If the main AI fails, it automatically switches to one of these. It's like having multiple phone numbers to call if the first one is busy.

### 2. Token Management (Word Counting)

**Token Counter**
```python
TOKEN_THRESHOLD = 8000
encoder = tiktoken.get_encoding('cl100k_base')

def count_tokens(text: str) -> int:
    return len(encoder.encode(text))
```
AIs have limits on how much text they can process at once (like reading capacity). This code:
- Sets a safe limit of 8000 "tokens" (roughly words)
- Uses OpenAI's token counter to measure text length
- Prevents sending too much text that would crash the AI

### 3. Smart Error Handling

**Error Detection Functions**
```python
def is_rate_limit_error(error_str: str) -> bool:
def is_token_limit_error(error_str: str) -> bool:
```
These functions read error messages and figure out what went wrong:
- Rate limit errors: "You're asking too fast, slow down"
- Token limit errors: "Your text is too long"
Each type needs different handling.

**Safe AI Calling**
```python
def safe_llm_call(chain, max_retries: int = 3, **kwargs):
```
This is the safety system that makes the app reliable:

1. **Try Main AI**: Attempts to use the primary Groq AI
2. **Handle Rate Limits**: If AI says "slow down", it waits and tries again (up to 3 times)
3. **Try Backup AIs**: If main AI fails completely, it tries each backup AI one by one
4. **Graceful Failure**: If everything fails, shows a nice error message instead of crashing

The waiting uses "exponential backoff" - waits 5 seconds, then 10, then 15, getting longer each time.

### 4. The Three AI Workers

**Wikipedia Research Helper**
```python
wikipedia = WikipediaAPIWrapper()
```
Before designing anything, the AI looks up information on Wikipedia to understand your topic better. Like doing research before writing a report.

**Designer AI**
```python
design_prompt = """
You are an expert system architect and designer...
Create a comprehensive technical design for: {query}
Please provide:
1. System Overview - High-level description and goals
2. Core Features - Key functionality and user experience  
3. Technology Stack - Recommended technologies and frameworks
4. Architecture Design - System components and their interactions
5. Development Roadmap - Phases and milestones
6. Implementation Notes - Important technical considerations
"""
```
This AI acts like a software architect. It creates detailed blueprints including:
- What the app will do overall
- What features users will see
- What programming tools to use
- How different parts will work together
- Step-by-step building plan
- Important things to remember while coding

**Reviewer AI**
```python
reason_prompt = """
You are a technical reviewer. Analyze this design...
Evaluate the design for:
1. Technical Feasibility - Can this be implemented?
2. Architecture Quality - Is the system design sound?
3. Missing Components - What important aspects are overlooked?
4. Risk Assessment - What are the main risks?
5. Improvement Suggestions - How can this be enhanced?
"""
```
This AI acts like a senior programmer reviewing work. It checks:
- Can this actually be built with current technology?
- Is the design well-structured and scalable?
- What important pieces are missing?
- What could go wrong?
- How to make it better?

If the design is good, it responds with "APPROVED". If not, it says "NEEDS_REVISION" and explains problems.

**Coder AI**
```python
code_prompt = """
You are a senior software developer...
Generate production-ready code that includes:
1. Complete file structure with all necessary files
2. Core functionality implementation
3. Error handling and validation
4. Documentation and comments
5. Configuration files if needed
"""
```
This AI writes the actual code. It creates:
- All the files needed for a working app
- The main program logic
- Code to handle errors gracefully
- Comments explaining how everything works
- Setup files and configurations

### 5. Code Organization

**File Parser**
```python
def parse_code_blocks(raw_code: str) -> dict:
    files = {}
    sections = re.split(r'###\s*FILENAME:\s*([^\n]+)', raw_code)
```
The AI generates code with special markers like "### FILENAME: app.py". This function:
- Splits the generated code at these markers
- Creates separate files for each piece
- Cleans up the code (removes extra formatting)
- Returns a dictionary with filename -> code content

If the AI doesn't use proper markers, it puts everything in a default "main.py" file.

### 6. The Smart Workflow

**Iterative Design Process**
```python
def iterative_design(query: str, max_iters: int = 2) -> str:
```
This creates designs through multiple rounds of improvement:

1. **Research**: Get Wikipedia background information
2. **Initial Design**: Designer AI creates first version
3. **Improvement Loop** (repeats based on settings):
   - Reviewer AI analyzes the design
   - Designer AI improves based on feedback
4. **Return Final Design**: After all improvements

**Iterative Code Generation**
```python
def iterative_code(design: str, max_iters: int = 2) -> str:
```
This creates code through multiple rounds of improvement:

1. **Initial Code**: Coder AI writes first version
2. **Improvement Loop** (repeats based on settings):
   - Reviewer AI analyzes the code
   - Coder AI improves based on feedback
3. **Return Final Code**: After all improvements

### 7. User Interface (Streamlit)

**Main Interface**
```python
st.title("Super AI")
user_query = st.text_input("🎯 Describe your task/need:")
```
Creates a web page with:
- Title and description
- Text box for user to describe their project
- Dropdown menus to choose how many improvement rounds
- Generate button to start the process

**Processing and Display**
```python
if st.button("🚀Generate", type="primary"):
    with st.spinner("Creating system design..."):
        final_design = iterative_design(user_query, design_iterations)
    
    st.markdown("## System Design")
    st.markdown(final_design)
```
When user clicks Generate:

1. **Design Phase**: Shows spinning wheel while creating design, then displays it
2. **Code Phase**: Shows spinning wheel while generating code
3. **File Organization**: Parses code into separate files
4. **Smart Display**: 
   - If 5 or fewer files: Creates tabs for each file
   - If more than 5 files: Uses expandable sections
   - Detects file type (.py, .js, .html) for proper syntax highlighting

**Language Detection**
```python
lang_map = {
    'py': 'python', 'js': 'javascript', 'html': 'html',
    'css': 'css', 'json': 'json', 'yml': 'yaml'
}
```
Looks at file extensions to show code in the right colors (Python in one style, JavaScript in another, etc.).
