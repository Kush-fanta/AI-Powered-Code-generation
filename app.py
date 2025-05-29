import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
import re
import tiktoken
from groq import RateLimitError
import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import BasePromptTemplate
from typing import Any, List, Optional, Mapping, Dict, AsyncIterator
import asyncio
from pydantic import Field
import time

# 1. LLM definitions & token utilities
groq_api_key = "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"
primary_llm = ChatGroq(model='llama-3.3-70b-versatile', groq_api_key=groq_api_key)

# Azure GPT-4.1 as guard LLM - properly implemented as LangChain Runnable
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
github_token = os.getenv("GITHUB_TOKEN")
if not github_token:
    st.error("GITHUB_TOKEN environment variable not set")
    st.stop()

class AzureGitHubAILLM(BaseLanguageModel):
    """LangChain compatible wrapper for Azure GitHub AI"""
    endpoint: str
    model_name: str
    credential: str
    client: Any = Field(default=None, exclude=True)
    
    def __init__(self, endpoint: str, model_name: str, credential: str, **kwargs):
        super().__init__(
            endpoint=endpoint,
            model_name=model_name,
            credential=credential,
            **kwargs
        )
        object.__setattr__(self, 'client', ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(credential),
        ))
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            response = self.client.complete(
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                top_p=0.9,
                model=self.model_name
            )
            text = response.choices[0].message.content
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)
    
    async def agenerate_prompt(
        self,
        prompts: List[BasePromptTemplate],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_strings = [str(prompt) for prompt in prompts]
        return self._generate(prompt_strings, stop, **kwargs)
    
    def generate_prompt(
        self,
        prompts: List[BasePromptTemplate],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_strings = [str(prompt) for prompt in prompts]
        return self._generate(prompt_strings, stop, **kwargs)
    
    def predict(self, text: str, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        return self._call(text, stop, **kwargs)
    
    async def apredict(self, text: str, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        return self._call(text, stop, **kwargs)
    
    def predict_messages(
        self, messages: List[BaseMessage], *, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> BaseMessage:
        prompt = "\n".join([msg.content for msg in messages])
        result = self._call(prompt, stop, **kwargs)
        return AIMessage(content=result)
    
    async def apredict_messages(
        self, messages: List[BaseMessage], *, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> BaseMessage:
        return self.predict_messages(messages, stop=stop, **kwargs)
    
    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> str:
        if isinstance(input, str):
            return self._call(input, **kwargs)
        elif isinstance(input, list) and all(isinstance(msg, BaseMessage) for msg in input):
            result = self.predict_messages(input, **kwargs)
            return result.content
        else:
            return self._call(str(input), **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "azure_github_ai"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        try:
            response = self.client.complete(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                top_p=0.9,
                model=self.model_name
            )
            return response.choices[0].message.content
        except Exception as e:
            # Re-raise with model info for better debugging
            raise Exception(f"Error with model {self.model_name}: {str(e)}")

# Initialize guard LLMs with better error handling
guard_models = [
    "openai/gpt-4o",
    "meta/Meta-Llama-3.1-405B-Instruct",
    "microsoft/Phi-4",
    "openai/gpt-4.1",
    "xai/grok-3"
]

guard_llms = []
for model_name in guard_models:
    try:
        # Test initialization with a simple call
        test_llm = AzureGitHubAILLM(
            endpoint=endpoint,
            model_name=model_name,
            credential=github_token
        )
        # Quick test to see if model works
        test_result = test_llm._call("Hello")
        if test_result:
            guard_llms.append(test_llm)
    except Exception as e:
        continue

if not guard_llms:
    pass  # Silently continue with primary LLM only

# Token threshold for fallback
TOKEN_THRESHOLD = 8000  # Reduced threshold to be more conservative
encoder = tiktoken.get_encoding('cl100k_base')

def count_tokens(text: str) -> int:
    """Count tokens in the given text using tiktoken encoder"""
    return len(encoder.encode(text))

def is_rate_limit_error(error_str: str) -> bool:
    """Check if error is actually a rate limit or quota error"""
    rate_limit_indicators = [
        "rate_limit",
        "rate limit",
        "429",
        "quota exceeded",
        "quota_exceeded",
        "too many requests",
        "requests per minute",
        "rpm exceeded"
    ]
    return any(indicator in error_str.lower() for indicator in rate_limit_indicators)

def is_token_limit_error(error_str: str) -> bool:
    """Check if error is related to token/context limits"""
    token_limit_indicators = [
        "context length",
        "token limit",
        "max tokens",
        "input too long",
        "request too large",
        "maximum context"
    ]
    return any(indicator in error_str.lower() for indicator in token_limit_indicators)

def safe_llm_call(chain, max_retries: int = 3, **kwargs):
    """Safely execute LLM chain with proper error handling and fallback"""
    
    # Try primary LLM first
    for attempt in range(max_retries):
        try:
            result = chain.run(**kwargs)
            return result
            
        except Exception as e:
            error_str = str(e)
            
            # If it's a rate limit error, wait and retry
            if is_rate_limit_error(error_str):
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    time.sleep(wait_time)
                    continue
            
            # If it's a token limit error or other error, try guard LLMs
            break
    
    # Try guard LLMs if primary fails
    if guard_llms:
        for i, guard_llm in enumerate(guard_llms):
            try:
                # Create new chain with guard LLM
                guard_chain = LLMChain(llm=guard_llm, prompt=chain.prompt)
                result = guard_chain.run(**kwargs)
                return result
                
            except Exception as e:
                continue
    
    # If all LLMs fail, provide a generic error message
    st.error("❌ Unable to process your request at the moment. Please try again later.")
    raise Exception("Service temporarily unavailable")

# Build chains
wikipedia = WikipediaAPIWrapper()

# Design Chain
design_prompt = """
You are an expert system architect and designer. Using this background information:
{background}

Create a comprehensive technical design for:
{query}

Please provide:
1. **System Overview** - High-level description and goals
2. **Core Features** - Key functionality and user experience
3. **Technology Stack** - Recommended technologies and frameworks  
4. **Architecture Design** - System components and their interactions
5. **Development Roadmap** - Phases and milestones
6. **Implementation Notes** - Important technical considerations

Format your response in clear markdown with proper headings.
"""

def create_design_chain(llm):
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["background","query"], template=design_prompt)
    )

# Reasoning Chain
reason_prompt = """You are a technical reviewer. Analyze this design:

{design}

Evaluate the design for:
1. **Technical Feasibility** - Can this be implemented with current technology?
2. **Architecture Quality** - Is the system design sound and scalable?
3. **Missing Components** - What important aspects might be overlooked?
4. **Risk Assessment** - What are the main technical and business risks?
5. **Improvement Suggestions** - How can this design be enhanced?

If the design is generally solid and implementable, start your response with 'APPROVED'.
Otherwise, start with 'NEEDS_REVISION' and explain the critical issues.
"""

def create_reason_chain(llm):
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["design"], template=reason_prompt)
    )

# Code Generation Chain
code_prompt = """You are a senior software developer. Based on these requirements:

{requirements}

Generate production-ready code that includes:

1. **Complete file structure** with all necessary files
2. **Core functionality** implementation
3. **Error handling** and validation
4. **Documentation** and comments
5. **Configuration** files if needed

**IMPORTANT**: Separate each file clearly using the format:
### FILENAME: filename.ext
[file content here]

### FILENAME: another_file.ext  
[file content here]

Make sure to include all necessary files for a working application.
"""

def create_code_chain(llm):
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=["requirements"], template=code_prompt)
    )

# Code parsing function
def parse_code_blocks(raw_code: str) -> dict:
    """Parse code blocks marked with '### FILENAME:' into separate files."""
    files = {}
    sections = re.split(r'###\s*FILENAME:\s*([^\n]+)', raw_code)
    
    if len(sections) > 1:
        it = iter(sections[1:])  # Skip first empty section
        for filename, content in zip(it, it):
            # Clean up filename and content
            filename = filename.strip()
            # Remove code block markers if present
            content = re.sub(r'^```[a-zA-Z]*\n|```$', '', content.strip(), flags=re.MULTILINE)
            files[filename] = content.strip()
    
    # Fallback if no proper file structure found
    if not files and raw_code.strip():
        files['main.py'] = raw_code.strip()
    
    return files

# Orchestrator functions
def iterative_design(query: str, max_iters: int = 2) -> str:
    """Generate design with iterative refinement"""
    try:
        # Get background information
        background = wikipedia.run(query)
        
        # Create design chain with primary LLM
        chain_design = create_design_chain(primary_llm)
        design = safe_llm_call(chain_design, background=background, query=query)
        
        # Iterative refinement
        for iteration in range(max_iters):
            # Get feedback
            chain_reason = create_reason_chain(primary_llm)
            feedback = safe_llm_call(chain_reason, design=design)
            
            # Refine design based on feedback
            refined_query = f"{query}\n\n**Previous Design Feedback:**\n{feedback}\n\nPlease improve the design based on this feedback."
            design = safe_llm_call(chain_design, background=background, query=refined_query)
        
        return design
        
    except Exception as e:
        return f"# Unable to Generate Design\n\nPlease try again with a different request."

def iterative_code(design: str, max_iters: int = 2) -> str:
    """Generate code with iterative refinement"""
    try:
        chain_code = create_code_chain(primary_llm)
        code = safe_llm_call(chain_code, requirements=design)
        
        # Iterative refinement
        for iteration in range(max_iters):
            # Get feedback on current code
            chain_reason = create_reason_chain(primary_llm)
            feedback = safe_llm_call(chain_reason, design=f"Generated Code:\n\n{code}")
            
            # Refine code based on feedback
            refined_requirements = f"{design}\n\n**Code Review Feedback:**\n{feedback}\n\nPlease improve the code based on this feedback."
            code = safe_llm_call(chain_code, requirements=refined_requirements)
        
        return code
        
    except Exception as e:
        return f"# Unable to Generate Code\n\nPlease try again."

# Streamlit UI
st.title("Super AI")
st.markdown("""
Generate complete application designs and production-ready code with automatic LLM fallback for reliability.
""")

# Sidebar info
st.sidebar.markdown("### 🔧 Settings")
st.sidebar.markdown("**AI-Powered Code Generation**")
st.sidebar.markdown("Intelligent system with automatic optimization")

user_query = st.text_input(
    "🎯 Describe your task/need:", 
    value="a task management web app with user authentication",
    help="Describe what you want to build in detail"
)

col1, col2 = st.columns(2)
with col1:
    design_iterations = st.selectbox("Design Iterations:", [1, 2, 3], index=1)
with col2:
    code_iterations = st.selectbox("Code Iterations:", [1, 2, 3], index=1)

if st.button("🚀Generate", type="primary"):
    if user_query.strip():
        # Design Phase
        with st.spinner("Creating system design..."):
            final_design = iterative_design(user_query, design_iterations)
        
        st.markdown("## System Design")
        st.markdown(final_design)
        
        # Code Generation Phase  
        with st.spinner("Generating code implementation..."):
            final_code = iterative_code(final_design, code_iterations)
        
        st.markdown("## Code Implementation")
        
        # Parse and display code files
        parsed_files = parse_code_blocks(final_code)
        
        if parsed_files:
            st.markdown("###Generated Files")
            
            # Create tabs for different files
            if len(parsed_files) <= 5:
                tabs = st.tabs(list(parsed_files.keys()))
                for tab, (filename, content) in zip(tabs, parsed_files.items()):
                    with tab:
                        # Determine language for syntax highlighting
                        if '.' in filename:
                            ext = filename.split('.')[-1].lower()
                            lang_map = {
                                'py': 'python', 'js': 'javascript', 'html': 'html',
                                'css': 'css', 'json': 'json', 'yml': 'yaml',
                                'yaml': 'yaml', 'md': 'markdown', 'sql': 'sql'
                            }
                            lang = lang_map.get(ext, 'text')
                        else:
                            lang = 'text'
                        
                        st.code(content, language=lang)
            else:
                # Use expanders for many files
                for filename, content in parsed_files.items():
                    with st.expander(f"📄 {filename}"):
                        if '.' in filename:
                            ext = filename.split('.')[-1].lower()
                            lang_map = {
                                'py': 'python', 'js': 'javascript', 'html': 'html',
                                'css': 'css', 'json': 'json', 'yml': 'yaml'
                            }
                            lang = lang_map.get(ext, 'text')
                        else:
                            lang = 'text'
                        st.code(content, language=lang)
        else:
            st.code(final_code, language="python")
            
    else:
        st.warning("⚠️ Please enter a project description to get started.")