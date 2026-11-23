import json
import os
import sys
import ctypes
import importlib
import subprocess
import sysconfig

ON_ZEROGPU = bool(os.environ.get("SPACES_ZERO_GPU"))
WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/"

def pip_install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])

def install_llama(flavor, force=False):
    extra = ["--force-reinstall", "--no-deps"] if force else []
    pip_install(
        "llama-cpp-python",
        "--only-binary=llama-cpp-python",
        "--extra-index-url", WHEEL_INDEX + flavor,
        *extra,
    )

def load_cuda_libs():
    pip_install("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12")
    root = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
    lib_dirs = [
        os.path.join(root, "cuda_runtime", "lib"),
        os.path.join(root, "cublas", "lib"),
    ]
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]
    )
    for name in ("libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12"):
        for d in lib_dirs:
            path = os.path.join(d, name)
            if os.path.exists(path):
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

def ensure_llama():
    global ON_ZEROGPU
    if ON_ZEROGPU:
        try:
            install_llama("cu124")
            load_cuda_libs()
            import llama_cpp
            return
        except Exception as e:
            # CPU build is the fallback if the CUDA build can't load on ZeroGPU
            print(f"CUDA llama-cpp-python failed to load ({e!r}); using CPU build instead")
            ON_ZEROGPU = False
            for name in [m for m in sys.modules if m.startswith("llama_cpp")]:
                del sys.modules[name]
            install_llama("cpu", force=True)
            importlib.invalidate_caches()
            return
    try:
        import llama_cpp
    except ImportError:
        install_llama("cpu")

ensure_llama()

def get_duration(message, history, max_tokens, temperature, top_p, progress=None):
    return 90 if message.get("files") else 30

try:
    import spaces
    gpu = spaces.GPU(duration=get_duration)
except ImportError:
    def gpu(fn):
        return fn

import pypdf
import gradio as gr
from docx import Document
from ftfy import TextFixerConfig, fix_text
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from striprtf.striprtf import rtf_to_text

FTFY_CONFIG = TextFixerConfig(uncurl_quotes=False)

print("Downloading GGUF...")
model_path = hf_hub_download(
    repo_id="ianro04/PentaGemma", 
    filename="gemma-4-e2b-it.Q4_K_S.gguf"
)

# Load model with CPU (also the fallback if ZeroGPU is unavailable, out of quota, or fails)
print("Loading model into RAM...")
cpu_llm = Llama(
    model_path=model_path,
    n_ctx=3072, 
    n_threads=2,
    n_threads_batch=2,
    n_gpu_layers=0,
)
print("Ready!")

_gpu_llm = None

def get_gpu_llm():
    global _gpu_llm
    if _gpu_llm is None:
        _gpu_llm = Llama(
            model_path=model_path,
            n_ctx=3072,
            n_gpu_layers=-1,
        )
    return _gpu_llm

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "penta_prompt.json"), encoding="utf-8") as f:
    PENTA_PROMPT = "".join(json.load(f))

def read_file(file_path): # Alt input method for Space
    if file_path is None:
        return ""
    elif file_path.lower().endswith((".txt", ".md")):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    elif file_path.lower().endswith(".rtf"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return rtf_to_text(f.read())
    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file_path.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(file_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    return ""

def chunk_text(text, max_tokens=2048): # Break down lengthier text into chunks
    tokens = cpu_llm.tokenize(text.encode("utf-8"))
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = cpu_llm.detokenize(chunk_tokens).decode("utf-8", errors="ignore")
        chunks.append(chunk_text)

    return chunks

def run(
    llm,
    message,
    history: list[dict[str, str]],
    max_tokens,
    temperature,
    top_p,
    progress,
):
    user_text = message.get("text", "")
    files = message.get("files", [])

    file_text = ""
    if files:
        for file_path in files:
            content = fix_text(read_file(file_path), config=FTFY_CONFIG)
            if content:
                file_text += f"\n\n--- Document Uploaded by User ---\n{content}"

    if not file_text: # Base case for non-summary tasks
        messages = [{"role": "system", "content": PENTA_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        stream = llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, 
            temperature=temperature, top_p=top_p, stream=True
        )
        response = ""
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                response += delta["content"]
                yield response
        return

    chunks = chunk_text(file_text)

    if len(chunks) == 1: # Summarizing text that fits in a single "chunk"
        final_prompt = user_text if user_text.strip() else "Summarize the following document: "
        final_prompt += file_text

        messages = [{"role": "system", "content": PENTA_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": final_prompt})

        stream = llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, 
            temperature=temperature, top_p=top_p, stream=True
        )
        response = ""
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                response += delta["content"]
                yield response

    else:
        # For summarizing lengthier documents by breaking into chunks, summarizing each chunk, then summarizing a concat of the summaries
        progress(0, desc="Processing large document...")

        intermediate_summaries = []

        for i, chunk in enumerate(chunks):
            progress((i, len(chunks)), desc=f"Summarizing part {i+1} of {len(chunks)}")
            map_prompt = f"Summarize the following section of a document in exactly 5 sentences:\n\n{chunk}"
            chunk_response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": PENTA_PROMPT},
                    {"role": "user", "content": map_prompt}
                ],
                max_tokens=min(max_tokens, 200),
                temperature=temperature,
                top_p=top_p,
                stream=False 
            )

            summary = chunk_response["choices"][0]["message"]["content"]
            intermediate_summaries.append(summary)

        progress((len(chunks), len(chunks)), desc="Writing final summary...")

        combined_summaries = "\n\n".join(intermediate_summaries)

        reduce_instruction = user_text.strip() if user_text.strip() else "Combine the following summaries of different parts of a document into a single, cohesive 5-sentence summary:"
        final_reduce_prompt = f"{reduce_instruction}\n\n--- Intermediate Summaries ---\n{combined_summaries}"

        messages = [{"role": "system", "content": PENTA_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": final_reduce_prompt})

        # Final result
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True
        )

        response = ""
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                response += delta["content"]
                yield response

@gpu
def respond_gpu(
    message,
    history: list[dict[str, str]],
    max_tokens,
    temperature,
    top_p,
    progress=gr.Progress(),
):
    yield from run(get_gpu_llm(), message, history, max_tokens, temperature, top_p, progress)

def respond(
    message,
    history: list[dict[str, str]],
    max_tokens,
    temperature,
    top_p,
    progress=gr.Progress(),
):
    if ON_ZEROGPU:
        try:
            yield from respond_gpu(message, history, max_tokens, temperature, top_p, progress)
            return
        except Exception as e:
            print(f"ZeroGPU failed ({e!r}); falling back to CPU")
    yield from run(cpu_llm, message, history, max_tokens, temperature, top_p, progress)

chatbot = gr.ChatInterface(
    respond,
    multimodal=True,
    textbox=gr.MultimodalTextbox(
        file_types=[".pdf", ".docx", ".txt", ".md", ".rtf"],
        placeholder="Type a message or upload a document to summarize...",
    ),
    additional_inputs=[
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
    title="PentaGemma: 5-Sentence Summarizer",
    description="PentaGemma is a fine-tuned version of Gemma 4 E2B IT, but for 5 sentence text summarization and general obsession with the number 5. Accepts .txt, .rtf, .md, .docx, .pdf file uploads for summarization. Otherwise usable as a basic chatbot.<br>Note: Uses ZeroGPU when available and basic CPU as a fallback, which can be <u>very</u> slow for larger documents.",
)

if __name__ == "__main__":
    demo = chatbot.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
    )
