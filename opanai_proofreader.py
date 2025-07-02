import asyncio
import json
import logging
import random
import sys
from pathlib import Path
from functools import lru_cache
from openai import AsyncOpenAI
from openai import OpenAIError, RateLimitError, APIConnectionError, APITimeoutError
from docx import Document
import tiktoken
from datetime import datetime
import argparse

# --- Model selection (change here!) ---
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4.1",
    "o3",
    "o3-pro",
    "o3-mini-high",
]
DEFAULT_MODEL = "gpt-4o"
# --------------------------------------

CHUNK_TOKENS = 2048
MAX_COMPLETION_TOKENS = 800
RETRIES = 5
CONCURRENCY = 5

script_dir = Path(__file__).resolve().parent
config_path = Path(script_dir, "config.json")

try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    api_key = config["OPENAI_API_KEY"]
except Exception as e:
    logging.critical("Missing or invalid config.json, or OPENAI_API_KEY missing.")
    raise SystemExit(1)

client = AsyncOpenAI(api_key=api_key)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@lru_cache(maxsize=1)
def _encoder(model_name=DEFAULT_MODEL):
    # Safe fallback if tiktoken doesn't know the model
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def token_len(text: str, model_name=DEFAULT_MODEL) -> int:
    return len(_encoder(model_name).encode(text))

def read_docx(path: str) -> list[str]:
    return [p.text for p in Document(str(path)).paragraphs if p.text.strip()]

def split_paragraphs(paragraphs: list[str], model_name=DEFAULT_MODEL) -> list[str]:
    chunks, current, tokens = [], [], 0
    for p in paragraphs:
        p_tokens = token_len(p, model_name) + 1  # newline
        # If single paragraph too long, split it by sentence
        if p_tokens > CHUNK_TOKENS:
            import re
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', p.replace('\n', ' ')) if s.strip()]
            sent_chunk, sent_tokens = [], 0
            for s in sents:
                s_token = token_len(s, model_name) + 1
                if sent_tokens + s_token > CHUNK_TOKENS and sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                    sent_chunk, sent_tokens = [], 0
                sent_chunk.append(s)
                sent_tokens += s_token
            if sent_chunk:
                chunks.append(" ".join(sent_chunk))
            continue

        if tokens + p_tokens > CHUNK_TOKENS and current:
            chunks.append("\n".join(current))
            current, tokens = [], 0
        current.append(p)
        tokens += p_tokens
    if current:
        chunks.append("\n".join(current))
    return chunks

SYSTEM_PROMPT = (
    "Ellenőrizd, van-e benne egyértelmű helyesírási, elírási, ragozási vagy nyelvtani egyeztetési hiba, "
    "különös figyelmet fordítva az egybe- és különírási esetekre (pl. mindeközben vs. mind eközben). "
    "Ne javasolj stílus- vagy tartalmi átfogalmazást, és ne írd át a szöveget. "
    "A hibákat pontokba szedve sorold fel: idézd a problémás részletet, majd adj rövid magyarázatot "
    "(pl. „szabadag - elírás, helyesen szabadság”). "
    "Ha nem találsz nyilvánvaló hibát, mondd, hogy nincs hiba."
)

sem = asyncio.Semaphore(CONCURRENCY)

async def proofread(chunk: str, model_name: str) -> str:
    async with sem:
        for attempt in range(RETRIES):
            try:
                prompt_tokens = token_len(SYSTEM_PROMPT, model_name)
                chunk_tokens = token_len(chunk, model_name)
                # Set context window (gpt-4o/o3: 128k, gpt-4.1: 1M)
                limit = 1_000_000 if model_name.startswith("gpt-4.1") else 128_000
                available = limit - prompt_tokens - chunk_tokens
                if available <= 0:
                    raise ValueError("Chunk plus prompt exceeds model context window")
                max_completion = min(MAX_COMPLETION_TOKENS, max(32, available))
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": chunk}
                    ],
                    max_tokens=max_completion,
                    temperature=0.1
                )
                return resp.choices[0].message.content.strip()
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                if attempt == RETRIES - 1:
                    raise
                wait = 2 ** attempt + random.random()
                logging.warning(f"{type(e).__name__}: {e}. Retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
            except OpenAIError as e:
                raise

async def main(input_docx: str, output_txt: str, model_name: str):
    try:
        paragraphs = read_docx(input_docx)
    except Exception as e:
        logging.error(f"Failed to read DOCX: {e}")
        return

    chunks = split_paragraphs(paragraphs, model_name)
    logging.info(f"{len(chunks)} chunks to proofread.")

    tasks = [proofread(c, model_name) for c in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    now = datetime.now().strftime('%Y%m%d_%H%M')
    header = (
        f"--- Proofreading Report ---\n"
        f"Model: {model_name}\n"
        f"Date: {now}\n"
        f"Source file: {Path(input_docx).name}\n"
        f"---\n\n"
    )

    output_path = Path(output_txt)
    out_lines = [header]
    for i, r in enumerate(results, 1):
        if isinstance(r, Exception):
            logging.error(f"Chunk {i} failed: {r}")
            out_lines.append(f"-- Szakasz {i} --\nHiba történt az ellenőrzés során: {r}\n\n")
        else:
            out_lines.append(f"-- Szakasz {i} --\n{r}\n\n")
    output_path.write_text("".join(out_lines), encoding="utf-8")
    logging.info(f"Proofreading completed. Output written to {output_path}")

if __name__ == "__main__":
    # Windows event loop policy fix
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser(description="Proofread a DOCX using an OpenAI model.")
    parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS, default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--input", type=str, default="input.docx", help="Input DOCX file")
    parser.add_argument("--output", type=str, default=None, help="Output TXT file (default: auto-named)")
    args = parser.parse_args()

    # Absolute paths from script_dir
    input_docx = Path(script_dir, args.input)
    now = datetime.now().strftime('%Y%m%d_%H%M')
    output_txt = (
        Path(script_dir, args.output)
        if args.output else
        Path(script_dir, f"proofreading_report_{args.model}_{now}.txt")
    )

    asyncio.run(main(input_docx, output_txt, args.model))
