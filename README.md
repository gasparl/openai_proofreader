# Docx Proofreader with OpenAI

This repository provides a Python script for **proofreading `.docx` documents using OpenAI models**.
The script checks for spelling, grammar, and agreement errors in Hungarian or any language, splits large texts into manageable chunks, and saves a detailed error report in plain text format.

---

## Features

* Proofreads `.docx` files for spelling, typo, grammar, and agreement errors using advanced OpenAI models.
* Handles large documents by automatically splitting them into model-sized chunks.
* Error findings are clearly listed and explained in the output.
* **Model selection is easy:** choose your preferred model with a command-line flag.
* Each run produces a report with the model name, date, and source file included in the header and filename.
* **Robust API error handling**: Retries API calls on rate limits and network issues.
* **Easy to use**: Just specify your `.docx` input file, and receive a full report as a `.txt` file.

---

## Requirements

1. **Python 3.7 or higher**
2. Required libraries:

   * `openai`
   * `python-docx`
   * `tiktoken`
3. A valid OpenAI API key.

To install the requirements, run:

```bash
pip install openai python-docx tiktoken
```

---

## Setup

### Configuration File

Create a `config.json` file in the same directory as the script and add your OpenAI API key:

```json
{
    "OPENAI_API_KEY": "your-api-key-here"
}
```

---

## Usage

1. **Place your input `.docx` file** (e.g., `input.docx`) in the script directory.

2. **Run the script** from the command line.
   Basic usage:

   ```bash
   python proofreader.py
   ```

   **Specify a different model or input/output file** if you want:

   ```bash
   python proofreader.py --model o3 --input myfile.docx
   ```

3. The script will create an output file with a name like:

   ```
   proofreading_report_gpt-4o_20250702_1334.txt
   ```

   containing a detailed error report, the model name, the date, and the name of the original file.

---

## Output

* The report is saved as a plain text `.txt` file.
* The file header shows the date, model name, and source file.
* Each section (“szakasz”) corresponds to a chunk of your document, with errors listed and explained.
* If no errors are found, the report will state so.

---

## Customization

* To change which model is used by default, edit the `DEFAULT_MODEL` variable at the top of the script.
* To adjust chunk size or API retry behavior, modify the `CHUNK_TOKENS` or `RETRIES` variables.
* You can further refine the proofreading prompt for other languages or criteria in the `SYSTEM_PROMPT` variable in the script.

---

## Example Command

```bash
python proofreader.py --model o3 --input document_to_check.docx --output my_report.txt
```

---
