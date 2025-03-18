import os
import logging
import requests
import fitz
import asyncio
import json
import httpx
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
import uvicorn
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()


class URLRequest(BaseModel):
    url: str


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI backend is running!"}


@app.post("/summarize_arxiv/")
async def summarize_arxiv(request: URLRequest):
    """Downloads an Arxiv PDF, extracts text, and summarizes it using Ollama (Gemma 3) in parallel."""
    try:
        url = request.url
        logger.info("---------------------------------------------------------")
        logger.info(f"Downloading PDF from: {url}")

        pdf_path = download_pdf(url)
        if not pdf_path:
            return {"error": "Failed to download PDF. Check the URL."}

        logger.info(f"PDF saved at: {pdf_path}")

        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "No text extracted from PDF"}

        logger.info(f"Extracted text length: {len(text)} characters")
        logger.info("---------------------------------------------------------")

        # Summarize extracted text in parallel
        summary = await summarize_text_parallel(text)
        logger.info("Summarization complete")

        return {"summary": summary}

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {"error": "Failed to process PDF"}


def download_pdf(url):
    """Downloads a PDF from a given URL and saves it locally."""
    try:
        if not url.startswith("https://arxiv.org/pdf/"):
            logger.error(f"Invalid URL: {url}")
            return None  # Prevents downloading non-Arxiv PDFs

        response = requests.get(url, timeout=30)  # Set timeout to prevent long waits
        if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
            # Create the downloaded_pdf directory if it doesn't exist
            download_dir = "downloaded_pdf"
            os.makedirs(download_dir, exist_ok=True)
            
            # Extract the filename from the URL or use a default name
            pdf_filename = url.split('/')[-1] if url.endswith('.pdf') else "arxiv_paper.pdf"
            
            # Add timestamp to the filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name_parts = pdf_filename.rsplit('.', 1)
            pdf_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
            
            pdf_path = os.path.join(download_dir, pdf_filename)
            
            # Save the PDF
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            return pdf_path
        else:
            logger.error(f"Failed to download PDF: {response.status_code} (Not a valid PDF)")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF: {e}")
        return None


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF (faster than Unstructured PDFLoader)."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""

async def summarize_chunk_with_retry(chunk, chunk_id, total_chunks, max_retries=2):
    """Retry mechanism wrapper for summarize_chunk_wrapper."""
    retries = 0
    while retries <= max_retries:
        try:
            if retries > 0:
                logger.info(f"🔄 Retry attempt {retries}/{max_retries} for chunk {chunk_id}/{total_chunks}")
            
            result = await summarize_chunk_wrapper(chunk, chunk_id, total_chunks)
            
            # If the result starts with "Error", it means there was an error but no exception was thrown
            if isinstance(result, str) and result.startswith("Error"):
                logger.warning(f"⚠️ Soft error on attempt {retries+1}/{max_retries+1} for chunk {chunk_id}: {result}")
                retries += 1
                if retries <= max_retries:
                    # Exponential backoff: 5s, 10s, 20s, etc.
                    wait_time = 5 * (2 ** (retries - 1))
                    logger.info(f"⏳ Waiting {wait_time}s before retry for chunk {chunk_id}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ All retry attempts failed for chunk {chunk_id}")
                    return result
            else:
                # Success
                if retries > 0:
                    logger.info(f"✅ Successfully processed chunk {chunk_id} after {retries} retries")
                return result
                
        except Exception as e:
            retries += 1
            logger.error(f"❌ Exception on attempt {retries}/{max_retries+1} for chunk {chunk_id}: {str(e)}")
            
            if retries <= max_retries:
                # Exponential backoff
                wait_time = 5 * (2 ** (retries - 1))
                logger.info(f"⏳ Waiting {wait_time}s before retry for chunk {chunk_id}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"❌ All retry attempts exhausted for chunk {chunk_id}")
                return f"Error processing chunk {chunk_id} after {max_retries+1} attempts: {str(e)}"
    
    # This should never be reached, but just in case
    return f"Error: Unexpected end of retry loop for chunk {chunk_id}"


async def summarize_text_parallel(text):
    """Process text in chunks optimized for Gemma 3's 128K context window with full parallelism and retry logic."""
    token_estimate = len(text) // 4
    logger.info(f"📝 Token Estimate: {token_estimate}")

    # Use larger chunks since Gemma 3 can handle 128K tokens
    chunk_size = 10000 * 4  # Approximately 32K tokens per chunk
    chunk_overlap = 100   # Larger overlap to maintain context
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    logger.info("---------------------------------------------------------")
    logger.info(f"📚 Split text into {len(chunks)} chunks")
    logger.info("---------------------------------------------------------")
    
    # Log chunk details
    for i, chunk in enumerate(chunks, 1):
        chunk_length = len(chunk)
        logger.info(f"📊 Length: {chunk_length} characters ({chunk_length // 4} estimated tokens)")

    logger.info("---------------------------------------------------------")
    logger.info(f"🔄 Processing {len(chunks)} chunks in parallel with retry mechanism...")

    # Create tasks for each chunk with retry mechanism
    tasks = [summarize_chunk_with_retry(chunk, i+1, len(chunks), max_retries=2) for i, chunk in enumerate(chunks)]
    
    # Process chunks with proper error handling at the gather level
    try:
        # Using return_exceptions=True to prevent one failure from canceling all tasks
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process the results, handling any exceptions
        processed_summaries = []
        for i, result in enumerate(summaries):
            if isinstance(result, Exception):
                # An exception was returned
                logger.error(f"❌ Task for chunk {i+1} returned an exception: {str(result)}")
                processed_summaries.append(f"Error processing chunk {i+1}: {str(result)}")
            else:
                # Normal result
                processed_summaries.append(result)
                
        summaries = processed_summaries
        
    except Exception as e:
        logger.error(f"❌ Critical error in gather operation: {str(e)}")
        return f"Critical error during processing: {str(e)}"

    logger.info("✅ All chunks processed (with or without errors)")

    # Check if we have at least some successful results
    successful_summaries = [s for s in summaries if not (isinstance(s, str) and s.startswith("Error"))]
    if not successful_summaries:
        logger.warning("⚠️ No successful summaries were generated.")
        return "No meaningful summary could be generated. All chunks failed processing."

    # Combine summaries with section markers, including error messages for failed chunks
    combined_chunk_summaries = "\n\n".join(f"Section {i+1}:\n{summary}" for i, summary in enumerate(summaries))
    logger.info(f"📝 Combined summaries length: {len(combined_chunk_summaries)} characters")
    logger.info("🔄 Generating final summary...")

    # Create final summary with system message
    final_messages = [
        {
            "role": "system",
            "content": "You are a technical documentation writer. Focus ONLY on technical details, implementations, and results. DO NOT mention papers, citations, or authors."
        },
        {
            "role": "user",
            "content": f"""Create a comprehensive technical document focusing ONLY on the implementation and results.
            Structure the content into these sections:

            1. System Architecture
            2. Technical Implementation
            3. Infrastructure & Setup
            4. Performance Analysis
            5. Optimization Techniques

            CRITICAL INSTRUCTIONS:
            - Focus ONLY on technical details and implementations
            - Include specific numbers, metrics, and measurements
            - Explain HOW things work
            - DO NOT include any citations or references
            - DO NOT mention other research or related work
            - Some sections may contain error messages - please ignore these and work with available information

            Content to organize:
            {combined_chunk_summaries}
            """
        }
    ]

    # Use async http client for the final summary with retry logic
    max_retries = 2
    retry_count = 0
    final_response = None
    
    while retry_count <= max_retries:
        try:
            # Use async http client for the final summary as well
            payload = {
                "model": "gemma3:4b",
                "messages": final_messages,
                "stream": False
            }
            
            logger.info(f"📤 Sending final summary request (attempt {retry_count+1}/{max_retries+1})")
            # Make async HTTP request with increased timeout for final summary
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/chat",
                    json=payload,
                    timeout=httpx.Timeout(connect=60, read=3600, write=60, pool=60)  # 15-minute read timeout
                )
                
                logger.info(f"📥 Received final summary response, status code: {response.status_code}")
                
                if response.status_code != 200:
                    raise Exception(f"API returned non-200 status code: {response.status_code} - {response.text}")
                    
                final_response = response.json()
                break  # Success, exit retry loop
                
        except Exception as e:
            retry_count += 1
            logger.error(f"❌ Error generating final summary (attempt {retry_count}/{max_retries+1}): {str(e)}")
            
            if retry_count <= max_retries:
                # Exponential backoff
                wait_time = 10 * (2 ** (retry_count - 1))
                logger.info(f"⏳ Waiting {wait_time}s before retrying final summary generation")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"❌ All retry attempts for final summary failed")
                return "Failed to generate final summary after multiple attempts. Please check the logs for details."
    
    if not final_response:
        return "Failed to generate final summary. Please check the logs for details."
    
    logger.info("✅ Final summary generated")
    logger.info(f"📊 Final summary length: {len(final_response['message']['content'])} characters")
    return final_response['message']['content']

async def summarize_chunk_wrapper(chunk, chunk_id, total_chunks):
    """Asynchronous wrapper for summarizing a single chunk using Ollama via async httpx."""
    logger.info("---------------------------------------------------------")
    logger.info(f"🎯 Starting processing of chunk {chunk_id}/{total_chunks}")
    try:
        # Add system message to better control output
        messages = [
            {"role": "system", "content": "Extract only technical details. No citations or references."},
            {"role": "user", "content": f"Extract technical content: {chunk}"}
        ]
        
        # Use httpx for truly parallel API calls
        payload = {
            "model": "gemma3:4b",
            "messages": messages,
            "stream": False
        }
        
        # Add better timeout and error handling
        try:
            # Make async HTTP request directly to Ollama API
            async with httpx.AsyncClient(timeout=3600) as client:  # Increased timeout to 10 minutes
                logger.info(f"📤 Sending request for chunk {chunk_id}/{total_chunks} to Ollama API - Gemma3 ")
                response = await client.post(
                    "http://localhost:11434/api/chat",  # Default Ollama API endpoint
                    json=payload,
                    # Adding connection timeout and timeout parameters
                    timeout=httpx.Timeout(connect=60, read=3600, write=60, pool=60)
                )
                logger.info("---------------------------------------------------------")
                logger.info(f"📥 Received response for chunk {chunk_id}/{total_chunks}, status code: {response.status_code}")
                
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return f"Error processing chunk {chunk_id}: API returned status code {response.status_code}"
                    
                response_data = response.json()
                summary = response_data['message']['content']
            
            logger.info(f"✅ Completed chunk {chunk_id}/{total_chunks}")
            logger.info(f"📑 Summary length: {len(summary)} characters")
            logger.info("---------------------------------------------------------")
            return summary
            
        except httpx.TimeoutException as te:
            error_msg = f"Timeout error for chunk {chunk_id}: {str(te)}"
            logger.error(error_msg)
            return f"Error in chunk {chunk_id}: Request timed out after 30 minutes. Consider increasing the timeout or reducing chunk size."
            
        except httpx.ConnectionError as ce:
            error_msg = f"Connection error for chunk {chunk_id}: {str(ce)}"
            logger.error(error_msg)
            return f"Error in chunk {chunk_id}: Could not connect to Ollama API. Check if Ollama is running correctly."
            
    except Exception as e:
        # Capture and log the full exception details
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"❌ Error processing chunk {chunk_id}: {str(e)}")
        logger.error(f"Traceback: {error_details}")
        return f"Error processing chunk {chunk_id}: {str(e)}"


# Keep this function as a reference or remove it as it's been replaced by summarize_chunk_wrapper
def summarize_chunk(chunk, chunk_id):
    """Summarizes a single chunk using Ollama (Gemma 3 LLM)."""
    logger.info(f"\n{'=' * 40} Processing Chunk {chunk_id} {'=' * 40}")
    logger.info(f"📄 Input chunk length: {len(chunk)} characters")

    prompt = f"""
    You are a technical content extractor. Extract and explain ONLY the technical details from this section.

    Focus on:
    1. **System Architecture** – Design, component interactions, algorithms, configurations.
    2. **Implementation** – Code/pseudocode, data structures, formulas (with explanations), parameter values.
    3. **Experiments** – Hardware (GPUs, RAM), software versions, dataset size, training hyperparameters.
    4. **Results** – Performance metrics (accuracy, latency, memory usage), comparisons.

    **Rules:**
    - NO citations, references, or related work.
    - NO mention of authors or external papers.
    - ONLY technical details, numbers, and implementations.

    Text to analyze:
    {chunk}
    """
    try:
        logger.info(f"🤖 Sending chunk {chunk_id} to Ollama...")
        response = ollama.chat(model="gemma3:4b", messages=[{"role": "user", "content": prompt}])
        summary = response['message']['content']
        logger.info(f"✅ Successfully processed chunk {chunk_id}")
        logger.info(f"📊 Summary length: {len(summary)} characters")
        print(summary)
        return summary
    except Exception as e:
        logger.error(f"❌ Error summarizing chunk {chunk_id}: {e}")
        return f"Error summarizing chunk {chunk_id}"


if __name__ == "__main__":
    logger.info("Starting FastAPI server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")