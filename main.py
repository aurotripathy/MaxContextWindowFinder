import argparse
import concurrent.futures
import json
import logging
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Tuple


def sanitize_filename(value: str) -> str:
    """Make a value safe for use in filenames across platforms."""
    safe = value.replace(os.sep, "_")
    if os.altsep:
        safe = safe.replace(os.altsep, "_")
    safe = safe.replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in safe)


def setup_logging(model_name: str, log_level: str) -> str:
    """Setup logging configuration and return the log filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = sanitize_filename(model_name)
    log_filename = f"context_test_{safe_model_name}_{timestamp}.log"

    # Ensure the logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    level = logging.getLevelName(log_level.upper())
    if not isinstance(level, int):
        level = logging.ERROR

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    return log_path


def timeout_handler(signum, frame):
    raise TimeoutError("Query timed out")


def run_with_timeout(func, timeout_seconds: int):
    """Run a callable with a timeout using a worker thread."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            raise TimeoutError("Query timed out") from e


def retry_on_timeout(max_retries=3, timeout_seconds=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return run_with_timeout(lambda: func(*args, **kwargs), timeout_seconds)

                except TimeoutError as e:
                    logging.warning(f"Attempt {attempt + 1}/{max_retries} timed out after {timeout_seconds} seconds")
                    if attempt == max_retries - 1:
                        raise TimeoutError(f"All {max_retries} attempts timed out")
                    logging.info("Retrying...")
                    time.sleep(1)  # Brief pause before retry

            return None  # Should never reach here due to raise in last attempt

        return wrapper

    return decorator


def analyze_test_sentence() -> Tuple[str, int]:
    """Return the test sentence and its actual token count."""
    test_sentence = "This is a test sentence to measure context performance. "

    # Actual tokens (approximately):
    # "This" = 1
    # "is" = 1
    # "a" = 1
    # "test" = 1
    # "sentence" = 1
    # "to" = 1
    # "measure" = 1
    # "context" = 1
    # "performance" = 1
    # "." = 1
    # " " = several tokens, roughly 1-2 additional tokens

    actual_tokens = 11  # More accurate token count

    return test_sentence, actual_tokens


def generate_test_prompt(context_size: int) -> Tuple[str, int, int]:
    """Generate prompt and return prompt, its token count, and repetitions."""
    base_prompt = "Count the number of characters in the following text and explain your counting process. Here's the text:\n\n"
    # Base prompt tokens:
    # Approximately 15-17 tokens for the base prompt
    base_prompt_tokens = 16

    test_sentence, tokens_per_sentence = analyze_test_sentence()

    # Calculate how many repetitions we can fit
    available_tokens = context_size - base_prompt_tokens
    repetitions = max(1, available_tokens // tokens_per_sentence)

    repeated_text = test_sentence * repetitions
    full_prompt = base_prompt + repeated_text

    total_tokens = base_prompt_tokens + (repetitions * tokens_per_sentence)

    return full_prompt, total_tokens, repetitions


@retry_on_timeout(max_retries=3, timeout_seconds=60)
@dataclass
class ResponseMetrics:
    response_text: str
    completion_tokens: int
    elapsed_seconds: float


def build_base_url(endpoint_url: str, port: int) -> str:
    """Build a base URL from endpoint URL and port, preserving any existing port."""
    parsed = urllib.parse.urlparse(endpoint_url)
    if parsed.port is not None:
        return endpoint_url.rstrip("/")
    if parsed.scheme:
        return f"{endpoint_url.rstrip('/')}:{port}"
    return f"http://{endpoint_url.rstrip('/')}:{port}"


def run_openai_query(model: str, context_size: int, base_url: str) -> Tuple[ResponseMetrics, str, int]:
    """Run a query to an OpenAI-compatible endpoint and return response metrics."""
    try:
        full_prompt, estimated_tokens, repetitions = generate_test_prompt(context_size)

        logging.debug(f"Estimated tokens in prompt: {estimated_tokens}")
        logging.debug(f"Number of repetitions: {repetitions}")

        request_body = {
            "model": model,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0
        }
        request_data = json.dumps(request_body).encode("utf-8")
        request_url = f"{base_url}/v1/chat/completions"
        request = urllib.request.Request(
            request_url,
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        start_time = time.perf_counter()
        with urllib.request.urlopen(request) as response:
            response_bytes = response.read()
        elapsed_seconds = time.perf_counter() - start_time

        response_json = json.loads(response_bytes.decode("utf-8"))
        response_text = response_json["choices"][0]["message"]["content"]
        completion_tokens = response_json.get("usage", {}).get("completion_tokens", 0)

        metrics = ResponseMetrics(
            response_text=response_text,
            completion_tokens=completion_tokens,
            elapsed_seconds=elapsed_seconds
        )
        return metrics, full_prompt, estimated_tokens
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = ""
        if error_body:
            print(f"Server error response: {error_body}")
        logging.error(f"OpenAI-compatible endpoint returned HTTP error: {str(e)}")
        logging.info(f"Make sure the server is running and accessible at {base_url}")
        raise
    except urllib.error.URLError as e:
        logging.error(f"Failed to connect to OpenAI-compatible endpoint: {str(e)}")
        logging.info(f"Make sure the server is running and accessible at {base_url}")
        raise


def calculate_tokens_per_second(response: ResponseMetrics) -> float:
    """Calculate the tokens per second rate from the response metrics."""
    if response.elapsed_seconds <= 0:
        return 0.0
    return response.completion_tokens / response.elapsed_seconds

def test_context_size(model: str, context_size: int, base_url: str, num_tests: int = 3) -> float:
    """Run multiple tests at a specific context size and return the average tokens/sec."""
    tokens_per_second_list = []

    print(f"--- Context Size: {context_size} ---")
    logging.info(f"\nContext Size: {context_size}")
    logging.info("-" * 50)

    for i in range(num_tests):
        try:
            response, prompt, estimated_tokens = run_openai_query(model, context_size, base_url)
            tokens_per_second = calculate_tokens_per_second(response)
            tokens_per_second_list.append(tokens_per_second)

            # Log detailed test information
            test_info = {
                "Test Number": i + 1,
                "Prompt Length (chars)": len(prompt),
                "Prompt Tokens": estimated_tokens,
                "Response Length (chars)": len(response.response_text),
                "Response Words": len(response.response_text.split()),
                "Response Estimated Tokens": int(len(response.response_text.split()) * 1.3),
                "Total Tokens Processed": response.completion_tokens,
                "Tokens/sec": f"{tokens_per_second:.2f}",
                "Eval Duration": f"{response.elapsed_seconds:.2f}s"
            }

            logging.info(f"Test {i + 1} Details:")
            for key, value in test_info.items():
                logging.info(f"  {key}: {value}")

            logging.info("Prompt Preview (first 200 chars):")
            logging.info(f"  {prompt[:200]}...")
            logging.info("Response Preview (first 200 chars):")
            logging.info(f"  {response.response_text[:200]}...\n")

        except (TimeoutError, Exception) as e:
            logging.error(f"Error in test {i + 1}: {str(e)}")
            continue

    if tokens_per_second_list:
        avg_tokens_per_second = statistics.mean(tokens_per_second_list)
        logging.info(f"Average tokens/sec for context size {context_size}: {avg_tokens_per_second:.2f}")
        return avg_tokens_per_second
    else:
        logging.warning(f"All tests failed for context size {context_size}")
        return 0.0


def find_max_context(model: str, base_url: str, start_size: int = 1024, step_size: int = 1024,
                     minimum_token_rate: int = 10, num_tests: int = 3) -> Tuple[int, float]:
    """Find the maximum context size for a model that maintains acceptable performance."""
    context_size = start_size
    previous_context_size = start_size
    previous_tokens_per_second = float('inf')

    logging.info(f"Starting maximum context size test for model: {model}")
    logging.info(f"Parameters:")
    logging.info(f"  Minimum acceptable token rate: {minimum_token_rate} tokens/sec")
    logging.info(f"  Starting context size: {start_size}")
    logging.info(f"  Step size: {step_size}")
    logging.info(f"  Tests per context size: {num_tests}")
    logging.info("=" * 50)

    while True:
        try:
            logging.info(f"\nTesting context size: {context_size}")
            avg_tokens_per_second = test_context_size(model, context_size, base_url, num_tests)

            # Check token rate only
            is_good = avg_tokens_per_second >= minimum_token_rate
            status = "GOOD" if is_good else "SLOW"

            print(f"Status: {status}")
            logging.info(f"Token Rate: {avg_tokens_per_second:.2f} tokens/sec")

            if not is_good:
                reason = []
                if avg_tokens_per_second < minimum_token_rate:
                    reason.append(f"Token rate below minimum threshold of {minimum_token_rate}")
                logging.info(f"Stopping due to: {', '.join(reason)}")
                return previous_context_size, previous_tokens_per_second

            previous_context_size = context_size
            previous_tokens_per_second = avg_tokens_per_second
            context_size += step_size

        except Exception as e:
            logging.error(f"Error occurred at context size {context_size}: {str(e)}")
            return previous_context_size, previous_tokens_per_second


def run_context_test(model: str, endpoint_url: str, port: int, log_level: str, min_token_rate: int = 10, start: int = 1024,
                     step: int = 1024, tests: int = 3) -> None:
    """
    Run the context size test with the given parameters and log results.

    Args:
        model: The model name
        min_token_rate: Minimum acceptable tokens per second
        start: Starting context size
        step: Step size for context increments
        tests: Number of tests per context size
    """
    # Setup logging
    log_filename = setup_logging(model, log_level)
    logging.info(f"Log file created: {log_filename}")

    base_url = build_base_url(endpoint_url, port)
    logging.info(f"Using OpenAI-compatible endpoint: {base_url}")

    max_context, final_tokens_per_second = find_max_context(
        model, base_url, start, step, min_token_rate, tests
    )

    # Log final results
    logging.info("\n" + "=" * 60)
    logging.info("FINAL RESULTS:")
    logging.info(f"Maximum recommended context size: {max_context}")
    logging.info(f"Average tokens per second at max context: {final_tokens_per_second:.2f}")
    logging.info(f"Minimum token rate threshold: {min_token_rate}")
    logging.info("=" * 60)

    # Also print to console
    print(f"\nResults have been saved to: {log_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the maximum usable context size for an OpenAI-compatible model endpoint")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct", 
                        help="The model name (default: meta-llama/Llama-3.3-70B-Instruct)")
    parser.add_argument("--endpoint_url", default="eval.nxt.furiosa.ai",
                        help="OpenAI-compatible server endpoint URL (default: eval.nxt.furiosa.ai)")
    parser.add_argument("--port", type=int, default=24401,
                        help="OpenAI-compatible server port (default: 24401)")
    parser.add_argument("--log_level", default="ERROR",
                        help="Logging level (default: ERROR)")
    parser.add_argument("--min_token_rate", type=int, default=10,
                        help="The minimum acceptable tokens per second rate (default: 10)")
    parser.add_argument("--start", type=int, default=1024, help="Starting context size")
    parser.add_argument("--step", type=int, default=1024, help="Step size for context increments")
    parser.add_argument("--tests", type=int, default=1, help="Number of tests per context size")

    args = parser.parse_args()
    run_context_test(**vars(args))