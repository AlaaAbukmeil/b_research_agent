import requests
import time


class DifyClient:
    """
    Thin wrapper around the Dify Workflow execution API.
    Handles retries with exponential backoff.
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 90):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def run_workflow(self, inputs: dict, max_retries: int = 3) -> dict:
        """
        Execute a Dify workflow in blocking mode.
        Returns the outputs dict on success; raises on failure.
        """
        url = f"{self.base_url}/workflows/run"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": inputs,
            "response_mode": "blocking",
            "user": "research-agent",
        }

        last_err = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
                resp.raise_for_status()
                body = resp.json()

                data = body.get("data", body)  # Dify wraps in "data"
                if data.get("status") == "succeeded":
                    return data.get("outputs", {})

                error_msg = data.get("error", "Unknown workflow error")
                last_err = RuntimeError(error_msg)

            except requests.exceptions.Timeout:
                last_err = TimeoutError(
                    f"Dify workflow timed out after {self.timeout}s"
                )
            except requests.exceptions.RequestException as exc:
                last_err = exc

            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(
                    f"  ⚠ Dify call failed (attempt {attempt + 1}/{max_retries}): "
                    f"{last_err}  — retrying in {wait}s"
                )
                time.sleep(wait)

        raise last_err  # type: ignore[misc]