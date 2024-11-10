from typing import Any, Dict, List, Optional, Tuple

import requests


REQUEST_TIMEOUT = 60


class OPEABackend:
    def __init__(
        self,
        api_url: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        self.session = requests.Session()
        self.session.headers.update(headers)

        self.api_url = api_url
        self.model_kwargs = model_kwargs or {}

    def embed(self, text: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:
        url = f"{self.api_url}/embeddings"

        try:
            res = self.session.post(
                url,
                json={
                    "text": text,
                    **self.model_kwargs,
                },
                timeout=REQUEST_TIMEOUT,
            )
            res.raise_for_status()
        except requests.HTTPError as e:
                msg = f"Failed to query embedding endpoint: Error - {e.response.text}"
                raise ValueError(msg) from e
        
        data = res.json()
        embedding = data['embedding']
        del data['embedding']

        return embedding, data