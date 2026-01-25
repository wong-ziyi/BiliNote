from typing import Optional, Union

from openai import OpenAI

from app.utils.logger import get_logger

logging= get_logger(__name__)
class OpenAICompatibleProvider:
    def __init__(self, api_key: str, base_url: str, model: Union[str, None]=None):
        base_url = self._normalize_base_url(base_url)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        if not base_url:
            return base_url

        if "generativelanguage.googleapis.com" in base_url:
            # 1. Strip trailing slashes
            url = base_url.rstrip("/")

            # 2. If it contains /models, strip it and everything after it
            if "/models" in url:
                url = url.split("/models")[0]

            # 3. Ensure it has /v1beta
            if "/v1beta" not in url:
                # Case for just base domain
                url = url.rstrip("/") + "/v1beta"
            else:
                # Case for v1beta being already present but maybe having other junk after it
                url = url.split("/v1beta")[0] + "/v1beta"

            # 4. Final suffix
            normalized = url + "/openai/"
            logging.info(f"Normalizing Gemini base_url: {base_url} -> {normalized}")
            return normalized

        return base_url

    @property
    def get_client(self):
        return self.client

    @staticmethod
    def test_connection(api_key: str, base_url: str) -> bool:
        try:
            base_url = OpenAICompatibleProvider._normalize_base_url(base_url)
            client = OpenAI(api_key=api_key, base_url=base_url)
            model = client.models.list()
            # for segment in model:
            #     print(segment)
            # print(model)
            logging.info("连通性测试成功")
            return True
        except Exception as e:
            logging.info(f"连通性测试失败：{e}")

            # print(f"Error connecting to OpenAI API: {e}")
            return False