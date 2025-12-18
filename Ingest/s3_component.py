import os
import tempfile

import boto3
from langchain.document_loaders import PyPDFLoader

from langflow.custom.custom_component.component import Component
from langflow.io import MessageTextInput, SecretStrInput, IntInput, Output
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame


class CloudianS3LoadPDFs(Component):
    display_name = "Cloudian S3 Load PDFs from Folder"
    description = "Loads and extracts text from PDF files stored in an S3-compatible bucket, with page batching."
    icon = "database-2-line"
    name = "CloudianS3LoadPDFs"

    inputs = [
        MessageTextInput(name="s3_endpoint", display_name="S3 Endpoint", required=True),
        MessageTextInput(name="access_key", display_name="Access Key", required=True),
        SecretStrInput(name="secret_key", display_name="Secret Key", required=True),
        MessageTextInput(name="bucket_name", display_name="Bucket Name", required=True),
        MessageTextInput(name="folder_prefix", display_name="Folder / Prefix", required=True),
        IntInput(name="start_page", display_name="Start Page (1-based)", value=1),
        IntInput(name="pages_per_batch", display_name="Pages per Batch (0 = all)", value=0),
    ]

    outputs = [
        Output(display_name="Documents", name="dataframe", method="load_documents")
    ]

    def load_documents(self) -> DataFrame:
        s3 = boto3.client(
            "s3",
            endpoint_url=self.s3_endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

        response = s3.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=self.folder_prefix,
        )

        data_items = []

        start_index = max(self.start_page - 1, 0)
        max_pages = self.pages_per_batch if self.pages_per_batch > 0 else None

        for obj in response.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".pdf"):
                continue

            with tempfile.TemporaryDirectory() as tmp:
                local_path = os.path.join(tmp, os.path.basename(key))
                s3.download_file(self.bucket_name, key, local_path)

                loader = PyPDFLoader(local_path)
                pages = loader.load()  # one Document per page

                if max_pages is not None:
                    pages = pages[start_index : start_index + max_pages]
                else:
                    pages = pages[start_index:]

                for page_number, doc in enumerate(pages, start=start_index + 1):
                    if not doc.page_content.strip():
                        continue

                    data_items.append(
                        Data(
                            text=doc.page_content,
                            data={
                                **doc.metadata,
                                "bucket": self.bucket_name,
                                "key": key,
                                "endpoint": self.s3_endpoint,
                                "page": page_number,
                            },
                        )
                    )

        return DataFrame(data_items)