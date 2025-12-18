from langflow.custom.custom_component.component import Component
from langflow.io import HandleInput, MessageTextInput, IntInput, Output
from langflow.schema.dataframe import DataFrame

from langchain_community.vectorstores import Qdrant


class QdrantHTTPOnly(Component):
    display_name = "Qdrant (HTTP Only)"
    description = "Stores embeddings in Qdrant using HTTP only (no local storage)."
    icon = "database-2-line"
    name = "QdrantHTTPOnly"

    inputs = [
        HandleInput(
            name="data_inputs",
            display_name="Chunks",
            input_types=["DataFrame"],
            required=True,
        ),
        HandleInput(
            name="embeddings",
            display_name="Embeddings",
            input_types=["Embeddings"],
            required=True,
        ),
        MessageTextInput(
            name="qdrant_url",
            display_name="Qdrant URL",
            value="http://localhost:6333",
            required=True,
        ),
        MessageTextInput(
            name="collection_name",
            display_name="Collection Name",
            required=True,
        ),
        IntInput(
            name="batch_size",
            display_name="Batch Size",
            value=64,
        ),
    ]

    outputs = [
        Output(
            display_name="Indexed Data",
            name="dataframe",
            method="index_data",
        )
    ]

    def index_data(self) -> DataFrame:
        if not isinstance(self.data_inputs, DataFrame):
            raise TypeError("Input must be a DataFrame")

        if not len(self.data_inputs):
            raise TypeError("Input DataFrame is empty")

        documents = self.data_inputs.to_lc_documents()

        # ✅ CORRECT API FOR THIS LANGCHAIN VERSION
        Qdrant.from_documents(
            documents=documents,
            embedding=self.embeddings,
            url=self.qdrant_url,
            collection_name=self.collection_name,
            batch_size=self.batch_size,
        )

        # Pass-through
        return self.data_inputs