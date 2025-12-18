from langflow.custom.custom_component.component import Component
from langflow.io import HandleInput, MessageTextInput, IntInput, Output
from langflow.schema.dataframe import DataFrame
from langflow.schema.data import Data

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient


class QdrantHTTPOnly(Component):
    display_name = "Qdrant (HTTP Only)"
    description = "Stores embeddings in Qdrant using HTTP only (no local storage) and retrieves documents via similarity search."
    icon = "database-2-line"
    name = "QdrantHTTPOnly"

    inputs = [
        HandleInput(
            name="data_inputs",
            display_name="Chunks",
            input_types=["DataFrame"],
            required=False,
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
        MessageTextInput(
            name="search_query",
            display_name="Search Query",
            required=False,
        ),
        IntInput(
            name="batch_size",
            display_name="Batch Size",
            value=64,
        ),
        IntInput(
            name="k",
            display_name="Number of Results (k)",
            value=4,
        ),
    ]

    outputs = [
        Output(
            display_name="Indexed Data",
            name="dataframe",
            method="index_data",
        ),
        Output(
            display_name="Retrieved Data",
            name="retrieved_dataframe",
            method="retrieve_data",
        ),
    ]

    def index_data(self) -> DataFrame:
        if not self.data_inputs:
            return DataFrame([])
        
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

    def retrieve_data(self) -> DataFrame:
        if not self.search_query:
            return DataFrame([])

        if not self.embeddings:
            raise ValueError("Embeddings are required for retrieval")

        # Create Qdrant client for HTTP connection
        client = QdrantClient(url=self.qdrant_url)

        # Connect to existing Qdrant collection using the client
        # The Qdrant constructor accepts: client, collection_name, and embeddings (plural)
        qdrant = Qdrant(
            client=client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )

        # Perform similarity search
        results = qdrant.similarity_search(
            query=self.search_query,
            k=self.k,
        )

        # Convert results to DataFrame format
        data_items = []
        for doc in results:
            data_items.append(
                Data(
                    text=doc.page_content,
                    data=doc.metadata,
                )
            )

        return DataFrame(data_items)