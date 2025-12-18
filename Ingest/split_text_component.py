from langchain_text_splitters import CharacterTextSplitter

from langflow.custom.custom_component.component import Component
from langflow.io import HandleInput, IntInput, MessageTextInput, Output
from langflow.schema.data import Data
from langflow.schema.dataframe import DataFrame
from langflow.schema.message import Message
from langflow.utils.util import unescape_string


class SplitTextCustom(Component):
    display_name = "Split Text"
    description = "Split text into chunks based on specified criteria."
    icon = "scissors-line-dashed"
    name = "SplitTextCustom"

    inputs = [
        HandleInput(
            name="data_inputs",
            display_name="Input",
            info="The data with texts to split into chunks.",
            input_types=["Data", "DataFrame", "Message"],
            required=True,
        ),
        IntInput(
            name="chunk_overlap",
            display_name="Chunk Overlap",
            value=200,
        ),
        IntInput(
            name="chunk_size",
            display_name="Chunk Size",
            value=1000,
        ),
        MessageTextInput(
            name="separator",
            display_name="Separator",
            value="\n",
        ),
    ]

    outputs = [
        Output(
            display_name="Chunks",
            name="dataframe",
            method="split_text",
        )
    ]

    # Convert LangChain Documents → LangFlow Data
    def _docs_to_data(self, docs) -> list[Data]:
        return [Data(text=doc.page_content, data=doc.metadata) for doc in docs]

    def split_text_base(self):
        separator = unescape_string(self.separator)

        # ---- Convert input → LangChain Documents ----
        if isinstance(self.data_inputs, DataFrame):
            if not len(self.data_inputs):
                raise TypeError("DataFrame is empty")
            documents = self.data_inputs.to_lc_documents()

        elif isinstance(self.data_inputs, Message):
            self.data_inputs = [self.data_inputs.to_data()]
            return self.split_text_base()

        else:
            if not self.data_inputs:
                raise TypeError("No data inputs provided")

            if isinstance(self.data_inputs, Data):
                documents = [self.data_inputs.to_lc_document()]
            else:
                documents = [
                    d.to_lc_document()
                    for d in self.data_inputs
                    if isinstance(d, Data)
                ]
                if not documents:
                    raise TypeError("No valid Data inputs found")

        # ---- Split text ----
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=separator,
        )

        return splitter.split_documents(documents)

    def split_text(self) -> DataFrame:
        return DataFrame(self._docs_to_data(self.split_text_base()))