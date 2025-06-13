"""Module docstring."""

import json
import os
from pathlib import Path
from typing import Literal, Optional, cast
from pydantic import ValidationError
import polars as pl

from dotenv import load_dotenv

from ..llm_client.models import LLMClientEnum
from ..llm_client.LLM_client import LLMClient
from ..llm_client.openai_client import OpenAIClient, OpenAIConfig
from ..llm_client.azureopenai_client import AzureOpenAIClient, AzureOpenAIConfig
from ..llm_client.serialization import (
    OpenAISerializedState,
    AzureOpenAISerializedState,
)
from ..data.EvalData import EvalData, SampleEvalData
from ..data.serialization import DataMetadata
from .exceptions import (
    MissingConfigurationException,
    ClientAlreadyExistsException,
    ClientNotFoundException,
    DataAlreadyExistsException,
    DataFilenameExtensionMismatchException,
)
from .serialization import MetaEvaluatorState

# Error message constants
INVALID_JSON_STRUCTURE_MSG = "Invalid JSON structure in state file"
INVALID_JSON_MSG = "Invalid JSON in state file"
STATE_FILE_NOT_FOUND_MSG = "State file not found"

load_dotenv()


class MetaEvaluator:
    """This is a placeholder for the MetaEvaluator class."""

    def __init__(self):
        """This is a placeholder for the MetaEvaluator class."""
        self.client_registry: dict[LLMClientEnum, LLMClient] = {}
        self.data: Optional[EvalData] = None

    def add_openai(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        override_existing: bool = False,
    ):
        """Add an OpenAI client to the registry.

        Args:
            api_key: OpenAI API key. If None, will look for OPENAI_API_KEY in environment.
            default_model: Default model to use. If None, will look for OPENAI_DEFAULT_MODEL in environment.
            default_embedding_model: Default embedding model. If None, will look for OPENAI_DEFAULT_EMBEDDING_MODEL in environment.
            override_existing: Whether to override existing client. Defaults to False.

        Raises:
            MissingConfigurationException: If required parameters are missing from both arguments and environment.
            ClientAlreadyExistsException: If client already exists and override_existing is False.
        """
        # Check if client already exists
        if LLMClientEnum.OPENAI in self.client_registry and not override_existing:
            raise ClientAlreadyExistsException("OPENAI")

        # Get configuration values, fallback to environment variables
        final_api_key = api_key or os.getenv("OPENAI_API_KEY")
        final_default_model = default_model or os.getenv("OPENAI_DEFAULT_MODEL")
        final_default_embedding_model = default_embedding_model or os.getenv(
            "OPENAI_DEFAULT_EMBEDDING_MODEL"
        )

        # Validate required parameters
        if not final_api_key:
            raise MissingConfigurationException(
                "api_key (or OPENAI_API_KEY environment variable)"
            )
        if not final_default_model:
            raise MissingConfigurationException(
                "default_model (or OPENAI_DEFAULT_MODEL environment variable)"
            )
        if not final_default_embedding_model:
            raise MissingConfigurationException(
                "default_embedding_model (or OPENAI_DEFAULT_EMBEDDING_MODEL environment variable)"
            )

        # Create configuration and client
        config = OpenAIConfig(
            api_key=final_api_key,
            default_model=final_default_model,
            default_embedding_model=final_default_embedding_model,
        )
        client = OpenAIClient(config)

        # Add to registry
        self.client_registry[LLMClientEnum.OPENAI] = client

    def add_azure_openai(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        override_existing: bool = False,
    ):
        """Add an Azure OpenAI client to the registry.

        Args:
            api_key: Azure OpenAI API key. If None, will look for AZURE_OPENAI_API_KEY in environment.
            endpoint: Azure OpenAI endpoint. If None, will look for AZURE_OPENAI_ENDPOINT in environment.
            api_version: Azure OpenAI API version. If None, will look for AZURE_OPENAI_API_VERSION in environment.
            default_model: Default model to use. If None, will look for AZURE_OPENAI_DEFAULT_MODEL in environment.
            default_embedding_model: Default embedding model. If None, will look for AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL in environment.
            override_existing: Whether to override existing client. Defaults to False.

        Raises:
            MissingConfigurationException: If required parameters are missing from both arguments and environment.
            ClientAlreadyExistsException: If client already exists and override_existing is False.
        """
        # Check if client already exists
        if LLMClientEnum.AZURE_OPENAI in self.client_registry and not override_existing:
            raise ClientAlreadyExistsException("AZURE_OPENAI")

        # Get configuration values, fallback to environment variables
        final_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        final_endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        final_api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        final_default_model = default_model or os.getenv("AZURE_OPENAI_DEFAULT_MODEL")
        final_default_embedding_model = default_embedding_model or os.getenv(
            "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL"
        )

        # Validate required parameters
        if not final_api_key:
            raise MissingConfigurationException(
                "api_key (or AZURE_OPENAI_API_KEY environment variable)"
            )
        if not final_endpoint:
            raise MissingConfigurationException(
                "endpoint (or AZURE_OPENAI_ENDPOINT environment variable)"
            )
        if not final_api_version:
            raise MissingConfigurationException(
                "api_version (or AZURE_OPENAI_API_VERSION environment variable)"
            )
        if not final_default_model:
            raise MissingConfigurationException(
                "default_model (or AZURE_OPENAI_DEFAULT_MODEL environment variable)"
            )
        if not final_default_embedding_model:
            raise MissingConfigurationException(
                "default_embedding_model (or AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL environment variable)"
            )

        # Create configuration and client
        config = AzureOpenAIConfig(
            api_key=final_api_key,
            endpoint=final_endpoint,
            api_version=final_api_version,
            default_model=final_default_model,
            default_embedding_model=final_default_embedding_model,
        )
        client = AzureOpenAIClient(config)

        # Add to registry
        self.client_registry[LLMClientEnum.AZURE_OPENAI] = client

    def get_client(self, client_type: LLMClientEnum) -> LLMClient:
        """Get a client from the registry by type.

        Args:
            client_type: The LLM client enum type to retrieve.

        Returns:
            LLMClient: The requested LLM client instance.

        Raises:
            ClientNotFoundException: If the client type is not found in the registry.
        """
        if client_type not in self.client_registry:
            raise ClientNotFoundException(client_type.value)

        return self.client_registry[client_type]

    def add_data(self, eval_data: EvalData, overwrite: bool = False) -> None:
        """Add evaluation data to the evaluator.

        Args:
            eval_data: The EvalData object to add to the evaluator.
            overwrite: Whether to overwrite existing data. Defaults to False.

        Raises:
            DataAlreadyExistsException: If data already exists and overwrite is False.
        """
        if self.data is not None and not overwrite:
            raise DataAlreadyExistsException()

        self.data = eval_data

    def save_state(
        self,
        state_file: str,
        include_data: bool = True,
        data_format: Optional[Literal["json", "csv", "parquet"]] = None,
        data_filename: Optional[str] = None,
    ) -> None:
        """Save MetaEvaluator state to JSON file with optional data serialization.

        Args:
            state_file: Path to JSON file for state (must end with .json).
            include_data: Whether to serialize EvalData. Defaults to True.
            data_format: Format for data file when include_data=True.
                Must be specified if include_data=True.
            data_filename: Optional custom filename for data file. If None,
                auto-generates using pattern {base_name}_data.{format}.
                Must have extension matching data_format.

        Raises:
            ValueError: If state_file doesn't end with .json or if include_data=True
                but data_format is None.
            DataFilenameExtensionMismatchException: If data_filename extension
                doesn't match data_format.
        """
        # Validate state_file ends with .json
        if not state_file.endswith(".json"):
            raise ValueError("state_file must end with .json")

        # Validate data_format when include_data is True
        if include_data and data_format is None:
            raise ValueError("data_format must be specified when include_data=True")

        # Validate data_filename extension matches data_format
        if data_filename is not None and include_data and data_format is not None:
            expected_extension = data_format
            if not data_filename.endswith(f".{expected_extension}"):
                raise DataFilenameExtensionMismatchException(
                    data_filename, expected_extension, data_format
                )

        # Extract base_name and directory from state_file
        state_path = Path(state_file)
        base_name = state_path.stem
        directory = state_path.parent

        # Generate or use provided data_filename if include_data
        final_data_filename = None
        if include_data:
            if data_filename is not None:
                final_data_filename = data_filename
            else:
                final_data_filename = f"{base_name}_data.{data_format}"

        # Generate state
        state = self._serialize(include_data, data_format, final_data_filename)

        # Ensure directory exists
        directory.mkdir(parents=True, exist_ok=True)

        # Write state.json to disk
        with open(state_file, "w") as f:
            f.write(state.model_dump_json(indent=2))

        # Write data file if needed
        if include_data and self.data is not None:
            data_filepath = directory / cast(str, final_data_filename)
            self.data.write_data(
                str(data_filepath), cast(Literal["json", "csv", "parquet"], data_format)
            )

    def _serialize(
        self,
        include_data: bool,
        data_format: Optional[Literal["json", "csv", "parquet"]],
        data_filename: Optional[str],
    ) -> MetaEvaluatorState:
        """Create complete state object - no I/O operations.

        Args:
            include_data: Whether to include data serialization metadata.
            data_format: Format for data serialization.
            data_filename: Name of data file if applicable.

        Returns:
            Complete state object ready for JSON serialization.
        """
        return MetaEvaluatorState(
            client_registry=self._serialize_client_registry(),
            data=self._serialize_data(include_data, data_format, data_filename),
        )

    def _serialize_client_registry(self) -> dict[str, dict]:
        """Serialize all clients using individual client serializers.

        Returns:
            Dictionary mapping client type names to their serialized configurations.
        """
        serialized_clients = {}
        for client_enum, client in self.client_registry.items():
            client_data = self._serialize_single_client(client_enum, client)
            # ASSERT: api_key not in client_data
            assert "api_key" not in str(client_data), (
                f"API key found in serialized {client_enum.value} client"
            )
            serialized_clients[client_enum.value] = client_data
        return serialized_clients

    def _serialize_single_client(
        self, client_enum: LLMClientEnum, client: LLMClient
    ) -> dict:
        """Serialize a single client using its config's serialize method.

        Args:
            client_enum: The LLM client enum type.
            client: The LLM client instance.

        Returns:
            Serialized client configuration dictionary.
        """
        serialized_state = client.config.serialize()
        return serialized_state.model_dump(mode="json")

    def get_client_list(self) -> list[tuple[LLMClientEnum, LLMClient]]:
        """Get a list of client tuples (type, client).

        Returns:
            List of client tuples.
        """
        return list(self.client_registry.items())

    def _serialize_data(
        self,
        include_data: bool,
        data_format: Optional[Literal["json", "csv", "parquet"]],
        data_filename: Optional[str],
    ) -> Optional[DataMetadata]:
        """Serialize EvalData metadata.

        Args:
            include_data: Whether to include data serialization metadata.
            data_format: Format for data serialization.
            data_filename: Name of data file if applicable.

        Returns:
            Data metadata object or None if data should not be included.

        Note:
            If data exists but id_column is None, a TypeError is raised by the underlying serialize method.
        """
        if not include_data or self.data is None:
            return None
        return self.data.serialize(data_format, data_filename)

    @classmethod
    def load_state(
        cls,
        state_file: str,
        load_data: bool = True,
        openai_api_key: Optional[str] = None,
        azure_openai_api_key: Optional[str] = None,
    ) -> "MetaEvaluator":
        """Load MetaEvaluator state from JSON file with automatic data loading.

        The state file contains all information needed to reconstruct the MetaEvaluator,
        including the location and format of any associated data files. You only need
        to provide the path to the state file - data files are automatically located
        and loaded based on information stored in the state.

        Args:
            state_file: Path to JSON file containing the state.
            load_data: Whether to load the data file referenced in the state. Defaults to True.
                When True, automatically finds and loads the data file that was saved with this state.
                When False, only loads client configurations and skips data loading.
            openai_api_key: API key for OpenAI clients. If None, will look for OPENAI_API_KEY in environment.
            azure_openai_api_key: API key for Azure OpenAI clients. If None, will look for AZURE_OPENAI_API_KEY in environment.

        Returns:
            MetaEvaluator: A new MetaEvaluator instance loaded from the JSON state.

        Examples:
            # Load everything (clients + data)
            evaluator = MetaEvaluator.load_state("my_state.json")

            # Load only clients, skip data
            evaluator = MetaEvaluator.load_state("my_state.json", load_data=False)

            # Provide custom API keys
            evaluator = MetaEvaluator.load_state("my_state.json",
                                                   openai_api_key="custom-key")

        Raises:
            ValueError: If state_file doesn't end with .json or if the JSON structure is invalid.
        """
        # Validate state_file ends with .json
        if not state_file.endswith(".json"):
            raise ValueError("state_file must end with .json")

        # Load JSON state
        state = cls._load_json_state(state_file)

        # Create new MetaEvaluator instance
        evaluator = cls()

        # Reconstruct clients
        evaluator._reconstruct_clients(
            state.client_registry, openai_api_key, azure_openai_api_key
        )

        # Load data if requested and available
        if load_data and state.data is not None:
            evaluator._load_data_from_state(state.data, state_file)

        return evaluator

    @classmethod
    def _load_json_state(cls, state_file: str) -> MetaEvaluatorState:
        """Load and validate JSON state file.

        Args:
            state_file: Path to the JSON state file.

        Returns:
            MetaEvaluatorState: The loaded and validated state object.

        Raises:
            FileNotFoundError: If the state file doesn't exist.
            ValueError: If the JSON structure is invalid.
        """
        try:
            with open(state_file, "r") as f:
                return MetaEvaluatorState.model_validate_json(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"{STATE_FILE_NOT_FOUND_MSG}: {state_file}")
        except ValidationError as e:
            raise ValueError(f"{INVALID_JSON_STRUCTURE_MSG}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"{INVALID_JSON_MSG}: {e}")

    def _reconstruct_clients(
        self,
        client_registry_data: dict,
        openai_api_key: Optional[str],
        azure_openai_api_key: Optional[str],
    ) -> None:
        """Reconstruct all clients from serialized data.

        Args:
            client_registry_data: Dictionary of serialized client configurations.
            openai_api_key: API key for OpenAI clients.
            azure_openai_api_key: API key for Azure OpenAI clients.
        """
        for client_type_str, client_data in client_registry_data.items():
            client_enum = LLMClientEnum(client_type_str)
            self._reconstruct_single_client(
                client_enum, client_data, openai_api_key, azure_openai_api_key
            )

    def _reconstruct_single_client(
        self,
        client_enum: LLMClientEnum,
        client_data: dict,
        openai_api_key: Optional[str],
        azure_openai_api_key: Optional[str],
    ) -> None:
        """Reconstruct a single client from serialized data using config deserialization.

        Args:
            client_enum: The client type enum.
            client_data: Serialized client configuration.
            openai_api_key: API key for OpenAI clients.
            azure_openai_api_key: API key for Azure OpenAI clients.

        Raises:
            ValueError: If client type is not supported.
            MissingConfigurationException: If required API keys are missing.
        """
        match client_enum:
            case LLMClientEnum.OPENAI:
                if not openai_api_key:
                    raise MissingConfigurationException(
                        "api_key (or OPENAI_API_KEY environment variable)"
                    )
                state = OpenAISerializedState.model_validate(client_data)
                config = OpenAIConfig.deserialize(state, openai_api_key)
                client = OpenAIClient(config)
                self.client_registry[client_enum] = client
            case LLMClientEnum.AZURE_OPENAI:
                if not azure_openai_api_key:
                    raise MissingConfigurationException(
                        "api_key (or AZURE_OPENAI_API_KEY environment variable)"
                    )
                state = AzureOpenAISerializedState.model_validate(client_data)
                config = AzureOpenAIConfig.deserialize(state, azure_openai_api_key)
                client = AzureOpenAIClient(config)
                self.client_registry[client_enum] = client
            case _:
                raise ValueError(f"Unsupported client type: {client_enum}")

    def _load_data_from_state(
        self, data_metadata: DataMetadata, state_file: str
    ) -> None:
        """Load data from external file based on metadata.

        Args:
            data_metadata: DataMetadata object containing data file metadata.
            state_file: Path to the original state file (for resolving relative paths).
        """
        # Resolve data file path relative to state file
        state_path = Path(state_file)
        data_filepath = state_path.parent / data_metadata.data_file

        # Load the data based on format
        df = self._load_dataframe_from_file(
            str(data_filepath), data_metadata.data_format
        )

        # Create appropriate EvalData object
        eval_data = self._create_eval_data_from_metadata(data_metadata, df)

        # Add to evaluator
        self.add_data(eval_data, overwrite=True)

    def _load_dataframe_from_file(self, filepath: str, data_format: str):
        """Load DataFrame from file in specified format.

        Args:
            filepath: Path to the data file.
            data_format: Format of the data file.

        Returns:
            Loaded DataFrame (polars).

        Raises:
            FileNotFoundError: If the data file doesn't exist.
            ValueError: If the data format is not supported.
        """
        try:
            match data_format:
                case "parquet":
                    return pl.read_parquet(filepath)
                case "csv":
                    return pl.read_csv(filepath)
                case "json":
                    with open(filepath, "r") as f:
                        data_dict = json.load(f)
                    return pl.DataFrame(data_dict)
                case _:
                    raise ValueError(f"Unsupported data format: {data_format}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")

    def _create_eval_data_from_metadata(self, data_metadata: DataMetadata, df):
        """Create EvalData or SampleEvalData object from metadata and DataFrame.

        Args:
            data_metadata: DataMetadata object containing data metadata.
            df: The loaded DataFrame.

        Returns:
            EvalData or SampleEvalData object.
        """
        if data_metadata.type == "SampleEvalData":
            return SampleEvalData(
                data=df,
                name=data_metadata.name,
                id_column=data_metadata.id_column,
                sample_name=cast(str, data_metadata.sample_name),
                stratification_columns=cast(list, data_metadata.stratification_columns),
                sample_percentage=cast(float, data_metadata.sample_percentage),
                seed=cast(int, data_metadata.seed),
                sampling_method=cast(str, data_metadata.sampling_method),
            )
        else:
            return EvalData(
                data=df,
                name=data_metadata.name,
                id_column=data_metadata.id_column,
            )
