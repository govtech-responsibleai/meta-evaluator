"""Base functionality for MetaEvaluator including initialization, paths, and core data/task management."""

import json
import logging
import os
from pathlib import Path
from typing import Literal, Optional, cast

from pydantic import ValidationError

from ..annotator.launcher import StreamlitLauncher
from ..common.error_constants import (
    INVALID_JSON_MSG,
    INVALID_JSON_STRUCTURE_MSG,
    STATE_FILE_NOT_FOUND_MSG,
)
from ..data import EvalData, SampleEvalData
from ..data.serialization import DataMetadata
from ..eval_task import EvalTask
from ..eval_task.serialization import EvalTaskState
from ..llm_client.async_azureopenai_client import (
    AsyncAzureOpenAIClient,
    AsyncAzureOpenAIConfig,
)
from ..llm_client.async_openai_client import AsyncOpenAIClient, AsyncOpenAIConfig
from ..llm_client.azureopenai_client import AzureOpenAIClient, AzureOpenAIConfig
from ..llm_client.enums import AsyncLLMClientEnum, LLMClientEnum
from ..llm_client.openai_client import OpenAIClient, OpenAIConfig
from ..llm_client.serialization import (
    AzureOpenAISerializedState,
    OpenAISerializedState,
)
from .clients import ClientsMixin
from .exceptions import (
    ClientNotFoundError,
    DataAlreadyExistsError,
    DataFormatError,
    EvalDataNotFoundError,
    EvalTaskAlreadyExistsError,
    EvalTaskNotFoundError,
    InvalidFileError,
    MissingConfigurationError,
)
from .judge import JudgesMixin
from .scoring import ScoringMixin
from .serialization import MetaEvaluatorState

_OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
_AZURE_OPENAI_API_KEY_ENV_VAR = "AZURE_OPENAI_API_KEY"


class Paths:
    """Simple namespace for managing project directory structure."""

    def __init__(self, project_dir: str | Path):
        """Initialize paths from project directory.

        Args:
            project_dir: Root project directory path.
        """
        self.project = Path(project_dir)
        self.data = self.project / "data"
        self.results = self.project / "results"
        self.annotations = self.project / "annotations"
        self.scores = self.project / "scores"

    def ensure_directories(self) -> None:
        """Create all project directories."""
        for path in [
            self.project,
            self.data,
            self.results,
            self.annotations,
            self.scores,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class MetaEvaluator(ClientsMixin, JudgesMixin, ScoringMixin):
    """Main class for managing evaluation workflows with LLM clients, data, and evaluation tasks.

    The MetaEvaluator provides a unified interface for:
    - Managing the evaluation dataset
    - Managing the evaluation task
    - Managing multiple LLM client configurations (OpenAI, Azure OpenAI)
    - Managing judges for evaluation
    - Initialising the annotator interface
    - Loading judge and human annotation results
    - Comparing judge and human results using various scoring metrics
    - Serializing and deserializing complete evaluation states
    - Supporting both structured and XML-based evaluation methods
    """

    def __init__(self, project_dir: Optional[str] = None):
        """Initialize a new MetaEvaluator instance.

        Creates an empty evaluator with no clients, data, or evaluation tasks configured.

        Args:
            project_dir: Directory for organizing all evaluation files. If provided, all file operations
                will be organized within this directory structure. If None, creates 'my_project' directory in current working directory.
        """
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.data: Optional[EvalData] = None
        self.eval_task: Optional[EvalTask] = None

        # If no project directory is provided, create a default directory in the current working directory
        if project_dir is None:
            project_dir = "./my_project"

        # Initialize project paths and create directory structure
        self.paths = Paths(project_dir)
        self.paths.ensure_directories()

    @property
    def project_dir(self) -> Path:
        """Get the project directory path.

        Returns:
            Path: The project directory path.
        """
        return self.paths.project

    # ===== DATA AND TASK MANAGEMENT METHODS =====

    def add_data(self, eval_data: EvalData, overwrite: bool = False) -> None:
        """Add evaluation data to the evaluator.

        Args:
            eval_data: The EvalData object to add to the evaluator.
            overwrite: Whether to overwrite existing data. Defaults to False.

        Raises:
            DataAlreadyExistsError: If data already exists and overwrite is False.
        """
        if self.data is not None and not overwrite:
            raise DataAlreadyExistsError()

        self.data = eval_data
        self.logger.info(
            f"Added evaluation data '{eval_data.name}' with {len(eval_data.data)} rows"
        )

    def add_eval_task(self, eval_task: EvalTask, overwrite: bool = False) -> None:
        """Add evaluation task to the evaluator.

        Args:
            eval_task: The EvalTask object to add to the evaluator.
            overwrite: Whether to overwrite existing evaluation task. Defaults to False.

        Raises:
            EvalTaskAlreadyExistsError: If evaluation task already exists and overwrite is False.
        """
        if self.eval_task is not None and not overwrite:
            raise EvalTaskAlreadyExistsError()

        self.eval_task = eval_task
        task_names = list(eval_task.task_schemas.keys())
        self.logger.info(
            f"Added evaluation task with {len(task_names)} task(s): {', '.join(task_names)}"
        )

    # ===== SERIALIZATION METHODS =====

    def save_state(
        self,
        state_filename: str = "main_state.json",
        include_task: bool = True,
        include_data: bool = True,
        data_format: Optional[Literal["json", "csv", "parquet"]] = None,
        data_filename: Optional[str] = None,
    ) -> None:
        """Save MetaEvaluator state to JSON file with optional data serialization.

        The state includes all configuration needed to reconstruct the evaluator:
        - LLM client configurations (without API keys for security)
        - Evaluation task configuration (task schemas, input/output columns, answering method)
        - Data metadata and optional data file serialization

        Files are saved within the project directory structure:
        - State file: project_dir/{state_filename}
        - Data file: project_dir/data/{data_filename}

        Args:
            state_filename: Filename for state JSON file. Defaults to 'main_state.json'.
            include_task: Whether to serialize EvalTask. Defaults to True.
            include_data: Whether to serialize EvalData. Defaults to True.
            data_format: Format for data file when include_data=True.
                Must be specified if include_data=True.
            data_filename: Optional custom filename for data file. If None,
                auto-generates using pattern {base_name}_data.{format}.
                Must have extension matching data_format.

        Raises:
            InvalidFileError: If state_filename doesn't end with .json
            DataFormatError: If include_data=True but data_format is None, or
                if data_filename extension doesn't match data_format.
        """
        # Validate state_filename ends with .json
        if not state_filename.endswith(".json"):
            raise InvalidFileError("state_filename must end with .json")

        # Validate data_format when include_data is True
        if include_data and data_format is None:
            raise DataFormatError(
                "data_format must be specified when include_data=True"
            )

        # Validate data_filename extension matches data_format
        if data_filename is not None and include_data and data_format is not None:
            expected_extension = data_format
            if not data_filename.endswith(f".{expected_extension}"):
                raise DataFormatError(
                    f"Data filename '{data_filename}' must have extension '.{expected_extension}' "
                    f"when data_format is '{data_format}'"
                )

        # Extract base_name from state_filename
        base_name = Path(state_filename).stem

        # Generate or use provided data_filename if include_data
        final_data_filename = None
        if include_data:
            if data_filename is not None:
                final_data_filename = data_filename
            else:
                final_data_filename = f"{base_name}_data.{data_format}"

        # Generate state
        state = self._serialize(
            include_task, include_data, data_format, final_data_filename
        )

        # Ensure project directory and subdirectories exist
        self.paths.ensure_directories()

        # Resolve state file path within project directory
        state_file_path = self.paths.project / state_filename

        # Ensure parent directories exist for state file (for nested paths)
        state_file_path.parent.mkdir(parents=True, exist_ok=True)

        data_filepath = (
            self.paths.data / final_data_filename if final_data_filename else None
        )

        # Write state.json to disk
        with open(state_file_path, "w") as f:
            f.write(state.model_dump_json(indent=2))

        # Write data file if needed
        if include_data and self.data is not None and data_filepath is not None:
            self.data.write_data(
                filepath=str(data_filepath),
                data_format=cast(Literal["json", "csv", "parquet"], data_format),
            )

    def _serialize(
        self,
        include_task: bool,
        include_data: bool,
        data_format: Optional[Literal["json", "csv", "parquet"]],
        data_filename: Optional[str],
    ) -> MetaEvaluatorState:
        """Create complete state object - no I/O operations.

        Args:
            include_task: Whether to include EvalTask serialization.
            include_data: Whether to include data serialization metadata.
            data_format: Format for data serialization.
            data_filename: Name of data file relative to project_dir/data.

        Returns:
            Complete state object ready for JSON serialization.
        """
        client_registry, async_client_registry = self._serialize_all_clients()
        return MetaEvaluatorState(
            client_registry=client_registry,
            async_client_registry=async_client_registry,
            data=self._serialize_data(include_data, data_format, data_filename),
            eval_task=self._serialize_eval_task(include_task),
        )

    def _serialize_all_clients(self) -> tuple[dict[str, dict], dict[str, dict]]:
        """Serialize both sync and async clients using individual client serializers.

        Returns:
            Tuple containing (sync_clients_dict, async_clients_dict) with client type names
            mapped to their serialized configurations.
        """
        # Serialize sync clients
        serialized_clients = {}
        for client_enum, client in self.client_registry.items():
            client_data = self._serialize_single_client(client_enum, client)
            # ASSERT: api_key not in client_data
            assert "api_key" not in str(client_data), (
                f"API key found in serialized {client_enum.value} client"
            )
            serialized_clients[client_enum.value] = client_data

        # Serialize async clients
        serialized_async_clients = {}
        for client_enum, client in self.async_client_registry.items():
            client_data = self._serialize_single_client(client_enum, client)
            # ASSERT: api_key not in client_data
            assert "api_key" not in str(client_data), (
                f"API key found in serialized async {client_enum.value} client"
            )
            serialized_async_clients[client_enum.value] = client_data

        return serialized_clients, serialized_async_clients

    def _serialize_single_client(self, client_enum, client) -> dict:
        """Serialize a single client (sync or async) using its config's serialize method.

        Args:
            client_enum: The LLM client enum type (sync or async).
            client: The LLM client instance (sync or async).

        Returns:
            Serialized client configuration dictionary.
        """
        serialized_state = client.config.serialize()
        return serialized_state.model_dump(mode="json")

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
            data_filename: Name of data file relative to project_dir/data.

        Returns:
            Serialized DataMetaData dictionary.

        Note:
            If data exists but id_column is None, a TypeError is raised by the underlying serialize method.
        """
        if not include_data or self.data is None:
            return None
        serialized_metadata = self.data.serialize_metadata(
            data_format=data_format,
            data_filename=data_filename,
        )
        return serialized_metadata

    def _serialize_eval_task(
        self,
        include_task: bool,
    ) -> Optional[EvalTaskState]:
        """Serialize EvalTask.

        Returns:
            Serialized EvalTask dictionary.
        """
        if not include_task or self.eval_task is None:
            return None
        serialized_state = self.eval_task.serialize()

        return serialized_state

    # ===== DESERIALIZATION METHODS =====

    @classmethod
    def load_state(
        cls,
        project_dir: str,
        state_filename: str = "main_state.json",
        load_data: bool = True,
        load_task: bool = True,
        openai_api_key: Optional[str] = None,
        azure_openai_api_key: Optional[str] = None,
    ) -> "MetaEvaluator":
        """Load MetaEvaluator state from JSON file with automatic data loading.

        The state file contains all information needed to reconstruct the MetaEvaluator,
        including the location and format of any associated data files. You only need
        to provide the path to the state file - data files are automatically located
        and loaded based on information stored in the state.

        The loaded evaluator will include:
        - LLM client configurations (requires API keys to be provided or in environment)
        - Evaluation task configuration (task schemas, input/output columns, answering method)
        - Data files (if load_data=True and data was included in the saved state)

        Args:
            project_dir: Project directory containing the evaluation files.
            state_filename: Filename of the state JSON file. Defaults to 'main_state.json'.
            load_data: Whether to load the data file referenced in the state. Defaults to True.
                When True, automatically finds and loads the data file that was saved with this state.
                When False, only loads client configurations and skips data loading.
            load_task: Whether to load the evaluation task configuration. Defaults to True.
                When True, loads the evaluation task with all its configuration including
                task schemas, input/output columns, and answering method.
                When False, skips evaluation task loading.
            openai_api_key: API key for OpenAI clients. If None, will look for OPENAI_API_KEY in environment.
            azure_openai_api_key: API key for Azure OpenAI clients. If None, will look for AZURE_OPENAI_API_KEY in environment.

        Returns:
            MetaEvaluator: A new MetaEvaluator instance loaded from the JSON state.

        Raises:
            InvalidFileError: If state_file doesn't end with .json or if the JSON structure is invalid.
        """
        # Validate state_filename ends with .json
        if not state_filename.endswith(".json"):
            raise InvalidFileError("state_filename must end with .json")

        # Resolve state file path
        state_file_path = Path(project_dir) / state_filename

        # Load JSON MetaEvaluatorState
        state = cls._load_json_state(str(state_file_path))

        # Deserialize state to MetaEvaluator instance
        return cls._deserialize(
            state=state,
            project_dir=project_dir,
            load_data=load_data,
            load_task=load_task,
            openai_api_key=openai_api_key,
            azure_openai_api_key=azure_openai_api_key,
        )

    @classmethod
    def _load_json_state(cls, state_file: str) -> MetaEvaluatorState:
        """Load and validate JSON state file.

        Args:
            state_file: Path to the JSON state file.

        Returns:
            MetaEvaluatorState: The loaded and validated state object.

        Raises:
            InvalidFileError: If the state file doesn't exist or the JSON structure is invalid.
        """
        try:
            with open(state_file, "r") as f:
                return MetaEvaluatorState.model_validate_json(f.read())
        except FileNotFoundError as e:
            raise InvalidFileError(f"{STATE_FILE_NOT_FOUND_MSG}: {state_file}", str(e))
        except ValidationError as e:
            raise InvalidFileError(INVALID_JSON_STRUCTURE_MSG, str(e))
        except json.JSONDecodeError as e:
            raise InvalidFileError(INVALID_JSON_MSG, str(e))

    @classmethod
    def _deserialize(
        cls,
        state: MetaEvaluatorState,
        project_dir: str,
        load_data: bool = True,
        load_task: bool = True,
        openai_api_key: Optional[str] = None,
        azure_openai_api_key: Optional[str] = None,
    ) -> "MetaEvaluator":
        """Deserialize MetaEvaluatorState to MetaEvaluator instance.

        Args:
            state: MetaEvaluatorState object containing serialized state.
            project_dir: Project directory containing the evaluation files.
            load_data: Whether to load the data file referenced in the state.
            load_task: Whether to load the evaluation task configuration.
            openai_api_key: API key for OpenAI clients.
            azure_openai_api_key: API key for Azure OpenAI clients.

        Returns:
            MetaEvaluator: A new MetaEvaluator instance.
        """
        # Create new MetaEvaluator instance with project_dir
        evaluator = cls(project_dir)

        # Use environment variables if API keys are not provided
        openai_api_key = openai_api_key or os.getenv(_OPENAI_API_KEY_ENV_VAR)
        azure_openai_api_key = azure_openai_api_key or os.getenv(
            _AZURE_OPENAI_API_KEY_ENV_VAR
        )

        # Reconstruct all clients (both sync and async)
        evaluator._reconstruct_all_clients(
            state.client_registry,
            state.async_client_registry,
            openai_api_key,
            azure_openai_api_key,
        )

        # Load data if requested and available
        if load_data and state.data is not None:
            evaluator._reconstruct_data(state.data)

        # Load task if requested and available
        if load_task and state.eval_task is not None:
            evaluator._reconstruct_task(state.eval_task)

        return evaluator

    def _reconstruct_all_clients(
        self,
        client_registry_data: dict,
        async_client_registry_data: dict,
        openai_api_key: Optional[str],
        azure_openai_api_key: Optional[str],
    ) -> None:
        """Reconstruct all clients (both sync and async) from serialized data.

        Args:
            client_registry_data: Dictionary of serialized sync client configurations.
            async_client_registry_data: Dictionary of serialized async client configurations.
            openai_api_key: API key for OpenAI clients.
            azure_openai_api_key: API key for Azure OpenAI clients.
        """
        # Reconstruct sync clients
        for client_type_str, client_data in client_registry_data.items():
            client_enum = LLMClientEnum(client_type_str)
            self._reconstruct_single_sync_client(
                client_enum, client_data, openai_api_key, azure_openai_api_key
            )

        # Reconstruct async clients
        for client_type_str, client_data in async_client_registry_data.items():
            client_enum = AsyncLLMClientEnum(client_type_str)
            self._reconstruct_single_async_client(
                client_enum, client_data, openai_api_key, azure_openai_api_key
            )

    def _reconstruct_single_sync_client(
        self,
        client_enum: LLMClientEnum,
        client_data: dict,
        openai_api_key: Optional[str],
        azure_openai_api_key: Optional[str],
    ) -> None:
        """Reconstruct a single sync client from serialized data using config deserialization.

        Args:
            client_enum: The sync client type enum.
            client_data: Serialized sync client configuration.
            openai_api_key: API key for OpenAI clients.
            azure_openai_api_key: API key for Azure OpenAI clients.

        Raises:
            ClientNotFoundError: If client type is not supported.
            MissingConfigurationError: If required API keys are missing.
        """
        match client_enum:
            case LLMClientEnum.OPENAI:
                if not openai_api_key:
                    raise MissingConfigurationError(
                        "api_key (or OPENAI_API_KEY environment variable)"
                    )
                state = OpenAISerializedState.model_validate(client_data)
                config = OpenAIConfig.deserialize(state, openai_api_key)
                client = OpenAIClient(config)
                self.client_registry[client_enum] = client
            case LLMClientEnum.AZURE_OPENAI:
                if not azure_openai_api_key:
                    raise MissingConfigurationError(
                        "api_key (or AZURE_OPENAI_API_KEY environment variable)"
                    )
                state = AzureOpenAISerializedState.model_validate(client_data)
                config = AzureOpenAIConfig.deserialize(state, azure_openai_api_key)
                client = AzureOpenAIClient(config)
                self.client_registry[client_enum] = client
            case _:
                raise ClientNotFoundError(client_enum.value)

    def _reconstruct_single_async_client(
        self,
        client_enum: AsyncLLMClientEnum,
        client_data: dict,
        openai_api_key: Optional[str],
        azure_openai_api_key: Optional[str],
    ) -> None:
        """Reconstruct a single async client from serialized data using config deserialization.

        Args:
            client_enum: The async client type enum.
            client_data: Serialized async client configuration.
            openai_api_key: API key for OpenAI clients.
            azure_openai_api_key: API key for Azure OpenAI clients.

        Raises:
            ClientNotFoundError: If client type is not supported.
            MissingConfigurationError: If required API keys are missing.
        """
        match client_enum:
            case AsyncLLMClientEnum.OPENAI:
                if not openai_api_key:
                    raise MissingConfigurationError(
                        "api_key (or OPENAI_API_KEY environment variable)"
                    )
                state = OpenAISerializedState.model_validate(client_data)
                config = AsyncOpenAIConfig.deserialize(state, openai_api_key)
                client = AsyncOpenAIClient(config)
                self.async_client_registry[client_enum] = client
            case AsyncLLMClientEnum.AZURE_OPENAI:
                if not azure_openai_api_key:
                    raise MissingConfigurationError(
                        "api_key (or AZURE_OPENAI_API_KEY environment variable)"
                    )
                state = AzureOpenAISerializedState.model_validate(client_data)
                config = AsyncAzureOpenAIConfig.deserialize(state, azure_openai_api_key)
                client = AsyncAzureOpenAIClient(config)
                self.async_client_registry[client_enum] = client
            case _:
                raise ClientNotFoundError(client_enum.value)

    def _reconstruct_task(self, state: EvalTaskState) -> None:
        """Reconstruct the evaluation task from serialized data.

        Args:
            state: EvalTaskState object containing serialized evaluation task.
        """
        self.eval_task = EvalTask.deserialize(state)

    def _reconstruct_data(self, metadata: DataMetadata) -> None:
        """Reconstruct the data from the data file metadata.

        Args:
            metadata: DataMetadata object containing data file metadata.
        """
        # Resolve data file path within project directory structure
        data_filepath = self.paths.data / metadata.data_file

        # Load the data based on format
        df = EvalData.load_data(
            filepath=str(data_filepath), data_format=metadata.data_format
        )

        # Create appropriate EvalData object
        if metadata.type == "SampleEvalData":
            eval_data = SampleEvalData.deserialize(data=df, metadata=metadata)
        else:
            eval_data = EvalData.deserialize(data=df, metadata=metadata)

        # Add to evaluator
        self.add_data(eval_data, overwrite=True)

    def launch_annotator(
        self,
        port: Optional[int] = None,
        use_ngrok: bool = False,
        traffic_policy_file: Optional[str] = None,
    ) -> None:
        """Launch the Streamlit annotator interface.

        This method launches the Streamlit annotation interface using the data and task
        that have been added to the MetaEvaluator. The annotations will be saved to
        the project's annotations directory.

        Args:
            port: Optional port number for Streamlit server. If None, uses default Streamlit port.
            use_ngrok: Whether to use ngrok to expose the Streamlit interface to the internet. Defaults to False.
            traffic_policy_file: Optional path to an ngrok traffic policy file for advanced
                configuration. See https://ngrok.com/docs/traffic-policy/ for details.
                Only used when use_ngrok=True.

        Raises:
            EvalTaskNotFoundError: If the evaluation task is not set.
            EvalDataNotFoundError: If the evaluation data is not set.
        """
        # Validate prerequisites
        if self.eval_task is None:
            raise EvalTaskNotFoundError(
                "eval_task must be set before launching annotator"
            )

        if self.data is None:
            raise EvalDataNotFoundError("data must be set before launching annotator")

        # Create launcher with data and task from MetaEvaluator
        launcher = StreamlitLauncher(
            eval_data=self.data,
            eval_task=self.eval_task,
            annotations_dir=str(self.paths.annotations),
            port=port,
        )

        # Launch the interface
        launcher.launch(use_ngrok=use_ngrok, traffic_policy_file=traffic_policy_file)
