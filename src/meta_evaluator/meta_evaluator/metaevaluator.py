"""Main class for MetaEvaluator."""

import json
import os
import yaml
from pathlib import Path
from typing import Literal, Optional, cast
from pydantic import ValidationError, BaseModel

from dotenv import load_dotenv

from ..llm_client.models import LLMClientEnum
from ..llm_client.LLM_client import LLMClient
from ..llm_client.openai_client import OpenAIClient, OpenAIConfig
from ..llm_client.azureopenai_client import AzureOpenAIClient, AzureOpenAIConfig
from ..llm_client.serialization import (
    OpenAISerializedState,
    AzureOpenAISerializedState,
)
from ..data import EvalData, SampleEvalData
from ..data.serialization import DataMetadata
from ..eval_task import EvalTask
from ..eval_task.serialization import EvalTaskState
from ..judge import Judge
from ..common.models import Prompt
from .exceptions import (
    MissingConfigurationException,
    ClientAlreadyExistsException,
    ClientNotFoundException,
    DataAlreadyExistsException,
    DataFilenameExtensionMismatchException,
    EvalTaskAlreadyExistsException,
    JudgeAlreadyExistsException,
    JudgeNotFoundException,
    InvalidYAMLStructureException,
    PromptFileNotFoundException,
)
from .serialization import MetaEvaluatorState

# Error message constants
INVALID_JSON_STRUCTURE_MSG = "Invalid JSON structure in state file"
INVALID_JSON_MSG = "Invalid JSON in state file"
STATE_FILE_NOT_FOUND_MSG = "State file not found"

_OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
_OPENAI_DEFAULT_MODEL_ENV_VAR = "OPENAI_DEFAULT_MODEL"
_OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR = "OPENAI_DEFAULT_EMBEDDING_MODEL"

_AZURE_OPENAI_API_KEY_ENV_VAR = "AZURE_OPENAI_API_KEY"
_AZURE_OPENAI_ENDPOINT_ENV_VAR = "AZURE_OPENAI_ENDPOINT"
_AZURE_OPENAI_API_VERSION_ENV_VAR = "AZURE_OPENAI_API_VERSION"
_AZURE_OPENAI_DEFAULT_MODEL_ENV_VAR = "AZURE_OPENAI_DEFAULT_MODEL"
_AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR = "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL"

load_dotenv()


class JudgeConfig(BaseModel):
    """Pydantic model for validating a single judge configuration from YAML."""

    id: str
    llm_client: str
    model: str
    prompt_file: str


class JudgeConfigList(BaseModel):
    """Pydantic model for validating the entire judges YAML configuration."""

    judges: list[JudgeConfig]


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

    def ensure_directories(self) -> None:
        """Create all project directories."""
        for path in [
            self.project,
            self.data,
            self.results,
            self.annotations,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class MetaEvaluator:
    """Main class for managing evaluation workflows with LLM clients, data, and evaluation tasks.

    The MetaEvaluator provides a unified interface for:
    - Managing multiple LLM client configurations (OpenAI, Azure OpenAI)
    - Loading and managing evaluation datasets
    - Configuring evaluation tasks with support for both predefined outcomes and free form text outputs
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
        self.client_registry: dict[LLMClientEnum, LLMClient] = {}
        self.data: Optional[EvalData] = None
        self.eval_task: Optional[EvalTask] = None
        self.judge_registry: dict[str, Judge] = {}

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

    # ===== CLIENT MANAGEMENT METHODS =====

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
        final_api_key = api_key or os.getenv(_OPENAI_API_KEY_ENV_VAR)
        final_default_model = default_model or os.getenv(_OPENAI_DEFAULT_MODEL_ENV_VAR)
        final_default_embedding_model = default_embedding_model or os.getenv(
            _OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR
        )

        # Validate required parameters
        if not final_api_key:
            raise MissingConfigurationException(
                f"api_key (or {_OPENAI_API_KEY_ENV_VAR} environment variable)"
            )
        if not final_default_model:
            raise MissingConfigurationException(
                f"default_model (or {_OPENAI_DEFAULT_MODEL_ENV_VAR} environment variable)"
            )
        if not final_default_embedding_model:
            raise MissingConfigurationException(
                f"default_embedding_model (or {_OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR} environment variable)"
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
        final_api_key = api_key or os.getenv(_AZURE_OPENAI_API_KEY_ENV_VAR)
        final_endpoint = endpoint or os.getenv(_AZURE_OPENAI_ENDPOINT_ENV_VAR)
        final_api_version = api_version or os.getenv(_AZURE_OPENAI_API_VERSION_ENV_VAR)
        final_default_model = default_model or os.getenv(
            _AZURE_OPENAI_DEFAULT_MODEL_ENV_VAR
        )
        final_default_embedding_model = default_embedding_model or os.getenv(
            _AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR
        )

        # Validate required parameters
        if not final_api_key:
            raise MissingConfigurationException(
                f"api_key (or {_AZURE_OPENAI_API_KEY_ENV_VAR} environment variable)"
            )
        if not final_endpoint:
            raise MissingConfigurationException(
                f"endpoint (or {_AZURE_OPENAI_ENDPOINT_ENV_VAR} environment variable)"
            )
        if not final_api_version:
            raise MissingConfigurationException(
                f"api_version (or {_AZURE_OPENAI_API_VERSION_ENV_VAR} environment variable)"
            )
        if not final_default_model:
            raise MissingConfigurationException(
                f"default_model (or {_AZURE_OPENAI_DEFAULT_MODEL_ENV_VAR} environment variable)"
            )
        if not final_default_embedding_model:
            raise MissingConfigurationException(
                f"default_embedding_model (or {_AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL_ENV_VAR} environment variable)"
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

    def get_client_list(self) -> list[tuple[LLMClientEnum, LLMClient]]:
        """Get a list of client tuples (type, client).

        Returns:
            List of client tuples.
        """
        return list(self.client_registry.items())

    # ===== DATA AND TASK MANAGEMENT METHODS =====

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

    def add_eval_task(self, eval_task: EvalTask, overwrite: bool = False) -> None:
        """Add evaluation task to the evaluator.

        Args:
            eval_task: The EvalTask object to add to the evaluator.
            overwrite: Whether to overwrite existing evaluation task. Defaults to False.

        Raises:
            EvalTaskAlreadyExistsException: If evaluation task already exists and overwrite is False.
        """
        if self.eval_task is not None and not overwrite:
            raise EvalTaskAlreadyExistsException()

        self.eval_task = eval_task

    # ===== JUDGE MANAGEMENT METHODS =====

    def add_judge(
        self,
        judge_id: str,
        llm_client_enum: LLMClientEnum,
        model: str,
        prompt: Prompt,
        override_existing: bool = False,
    ) -> None:
        """Add a judge to the evaluator programmatically.

        Args:
            judge_id: Unique identifier for the judge.
            llm_client_enum: The LLM client enum to use.
            model: The model name to use.
            prompt: The Prompt object containing the evaluation instructions.
            override_existing: Whether to override existing judge. Defaults to False.

        Raises:
            JudgeAlreadyExistsException: If judge already exists and override_existing is False.
            ValueError: If eval_task is not set.
        """
        if self.eval_task is None:
            raise ValueError("eval_task must be set before adding judges")

        if judge_id in self.judge_registry and not override_existing:
            raise JudgeAlreadyExistsException(judge_id)

        judge = Judge(
            id=judge_id,
            eval_task=self.eval_task,
            llm_client_enum=llm_client_enum,
            model=model,
            prompt=prompt,
        )

        self.judge_registry[judge_id] = judge

    def get_judge(self, judge_id: str) -> Judge:
        """Get a judge from the registry by ID.

        Args:
            judge_id: The judge ID to retrieve.

        Returns:
            Judge: The requested Judge instance.

        Raises:
            JudgeNotFoundException: If the judge ID is not found in the registry.
        """
        if judge_id not in self.judge_registry:
            raise JudgeNotFoundException(judge_id)

        return self.judge_registry[judge_id]

    def get_judge_list(self) -> list[tuple[str, Judge]]:
        """Get a list of judge tuples (id, judge).

        Returns:
            List of judge tuples.
        """
        return list(self.judge_registry.items())

    def load_judges_from_yaml(
        self,
        yaml_file: str,
        override_existing: bool = False,
    ) -> None:
        """Load judges from a YAML configuration file.

        The YAML file should have the following structure:
        ```yaml
        judges:
          - id: judge_id_1
            llm_client: openai
            model: gpt-4
            prompt_file: /absolute/path/to/toxicity_prompt.md
          - id: judge_id_2
            llm_client: azure_openai
            model: gpt-4
            prompt_file: ./relative/path/to/relevance_prompt.txt
        ```

        Args:
            yaml_file: Absolute or relative path to the YAML configuration file.
            override_existing: Whether to override existing judges. Defaults to False.

        Raises:
            FileNotFoundError: If the YAML file is not found.
            InvalidYAMLStructureException: If the YAML structure is invalid.
            ValueError: If eval_task is not set or if llm_client value is invalid.
        """
        if self.eval_task is None:
            raise ValueError("eval_task must be set before loading judges from YAML")

        # Resolve YAML file path (can be absolute or relative)
        yaml_path = Path(yaml_file)

        # Load and validate YAML
        try:
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        except yaml.YAMLError as e:
            raise InvalidYAMLStructureException(f"Invalid YAML syntax: {e}")

        # Validate YAML structure
        try:
            judges_config = JudgeConfigList.model_validate(yaml_data)
        except ValidationError as e:
            raise InvalidYAMLStructureException(f"YAML validation failed: {e}")

        # Load each judge
        for judge_config in judges_config.judges:
            self._load_single_judge_from_config(
                judge_config, override_existing, yaml_path
            )

    def _load_single_judge_from_config(
        self,
        judge_config: JudgeConfig,
        override_existing: bool,
        yaml_path: Path,
    ) -> None:
        """Load a single judge from YAML configuration.

        Args:
            judge_config: The judge configuration from YAML.
            override_existing: Whether to override existing judges.
            yaml_path: Path to the YAML file (for resolving relative prompt paths).

        Raises:
            ValueError: If llm_client value is invalid.
        """
        # Parse LLM client enum
        try:
            llm_client_enum = LLMClientEnum(judge_config.llm_client)
        except ValueError:
            valid_clients = [e.value for e in LLMClientEnum]
            raise ValueError(
                f"Invalid llm_client '{judge_config.llm_client}'. "
                f"Valid options: {valid_clients}"
            )

        # Load prompt file from absolute or relative path
        prompt = self._load_prompt_from_file(judge_config.prompt_file, yaml_path)

        # Use add_judge method to avoid duplication
        self.add_judge(
            judge_id=judge_config.id,
            llm_client_enum=llm_client_enum,
            model=judge_config.model,
            prompt=prompt,
            override_existing=override_existing,
        )

    def _load_prompt_from_file(
        self, prompt_file: str, yaml_path: Optional[Path] = None
    ) -> Prompt:
        """Load prompt content from file using absolute or relative path.

        Args:
            prompt_file: Absolute or relative path to the prompt file.
            yaml_path: Optional path to YAML file for resolving relative prompt paths.

        Returns:
            Prompt: The loaded Prompt object.

        Raises:
            PromptFileNotFoundException: If prompt file cannot be found.
        """
        # Resolve prompt file path
        prompt_path = Path(prompt_file)

        if prompt_path.is_absolute():
            # Use absolute path as-is
            resolved_prompt_path = prompt_path
        else:
            # Resolve relative path
            if yaml_path is not None:
                # Relative to YAML file directory
                resolved_prompt_path = yaml_path.parent / prompt_path
            else:
                # Relative to current working directory
                resolved_prompt_path = prompt_path

        # Load prompt content
        try:
            with open(resolved_prompt_path, "r", encoding="utf-8") as f:
                prompt_content = f.read().strip()
        except FileNotFoundError:
            raise PromptFileNotFoundException(str(resolved_prompt_path))

        # Create prompt with file stem as ID
        prompt_id = resolved_prompt_path.stem
        return Prompt(id=prompt_id, prompt=prompt_content)

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
            ValueError: If state_filename doesn't end with .json or if include_data=True
                but data_format is None.
            DataFilenameExtensionMismatchException: If data_filename extension
                doesn't match data_format.
        """
        # Validate state_filename ends with .json
        if not state_filename.endswith(".json"):
            raise ValueError("state_filename must end with .json")

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
        return MetaEvaluatorState(
            client_registry=self._serialize_client_registry(),
            data=self._serialize_data(include_data, data_format, data_filename),
            eval_task=self._serialize_eval_task(include_task),
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

        Examples:
            # Load everything (clients + data + task)
            evaluator = MetaEvaluator.load_state("my_state.json")

            # Load only clients, skip data and task
            evaluator = MetaEvaluator.load_state("my_state.json", load_data=False, load_task=False)

            # Load clients and task, skip data
            evaluator = MetaEvaluator.load_state("my_state.json", load_data=False, load_task=True)

            # Provide custom API keys
            evaluator = MetaEvaluator.load_state("my_state.json",
                                                   openai_api_key="custom-key")

        Raises:
            ValueError: If state_file doesn't end with .json or if the JSON structure is invalid.
        """
        # Validate state_filename ends with .json
        if not state_filename.endswith(".json"):
            raise ValueError("state_filename must end with .json")

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

        # Reconstruct clients
        evaluator._reconstruct_clients(
            state.client_registry, openai_api_key, azure_openai_api_key
        )

        # Load data if requested and available
        if load_data and state.data is not None:
            evaluator._reconstruct_data(state.data)

        # Load task if requested and available
        if load_task and state.eval_task is not None:
            evaluator._reconstruct_task(state.eval_task)

        return evaluator

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
