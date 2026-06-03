"""Project settings loaded from a `.env` file (all values optional).

None of the credentials are required to train models or write local results.
They are only needed for specific features:
  - openai_api_key : the `LLM` coherence-rating metric + vocab embeddings
  - hf_token       : downloading/uploading processed datasets on the HF Hub
  - wandb_api_key  : logging runs to Weights & Biases (only with `--wandb`)

Use `settings.require("openai_api_key", "the LLM rating metric")` to fetch a
value and raise a clear, actionable error only when a feature actually needs it.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
import warnings
import os


class Settings(BaseSettings):
    openai_api_key: str | None = Field(None)
    hf_token: str | None = Field(None)
    hf_username: str | None = Field(None)
    wandb_api_key: str | None = Field(None)
    wandb_entity: str | None = Field(None)

    # Optional cache directory configurations
    numba_cache_dir: str | None = Field(None, description="Numba cache directory")
    wandb_dir: str | None = Field(None, description="WandB working directory")
    wandb_cache_dir: str | None = Field(None, description="WandB cache directory")
    wandb_data_dir: str | None = Field(None, description="WandB data directory")
    hf_home: str | None = Field(None, description="Hugging Face cache home")
    transformers_cache: str | None = Field(None, description="Transformers cache directory")

    class Config:
        env_file = '.env'
        extra = "allow"

    def model_post_init(self, __context) -> None:
        """Configure cache directories and log in to services if credentials exist."""
        # Set cache directories if specified
        if self.numba_cache_dir:
            os.environ.setdefault('NUMBA_CACHE_DIR', self.numba_cache_dir)
        if self.wandb_dir:
            os.environ.setdefault('WANDB_DIR', self.wandb_dir)
        if self.wandb_cache_dir:
            os.environ.setdefault('WANDB_CACHE_DIR', self.wandb_cache_dir)
        if self.wandb_data_dir:
            os.environ.setdefault('WANDB_DATA_DIR', self.wandb_data_dir)
        if self.hf_home:
            os.environ.setdefault('HF_HOME', self.hf_home)
        if self.transformers_cache:
            os.environ.setdefault('TRANSFORMERS_CACHE', self.transformers_cache)

        # Log in to services only when credentials are provided.
        if self.hf_token:
            try:
                from huggingface_hub import login as hf_login
                hf_login(token=self.hf_token)
            except Exception as e:  # pragma: no cover - network/credential issues
                warnings.warn(f"Failed to login to Hugging Face Hub: {e}")

        if self.wandb_api_key:
            try:
                from wandb import login as wandb_login
                wandb_login(key=self.wandb_api_key)
            except Exception as e:  # pragma: no cover - network/credential issues
                warnings.warn(f"Failed to login to Weights & Biases: {e}")

    def require(self, attr: str, feature: str) -> str:
        """Return a credential, raising a clear error if it is missing."""
        value = getattr(self, attr, None)
        if not value:
            raise RuntimeError(
                f"Setting '{attr}' is required for {feature}, but it is not set. "
                f"Add it to your .env file (see .env.example)."
            )
        return value


settings = Settings()
