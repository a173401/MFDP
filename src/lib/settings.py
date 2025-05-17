from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    token: str = Field(..., env="TOKEN")
    openai_api_url: AnyUrl = Field(..., env="OPENAI_API_URL")
    prompt_directory: str = Field(..., env="PROMPT_DIRECTORY")

    model_config = SettingsConfigDict(env_file=".settings", env_file_encoding="utf-8", extra="ignore")

        

settings = Settings()