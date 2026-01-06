"""Data validation."""

from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class PromptRecord(BaseModel):
    text: str = Field(min_length=1)
    label: int = Field(ge=0, le=1)

def validate_dataset(df) -> None:
    for _, row in df.iterrows():
        PromptRecord(text=row["text"], label=row["label"])
    logger.info("Data validation passed")