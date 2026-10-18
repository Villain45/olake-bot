from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    # Official contract: {"question": "..."}
    question: str = Field(..., min_length=1, max_length=2000)
    product: Literal["ingestion", "fusion"] | None = None

    @field_validator("question", mode="before")
    @classmethod
    def preserve_question_format(cls, value):
        if isinstance(value, str):
            value = value.replace("\r\n", "\n").strip()
            if not value:
                raise ValueError("Question must contain non-whitespace text")
        return value


class AskResponse(BaseModel):
    # Official contract: {"answer": "...", "sources": ["<url>", ...]}
    answer: str
    sources: list[str]
