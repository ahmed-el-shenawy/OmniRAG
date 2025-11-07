from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):

    query: str


    model_config = {"from_attributes": True}