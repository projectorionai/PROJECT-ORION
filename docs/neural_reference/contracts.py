from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Resources(StrictModel):
    tool: Literal["resources"] = "resources"


class ReadText(StrictModel):
    tool: Literal["read_text"] = "read_text"
    path: str = Field(min_length=1, max_length=240)


class CreateText(StrictModel):
    tool: Literal["create_text"] = "create_text"
    path: str = Field(min_length=1, max_length=240)
    text: str = Field(max_length=16_000)


class LaunchNotepad(StrictModel):
    tool: Literal["launch_notepad"] = "launch_notepad"


class GpuStatus(StrictModel):
    tool: Literal["gpu_status"] = "gpu_status"


class Screenshot(StrictModel):
    tool: Literal["screenshot"] = "screenshot"


Action = Annotated[
    Resources | ReadText | CreateText | LaunchNotepad | GpuStatus | Screenshot,
    Field(discriminator="tool"),
]


class Step(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    depends_on: tuple[str, ...] = Field(default=(), max_length=16)
    action: Action


class Plan(StrictModel):
    summary: str = Field(min_length=1, max_length=500)
    steps: tuple[Step, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def valid_dag(self):
        ids = {s.id for s in self.steps}
        if len(ids) != len(self.steps):
            raise ValueError("Duplicate step identifiers")
        if any(set(s.depends_on) - ids for s in self.steps):
            raise ValueError("Unknown dependency")
        try:
            tuple(
                TopologicalSorter(
                    {s.id: s.depends_on for s in self.steps}
                ).static_order()
            )
        except CycleError as exc:
            raise ValueError("Cyclic plan") from exc
        return self


class Outcome(StrictModel):
    ok: bool
    data: dict = Field(default_factory=dict)
    error: str = ""
