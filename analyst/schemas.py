"""Typed inputs and outputs for the analysis tools.

A model (or a rule-based parser) proposes tool parameters as JSON; these
classes decide whether they are acceptable before any query runs.
extra="forbid" makes an unexpected field an error instead of being
silently ignored. The same classes later generate the JSON Schema that
tool definitions send to the model.
"""

from __future__ import annotations

from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

Geography = Literal["CITYWIDE", "BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
YearMonth = Annotated[str, StringConstraints(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
CategoryName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class MonthRange(BaseModel):
    """Whole calendar months, both ends included, written "YYYY-MM"."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: YearMonth
    end: YearMonth

    @model_validator(mode="after")
    def _start_not_after_end(self) -> MonthRange:
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        return self

    @property
    def months(self) -> int:
        start_year, start_month = map(int, self.start.split("-"))
        end_year, end_month = map(int, self.end.split("-"))
        return (end_year - start_year) * 12 + (end_month - start_month) + 1

    def each_month(self) -> list[str]:
        year, month = map(int, self.start.split("-"))
        out = []
        for _ in range(self.months):
            out.append(f"{year:04d}-{month:02d}")
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return out

    def label(self) -> str:
        return self.start if self.start == self.end else f"{self.start} to {self.end}"


class ComparePeriodsParams(BaseModel):
    """Compare complaint counts by complaint type between two equal-length periods."""

    model_config = ConfigDict(extra="forbid")

    geography: Geography
    period_a: MonthRange = Field(description="The earlier, baseline period")
    period_b: MonthRange = Field(description="The later period, compared with period_a")
    categories: list[CategoryName] | None = Field(
        default=None, min_length=1, max_length=25,
        description="Only these complaint types; omit to include all of them",
    )
    top_n: int = Field(default=10, ge=1, le=25, description="How many categories to return")

    @field_validator("geography", mode="before")
    @classmethod
    def _uppercase_geography(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _comparable_periods(self) -> ComparePeriodsParams:
        if self.period_a.end >= self.period_b.start:
            raise ValueError("period_a must end before period_b starts")
        if self.period_a.months != self.period_b.months:
            raise ValueError(
                f"periods must be the same length (period_a has {self.period_a.months} "
                f"months, period_b has {self.period_b.months})"
            )
        return self


class CategoryChange(BaseModel):
    """One complaint type's counts in the two periods."""

    model_config = ConfigDict(extra="forbid")

    category: str
    count_a: int
    count_b: int
    change: int
    pct_change: float | None = Field(description="None when count_a is 0")
    share_of_change: float | None = Field(
        description="Percent of the total change; None when the total change is 0 or too small")
    flags: list[str]
    labels: list[str] = Field(description="Raw data labels combined into this row")


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str


class Provenance(BaseModel):
    """Enough to rerun the calculation and get the same numbers."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    params: dict[str, Any]
    unit: str
    calculation: str
    filters: list[str]
    sources: list[SourceFile]
    data_complete_through: str
    data_refreshed_at: str
    computed_at: str


RowT = TypeVar("RowT", bound=BaseModel)


class ToolResult(BaseModel, Generic[RowT]):
    """The envelope every tool returns. Only the row type differs between tools."""

    model_config = ConfigDict(extra="forbid")

    summary: dict[str, int | float | None]
    rows: list[RowT]
    warnings: list[str]
    provenance: Provenance
