import math
from typing import Literal
from pydantic import BaseModel, Field, model_validator

Severity = Literal["low", "moderate", "high"]
Category = Literal["condition", "alignment"]
ModuleViewStatus = Literal["assessed_with_findings", "assessed_no_findings", "inconclusive"]


class Box(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def box_inside_image(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.w, self.h)):
            raise ValueError("Bounding box coordinates must be finite numbers.")
        if self.x + self.w > 1.001 or self.y + self.h > 1.001:
            raise ValueError(
                f"Bounding box must stay within normalized coordinates: x={self.x}, y={self.y}, w={self.w}, h={self.h}"
            )
        return self


class Finding(BaseModel):
    name: str
    category: Category = "condition"
    condition: str | None = None
    findingType: str | None = None
    involvedTeeth: list[str] = Field(default_factory=list)
    gapMeasurement: str | None = None
    affectedRegion: str | None = None
    confidence: float = Field(ge=0, le=1)
    severity: Severity
    description: str
    recommendation: str
    box: Box
    rawBox2d: list[int] | None = None
    tooth: str | None = None
    toothNumber: str | None = None

    @model_validator(mode="after")
    def sync_finding_fields(self):
        if not math.isfinite(self.confidence):
            raise ValueError("Confidence must be a finite number between 0 and 1.")

        if not self.condition and self.name:
            self.condition = self.name
        elif not self.name and self.condition:
            self.name = self.condition

        if not self.toothNumber and self.tooth:
            self.toothNumber = self.tooth
        elif not self.tooth and self.toothNumber:
            self.tooth = self.toothNumber

        if not self.tooth and self.involvedTeeth:
            self.tooth = ", ".join(self.involvedTeeth)
            self.toothNumber = self.tooth

        return self


class Views(BaseModel):
    upper: list[Finding]
    lower: list[Finding]
    front: list[Finding]


class ModulesSummary(BaseModel):
    toothConditions: dict[str, list[Finding]]
    dentalAlignment: dict[str, list[Finding]]


class ModuleStatusesSummary(BaseModel):
    toothConditions: dict[str, ModuleViewStatus]
    dentalAlignment: dict[str, ModuleViewStatus]


class ImageQualityReport(BaseModel):
    width: int
    height: int
    aspectRatio: float
    minDimension: int
    laplacianVariance: float
    edgeContrast: float
    meanBrightness: float
    qualityStatus: str
    qualityNotes: str | None = None


class ViewDiagnostic(BaseModel):
    viewStatus: ModuleViewStatus
    conditionStatus: ModuleViewStatus
    alignmentStatus: ModuleViewStatus
    visibleToothSummary: str | None = None
    unassessableTeethReason: str | None = None
    rawConditionsCount: int = 0
    rawAlignmentCount: int = 0
    rejectedCount: int = 0
    cropPassUsed: bool = False
    cropsAnalyzed: list[str] = Field(default_factory=list)


class ScreeningResponse(BaseModel):
    overallScore: float = Field(ge=0, le=100)
    views: Views
    modules: ModulesSummary | None = None
    viewStatuses: dict[str, ModuleViewStatus] | None = None
    moduleStatuses: ModuleStatusesSummary | None = None
    imageQuality: dict[str, ImageQualityReport] | None = None
    viewDiagnostics: dict[str, ViewDiagnostic] | None = None
    screeningDisclaimer: str = (
        "AI-assisted screening only. This result is not a diagnosis and must be reviewed by a qualified dental professional."
    )
    isMock: bool = False
    modelProvider: str = "nvidia (nvidia/nemotron-3.5-super-vl-120b-a12b)"
    inconclusive: bool = False
    inconclusiveViews: list[str] = Field(default_factory=list)
    inconclusiveReasons: dict[str, str] = Field(default_factory=dict)
    statusMessage: str | None = None
    rejectedFindingsCount: int = 0
