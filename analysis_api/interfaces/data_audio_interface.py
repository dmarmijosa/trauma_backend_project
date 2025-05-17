from typing import TypedDict, List, Optional, Dict, Any # O usa dataclasses

# Usando TypedDict para simular interfaces
class PcmData(TypedDict):
    samples: Any # Debería ser np.ndarray o similar, pero Float32Array no es un tipo Python estándar
    sampleRate: int

class ExtractedFeatures(TypedDict):
    totalDurationSec: float
    speakingDurationSec: float
    numberOfPauses: int
    totalPauseDurationSec: float
    averageEnergyRMS: float
    minPauseDurationMsUsed: Optional[int]
    notes: Optional[str]

class YamnetClassScore(TypedDict):
    className: str
    score: float

class YamnetResult(TypedDict):
    topClasses: List[YamnetClassScore]
    notes: Optional[str]

class F0ContourPoint(TypedDict):
    timeSec: float
    frequencyHz: float
    confidence: float

class CrepePitchResult(TypedDict):
    meanF0Hz: Optional[float]
    stdDevF0Hz: Optional[float]
    minF0Hz: Optional[float]
    maxF0Hz: Optional[float]
    f0Contour: Optional[List[F0ContourPoint]]
    notes: Optional[str]

class EmotionScore(TypedDict):
    emotion: str
    score: float

class SpeechEmotionResult(TypedDict):
    emotions: List[EmotionScore]
    notes: Optional[str]

class CombinedAnalysisData(TypedDict):
    fileName: str
    analysisTimestamp: str # ISO format string
    basicAcousticFeatures: Optional[ExtractedFeatures]
    yamnetEvents: Optional[YamnetResult]
    pitchAnalysisF0: Optional[CrepePitchResult]
    emotionAnalysisSER: Optional[SpeechEmotionResult]
    transcription: Optional[str]
    warnings: Optional[List[str]]

class FinalAnalysisResponse(CombinedAnalysisData): # Hereda de CombinedAnalysisData
    pdfReportUrl: Optional[str]

# Si prefieres dataclasses (requiere Python 3.7+):
# from dataclasses import dataclass, field
# from typing import List, Optional, Any

# @dataclass
# class PcmData:
#     samples: Any # O np.ndarray
#     sampleRate: int
# ... y así para las demás ...