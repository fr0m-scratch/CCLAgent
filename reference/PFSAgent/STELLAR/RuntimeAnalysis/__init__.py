from .darshan_agent import DarshanAnalysisAgent
from .elbencho_agent import ElbenchoAnalysisAgent


ANALYSIS_AGENTS = {
    "Darshan": DarshanAnalysisAgent,
    "Elbencho": ElbenchoAnalysisAgent
}

__all__ = ["DarshanAnalysisAgent", "ElbenchoAnalysisAgent", "ANALYSIS_AGENTS"]