from queue import Queue
from .candidate import Candidate
from STELLAR.Utils.logger import LOGGERS
import threading
import traceback
logger = LOGGERS["candidate_queue"]

class CandidateQueue:
    
    def __init__(self):
        self.queue = Queue()
        self.process_thread = threading.Thread(target=self.process_candidates, daemon=True)
        self.process_thread.start()
        self.runs_per_candidate = 1
    
    def add_candidate(self, candidate: Candidate):
        self.queue.put(candidate)

    def process_candidates(self):
        while True:
            candidate = self.queue.get()
            try:
                for i in range(self.runs_per_candidate):
                    if i == 0:
                        candidate.init_and_run()
                    else:
                        candidate.init_and_run(force_disable_darshan=True)
                if self.runs_per_candidate > 1:
                    candidate.calculate_average_score(self.runs_per_candidate)
                candidate.status = "success"
            except Exception as e:
                logger.error(f"Error processing candidate: {e}")
                logger.error(traceback.format_exc())
                candidate.status = "failed"
            finally:
                self.queue.task_done()

CANDIDATE_QUEUE = None

def get_candidate_queue():
    global CANDIDATE_QUEUE
    if CANDIDATE_QUEUE is None:
        CANDIDATE_QUEUE = CandidateQueue()
    return CANDIDATE_QUEUE
