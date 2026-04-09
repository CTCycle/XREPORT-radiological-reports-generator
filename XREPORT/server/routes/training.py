"""Backward-compatible import surface for training route symbols."""

from XREPORT.server.api import training as training_api

TrainingEndpoint = training_api.TrainingEndpoint
TrainingState = training_api.TrainingState
drain_worker_progress = training_api.drain_worker_progress
enforce_worker_stop_timeout = training_api.enforce_worker_stop_timeout
handle_training_progress = training_api.handle_training_progress
job_manager = training_api.job_manager
monitor_training_process = training_api.monitor_training_process
read_worker_result = training_api.read_worker_result
request_worker_stop_if_needed = training_api.request_worker_stop_if_needed
resolve_checkpoint_path = training_api.resolve_checkpoint_path
router = training_api.router
run_resume_training_job = training_api.run_resume_training_job
run_training_job = training_api.run_training_job
training_endpoint = training_api.training_endpoint
training_state = training_api.training_state

__all__ = [
    "TrainingEndpoint",
    "TrainingState",
    "drain_worker_progress",
    "enforce_worker_stop_timeout",
    "handle_training_progress",
    "job_manager",
    "monitor_training_process",
    "read_worker_result",
    "request_worker_stop_if_needed",
    "resolve_checkpoint_path",
    "router",
    "run_resume_training_job",
    "run_training_job",
    "training_endpoint",
    "training_state",
]
