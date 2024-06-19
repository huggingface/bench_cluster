from enum import Enum
import os
from jinja2 import Template
import subprocess

class Status(Enum):
    # INIT -> PENDING -> [RUNNING | FAIL] -> COMPLETED
    INIT = "init"           # Job is created
    PENDING = "pending"     # Job is waiting for ressources
    RUNNING = "running"     # Job is running
    FAIL = "fail"           # Job failed
    COMPLETED = "completed" # Job is completed

class Job:
    def __init__(self, root_path: str, qos: str) -> None:
        self.root_path = root_path        
        self.name = os.path.basename(root_path)
        self.config = os.path.join(root_path, "config.yaml")
        self.qos = qos
        self.status = Status.INIT
        
    def set_status(self, status: Status) -> Status:
        """
        Update the status of the job in `status.txt` and return the new status
        """
        status_file_path = os.path.join(self.root_path, "status.txt")
        with open(status_file_path, 'w') as f:
            f.write(status.value)
            return status        
    
class Scheduler:
    
    def __init__(self, inp_dir: str, qos: str) -> None:
        self.inp_dir = inp_dir
        self.jobs_directory_paths = [os.path.abspath(root) for root, dirs, _ in os.walk(inp_dir) if not dirs]
        self.job_lists = [Job(path, qos) for path in self.jobs_directory_paths]

    def filter_job(self, status: Status):
        return [job for job in self.jobs if job.status == status]
    
    def create_slurm_script(self, job: Job):
        # Submit job to the cluster (edit jinja)       
        context = {
            'run_id': "12345",
            'nodes': 1,
            'n_proc_per_node': 1,
            'root_path': job.root_path,
            "config": job.config,
            "qos": job.qos
        }
        
        with open("/fsx/ferdinandmom/ferdinand-hf/bench_cluster/bench_cluster/template/base_bench.slurm", 'r') as file:
            script_template = file.read()

        template = Template(script_template)
        
        # Render the template with the context
        script = template.render(context)

        # Write the rendered script to a new file located at the job root_path
        output_file_path = os.path.join(job.root_path, "bench.slurm")
        with open(output_file_path, 'w') as file:
            file.write(script)
            
        print(f"Slurm script created at {output_file_path}")
    
    def submit_job(self, job: Job):
        job.set_status(Status.INIT)
        self.create_slurm_script(job)
        
        # Submit job to the cluster using subprocess
        subprocess.run(["sbatch", os.path.join(job.root_path, "bench.slurm")])
        
        # Edit the status of the job (by parsing squeue logs)    
    
def submit_jobs(inp_dir, qos):
    scheduler = Scheduler(inp_dir, qos)

    #TODO: relaunch only with status failed

    # Submit job in parallel
    for job in scheduler.job_lists:
        scheduler.submit_job(job)
        

