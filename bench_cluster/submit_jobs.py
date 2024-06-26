from enum import Enum
import os
from jinja2 import Template
import subprocess
import yaml
from typing import List 

class Status(Enum):
    # INIT -> PENDING -> [RUNNING | FAIL | OOM] -> COMPLETED
    INIT = "init"           # Job is created
    PENDING = "pending"     # Job is waiting for ressources
    RUNNING = "running"     # Job is running
    FAIL = "fail"           # Job failed
    OOM = "oom"             # Job failed due to out of memory (expected behavior)
    COMPLETED = "completed" # Job is completed

class Job:
    def __init__(self, root_path: str, qos: str) -> None:
        self.root_path = root_path        
        self.name = os.path.basename(root_path)
        self.config = os.path.join(root_path, "config.yaml")
        self.qos = qos
        
        # Check if the status.txt file exists
        status_file_path = os.path.join(self.root_path, "status.txt")
        if not os.path.exists(status_file_path):
            # Create the status.txt file with INIT status
            with open(status_file_path, 'w') as f:
                f.write(Status.INIT.value)
        self.status = self.get_status()
        
    def get_status(self) -> Status:
        """
        Read the status of the job from `status.txt` and return it
        """
        is_existing = lambda value_to_check: any(value.value == value_to_check for value in Status.__members__.values())

        status_file_path = os.path.join(self.root_path, "status.txt")
        with open(status_file_path, 'r') as f:
            status = f.read()
            if not is_existing(status):
                raise ValueError("Invalid status")
            return Status(status)
        
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
        jobs_directory_paths = [os.path.abspath(root) for root, dirs, _ in os.walk(inp_dir) if not dirs]
        jobs_directory_paths = [job_path.replace("/profiler", "") if "profiler" in job_path else job_path for job_path in jobs_directory_paths]
        self.job_lists = [Job(job_path, qos) for job_path in jobs_directory_paths]

    def keep_only_jobs(self, status: Status):
        return [job for job in self.job_lists if job.status == status]
    
    def filter_out_jobs(self, status: Status):
        return [job for job in self.job_lists if job.status != status]
     
    def create_slurm_script(self, job: Job):
        # Submit job to the cluster (edit jinja)    
        # load yaml config.yaml
        with open(job.config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        
        # Pick the right number of nodes and n_proc_per_node
        world_size = config['parallelism']['pp'] * config['parallelism']['dp'] * config['parallelism']['tp']
        assert world_size <= 8 or world_size % 8 == 0
        nodes = max(1, world_size // 8)
        n_proc_per_node = min(8, world_size // nodes)
        assert nodes * n_proc_per_node == world_size
        
        target_path_hf_hub = os.path.join(os.path.basename(os.path.dirname(job.root_path)), os.path.basename(job.root_path))
        
        context_bench = {
            'run_id': "12345",
            'nodes': nodes,
            'n_proc_per_node': n_proc_per_node,
            'root_path': job.root_path,
            'target_path_hf_hub': target_path_hf_hub,
            "config": job.config,
            "qos": job.qos,
        }
        
        with open("/fsx/ferdinandmom/ferdinand-hf/bench_cluster/bench_cluster/template/base_bench.slurm", 'r') as file:
            base_bench_file = file.read()
        
        base_bench_template = Template(base_bench_file)
                
        # Write the rendered script to a new file located at the job root_path
        output_file_path = os.path.join(job.root_path, "bench.slurm")
        with open(output_file_path, 'w') as file:
            file.write(base_bench_template.render(context_bench))

        print(f"Slurm script created at {output_file_path}")
    
    def launch_dependency(self, job_array: List[Job], index_with_array, previous_job_id=None):

        slurm_scripts = [os.path.join(job.root_path, "bench.slurm") for job in job_array]
                
        slurm_command = [
            "sbatch",
            f"--array=0-{len(job_array) - 1}",
            f"--job-name=bench_job_{index_with_array}_array_%A",
        ]
        
        if previous_job_id:
            slurm_command.append(f"--dependency=afterany:{previous_job_id}")
        
        slurm_command.append(slurm_scripts[index_with_array])
        result = subprocess.run(slurm_command, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        return job_id
  
    def check_status(self):
        # find all status files using self.jobs_directory_paths
        status_files = [os.path.join(job.root_path, "status.txt") for job in self.job_lists]
                
        status_counts = {
            "init": 0,
            "pending": 0,
            "running": 0,
            "fail": 0,
            "oom": 0,
            "completed": 0
        }
        
        for status_file in status_files:
            with open(status_file, 'r') as f:
                status = f.read().strip()
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    raise ValueError(f"Invalid status: {status}")

        total = sum(status_counts.values())
        
        # Print the status counts in a formatted table
        print(f"{'Status':<10} | {'Count':<6}")
        print(f"{'-'*10}-|-{'-'*6}")
        for status, count in status_counts.items():
            print(f"{status.capitalize():<10} | {count:<6}")
        
        print(f"{'-'*10}-|-{'-'*6}")
        print(f"{'Total':<10} | {total:<6}")

def submit_jobs(inp_dir, qos, hf_token, nb_slurm_array, only_fails=False):
    scheduler = Scheduler(inp_dir, qos)

    #TODO: batch into job arrays
    env_vars = os.environ.copy()
    env_vars["HUGGINGFACE_TOKEN"] = hf_token
    total_jobs = len(scheduler.job_lists)

    if only_fails:
        scheduler.job_lists = scheduler.keep_only_jobs(Status.FAIL)
        failed_jobs = len(scheduler.job_lists)
        if failed_jobs == 0:
            print("No failed jobs to resubmit")
            return
        print(f"Only {failed_jobs}/{total_jobs} jobs will be resubmitted")
    
    scheduler.job_lists = scheduler.filter_out_jobs(Status.COMPLETED)

    if nb_slurm_array > 0:
        # Use job dependecies
        
        # Distribute the jobs into the arrays        
        base_jobs_per_array = len(scheduler.job_lists) // nb_slurm_array
        extra_jobs = len(scheduler.job_lists) % nb_slurm_array
        distribution = [base_jobs_per_array] * nb_slurm_array
        for i in range(extra_jobs):
            distribution[i] += 1
        
        start = 0
        
        for i, nb_jobs in enumerate(distribution):
            previous_job_id = None
            end = start + nb_jobs
            job_array = scheduler.job_lists[start:end]
            
            print(f"Launching job Dependency array {i+1} with {nb_jobs} jobs")
            
            for index_within_array, job in enumerate(job_array):
                scheduler.create_slurm_script(job)
                job.set_status(Status.PENDING)
                previous_job_id = scheduler.launch_dependency(job_array, index_within_array, previous_job_id)
            
            start = end
    else:
        # Don't use job dependecies
        for job in scheduler.job_lists:
            scheduler.create_slurm_script(job)
            subprocess.run(["sbatch", os.path.join(job.root_path, "bench.slurm")], env=env_vars)
            job.set_status(Status.PENDING)
        
def check_status(inp_dir):
    scheduler = Scheduler(inp_dir, "")
    scheduler.check_status()