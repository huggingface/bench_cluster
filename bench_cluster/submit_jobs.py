from enum import Enum
import os
from jinja2 import Template
import subprocess
import yaml
import glob

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
        
        #TODO: think about how to set the proper value for 
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task=11
        
        target_path_hf_hub = os.path.join(os.path.basename(os.path.dirname(job.root_path)), os.path.basename(job.root_path))
        
        context_bench = {
            'run_id': "12345",
            'nodes': nodes,
            'n_proc_per_node': n_proc_per_node,
            'root_path': job.root_path,
            'target_path_hf_hub': target_path_hf_hub,
            "config": job.config,
            "qos": job.qos
        }
        
        with open("/fsx/ferdinandmom/ferdinand-hf/bench_cluster/bench_cluster/template/base_bench.slurm", 'r') as file:
            base_bench_file = file.read()
        
        base_bench_template = Template(base_bench_file)
                
        # Write the rendered script to a new file located at the job root_path
        output_file_path = os.path.join(job.root_path, "bench.slurm")
        with open(output_file_path, 'w') as file:
            file.write(base_bench_template.render(context_bench))

        print(f"Slurm script created at {output_file_path}")
            
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

def submit_jobs(inp_dir, qos, hf_token, only_fails=False):
    scheduler = Scheduler(inp_dir, qos)

    #TODO: Launch using slurm job array
    #TODO: Edit time in base_bench.slurm script
    #TODO: For how many steps do we have to run ?
    #TODO: add option to do recomputer layer in Nanotron
    #TODO: add info in logs about profiler ?
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
    
    for job in scheduler.job_lists:
        scheduler.create_slurm_script(job)
        subprocess.run(["sbatch", os.path.join(job.root_path, "bench.slurm")], env=env_vars)
        job.set_status(Status.PENDING)
        
def check_status(inp_dir):
    scheduler = Scheduler(inp_dir, "")
    scheduler.check_status()