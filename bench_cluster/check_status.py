import os
import glob

def check_status(inp_dir):
    status_files = glob.glob(os.path.join(inp_dir, "**", "status.txt"), recursive=True)
    
    status_counts = {
        "init": 0,
        "pending": 0,
        "running": 0,
        "fail": 0,
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