import multiprocessing as mp
from multiprocessing import Process, Queue
import time
from typing import Any, Dict
import queue


class ComplexObject:
    """Example complex object that workers will maintain"""
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.state = {}
        self.counter = 0
        print(f"Worker {worker_id}: Initialized complex object")
    
    def update(self, params: Dict[str, Any]):
        """Update the object based on parameters"""
        self.counter += 1
        self.state.update(params)
        # Simulate some work
        result = sum(params.values()) if all(isinstance(v, (int, float)) for v in params.values()) else 0
        print(f"Worker {self.worker_id}: Updated with {params}, counter={self.counter}, result={result}")
        return result


def worker_process(worker_id: int, task_queue: Queue, result_queue: Queue, ready_queue: Queue):
    """Worker process that maintains its own complex object"""
    # Initialize the complex object once
    obj = ComplexObject(worker_id)
    
    # Signal to master that this worker is ready
    ready_queue.put(worker_id)
    print(f"Worker {worker_id}: Ready and waiting for tasks")
    
    while True:
        try:
            # Wait for task from master
            task = task_queue.get()
            
            # Check for shutdown signal
            if task is None:
                print(f"Worker {worker_id}: Received shutdown signal")
                break
            
            target_worker_id, params = task
            
            # Only process if this task is for this worker
            if target_worker_id == worker_id:
                # Process the task
                result = obj.update(params)
                
                # Send result back to master
                result_queue.put((worker_id, result, obj.counter))
            else:
                # Put task back for other workers
                task_queue.put(task)
                time.sleep(0.01)  # Brief sleep to avoid busy waiting
                
        except Exception as e:
            print(f"Worker {worker_id}: Error - {e}")
            result_queue.put((worker_id, None, str(e)))
    
    print(f"Worker {worker_id}: Shutting down")


class Master:
    """Master that manages workers"""
    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.ready_queue = Queue()
        self.workers = []
        
    def start_workers(self):
        """Initialize and start all worker processes"""
        print(f"Master: Starting {self.num_workers} workers...")
        
        for worker_id in range(self.num_workers):
            p = Process(
                target=worker_process,
                args=(worker_id, self.task_queue, self.result_queue, self.ready_queue)
            )
            p.start()
            self.workers.append(p)
        
        # Wait for all workers to signal they are ready
        print("Master: Waiting for workers to initialize...")
        ready_workers = []
        for _ in range(self.num_workers):
            worker_id = self.ready_queue.get()  # Blocks until a worker is ready
            ready_workers.append(worker_id)
            print(f"Master: Worker {worker_id} is ready")
        
        print(f"Master: All {len(ready_workers)} workers are ready")
    
    def send_task(self, worker_id: int, params: Dict[str, Any]):
        """Send a task to a specific worker"""
        if worker_id >= self.num_workers:
            raise ValueError(f"Worker ID {worker_id} out of range")
        
        print(f"Master: Sending task to worker {worker_id} with params {params}")
        self.task_queue.put((worker_id, params))
    
    def get_result(self, timeout: float = 5.0):
        """Get result from any worker"""
        try:
            worker_id, result, counter = self.result_queue.get(timeout=timeout)
            print(f"Master: Received result from worker {worker_id}: {result} (counter={counter})")
            return worker_id, result, counter
        except queue.Empty:
            print("Master: Timeout waiting for result")
            return None, None, None
    
    def shutdown(self):
        """Shutdown all workers gracefully"""
        print("Master: Sending shutdown signal to all workers...")
        
        # Send shutdown signal to each worker
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        
        # Wait for all workers to finish
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                print(f"Master: Force terminating worker {p.pid}")
                p.terminate()
        
        print("Master: All workers shut down")


def main():
    """Example usage"""
    # Create master with 4 workers
    master = Master(num_workers=4)
    
    # Start all workers
    master.start_workers()
    
    try:
        # Send tasks to different workers
        master.send_task(0, {"value": 10, "multiplier": 2})
        master.send_task(1, {"value": 20, "multiplier": 3})
        master.send_task(2, {"value": 30, "multiplier": 4})
        master.send_task(3, {"value": 40, "multiplier": 5})
        
        # Get results
        for _ in range(4):
            master.get_result()
        
        # Send more tasks to specific workers
        time.sleep(0.5)
        master.send_task(0, {"value": 15, "multiplier": 2})
        master.send_task(0, {"value": 25, "multiplier": 3})
        master.send_task(2, {"value": 35, "multiplier": 4})
        
        # Get results
        for _ in range(3):
            master.get_result()
        
    finally:
        # Always shutdown workers
        master.shutdown()


if __name__ == "__main__":
    # Required for Windows
    mp.set_start_method('spawn', force=True)
    main()