import psutil

# return psutil.virtual_memory().percent


# # Get the total memory available on the GPU
#     total_memory = torch.cuda.get_device_properties(0).total_memory
#     print(f"Total memory: {total_memory / (1024 ** 3):.2f} GB")

#     # Get the allocated memory
#     allocated_memory = torch.cuda.memory_allocated(0)
#     print(f"Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")

#     # Get the cached memory
#     cached_memory = torch.cuda.memory_reserved(0)
#     print(f"Cached memory: {cached_memory / (1024 ** 3):.2f} GB")

#     # Get the maximum memory allocated by PyTorch
#     max_memory_allocated = torch.cuda.max_memory_allocated(0)
#     print(f"Max memory allocated: {max_memory_allocated / (1024 ** 3):.2f} GB")


def get_gpu_memory():

    if torch.cuda.is_available():
        # Get the total memory available on the GPU
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"Total memory: {total_memory / (1024 ** 3):.2f} GB")

        # Get the allocated memory
        allocated_memory = torch.cuda.memory_allocated(0)
        print(f"Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")

        # Get the cached memory
        cached_memory = torch.cuda.memory_reserved(0)
        print(f"Cached memory: {cached_memory / (1024 ** 3):.2f} GB")

        # Get the maximum memory allocated by PyTorch
        max_memory_allocated = torch.cuda.max_memory_allocated(0)
        print(f"Max memory allocated: {max_memory_allocated / (1024 ** 3):.2f} GB")
    else:
        print("CUDA is not available.")


import logging

import torch


class GPUMonitor:
    def __init__(self, device=0):
        self.device = device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        self.device_name = torch.cuda.get_device_name(self.device)

    def get_memory_info(self):
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        allocated_memory = torch.cuda.memory_allocated(self.device)
        reserved_memory = torch.cuda.memory_reserved(self.device)
        max_memory_allocated = torch.cuda.max_memory_allocated(self.device)
        max_memory_reserved = torch.cuda.max_memory_reserved(self.device)

        memory_info = {
            "device_name": self.device_name,
            "total_memory": total_memory / (1024**3),
            "allocated_memory": allocated_memory / (1024**3),
            "reserved_memory": reserved_memory / (1024**3),
            "max_memory_allocated": max_memory_allocated / (1024**3),
            "max_memory_reserved": max_memory_reserved / (1024**3),
        }
        return memory_info

    def print_memory_info(self):
        info = self.get_memory_info()
        print(f"GPU Device: {info['device_name']}")
        print(f"Total memory: {info['total_memory']:.2f} GB")
        print(f"Allocated memory: {info['allocated_memory']:.2f} GB")
        print(f"Reserved memory: {info['reserved_memory']:.2f} GB")
        print(f"Max memory allocated: {info['max_memory_allocated']:.2f} GB")
        print(f"Max memory reserved: {info['max_memory_reserved']:.2f} GB")

    def log_memory_info(self, logger=None):
        info = self.get_memory_info()
        log_message = (
            f"GPU Device: {info['device_name']}\n"
            f"Total memory: {info['total_memory']:.2f} GB\n"
            f"Allocated memory: {info['allocated_memory']:.2f} GB\n"
            f"Reserved memory: {info['reserved_memory']:.2f} GB\n"
            f"Max memory allocated: {info['max_memory_allocated']:.2f} GB\n"
            f"Max memory reserved: {info['max_memory_reserved']:.2f} GB"
        )
        if logger:
            logger.error(log_message)
        else:
            print(log_message)

    def reset_peak_memory_stats(self):
        torch.cuda.reset_peak_memory_stats(self.device)

    def empty_cache(self):
        torch.cuda.empty_cache()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        filename="./logs/running.log",
        filemode="a",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    gpu_monitor = GPUMonitor()

    try:
        # Your training code here
        gpu_monitor.print_memory_info()
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        gpu_monitor.empty_cache()
