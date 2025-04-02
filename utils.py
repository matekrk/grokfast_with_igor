import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

# info asynchronous unlinking of several files
async def remove_files_async(file_list):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(
            executor,
            lambda: [os.remove(f) for f in file_list if os.path.exists(f)]
        )

### info to execute
# file_list = [f"checkpoint_{i}.pth" for i in range(100)]
# asyncio.run(remove_files_async(file_list))