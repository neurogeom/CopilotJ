# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import re
import threading

from jupyter_client.manager import KernelManager


class JupyterNotebook:
    def __init__(self):
        self.km = KernelManager()
        self.km.start_kernel()
        self.kc = self.km.client()

    def clean_output(self, outputs, max_length: int = 8000) -> str:
        outputs_only_str = list()
        for i in outputs:
            if isinstance(i, dict):
                if "text/plain" in list(i.keys()):
                    outputs_only_str.append(i["text/plain"])
            elif isinstance(i, str):
                outputs_only_str.append(i)
            elif isinstance(i, list):
                error_msg = "\n".join(i)
                error_msg = re.sub(r"\x1b\[.*?m", "", error_msg)
                outputs_only_str.append(error_msg)

        result = "\n".join(outputs_only_str).strip()

        # Limit output length to prevent token explosion
        if len(result) > max_length:
            truncated_length = max_length - 200  # Reserve space for truncation notice
            result = (
                result[:truncated_length]
                + f"\n\n... [Output truncated, original length: {len(result)} characters, showing first {truncated_length} characters] ..."
            )

        return result

    def add_and_run(self, code_string: str) -> tuple[str, bool]:
        outputs = []
        ok = True

        # This inner function will be executed in a separate thread
        def run_code_in_thread():
            nonlocal outputs, ok

            # Execute the code and get the execution count
            self.kc.execute(code_string)

            while True:
                try:
                    msg = self.kc.get_iopub_msg(timeout=20)

                    msg_type = msg["header"]["msg_type"]
                    content = msg["content"]

                    if msg_type == "execute_result":
                        outputs.append(content["data"])
                    elif msg_type == "stream":
                        outputs.append(content["text"])
                    elif msg_type == "error":
                        ok = False
                        outputs.append(content["traceback"])

                    # If the execution state of the kernel is idle, it means the cell finished executing
                    if msg_type == "status" and content["execution_state"] == "idle":
                        break

                except Exception:
                    break

        # Start the thread to run the code
        thread = threading.Thread(target=run_code_in_thread)
        thread.start()

        # Wait for 10 seconds for the thread to finish
        thread.join(timeout=300)

        # If the thread is still alive after 10 seconds, it's a timeout
        if thread.is_alive():
            outputs = ["Timeout after 300 seconds"]
            ok = False

        return self.clean_output(outputs, max_length=8000), ok

    def close(self):
        """Shutdown the kernel."""
        self.km.shutdown_kernel()
