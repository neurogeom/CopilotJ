# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import logging
import platform
import signal
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Annotated

from copilotj.core import ImageMessage, TextMessage, new_vlm_model_client
from copilotj.core.config import Config
from copilotj.plugin.api import ClientPluginAPI
from copilotj.util import JupyterNotebook

__all__ = [
    "PluginTools",
    "PyimagejRunner",
    "execute_python_script",
]

_log = logging.getLogger(__name__)


class PluginTools:
    def __init__(self, apis: ClientPluginAPI, *, cfg: Config):
        super().__init__()
        self.apis = apis
        self._cfg = cfg

    def _detect_timeout_from_script(self, script: str) -> float:
        script_lower = script.lower()
        if any(keyword in script_lower for keyword in ["batch", "for(", "for (", "getfilelist", "while"]):
            return 180.0
        return 15.0

    async def run_macro(
        self,
        script: Annotated[str, "Valid ImageJ macro script to execute"],
        timeout: Annotated[float | None, "Timeout in seconds (auto-detected: 15s normal, 180s for batch)"] = None,
        verify_result: Annotated[bool, "Enable visual verification of operation results (costs extra time)"] = False,
        operation_intent: Annotated[
            str | None, "Required if verify_result=True: describe what the operation should achieve"
        ] = None,
    ) -> str:
        if timeout is None:
            timeout = self._detect_timeout_from_script(script)
        try:
            script = script + "\n" + 'print("Macro executed.");'
            response = await self.apis.run_script("macro", script, timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Script execution timeout ({timeout}s). For batch processing, consider breaking down the script or manually setting a longer timeout."
            )

        if response.err or "Error" in str(response):
            # Get basic window info instead of full perception for errors
            window_info = await self.imagej_windowInfo()
            raise RuntimeError(
                f"Error during execution: {str(response)} \n Please check and adjust the script and try again. \n ImageJ window info: {window_info}"
            )

        # Perform state check if needed
        result = f"{script} has executed successfully.\n"

        # Perform result verification if requested
        if verify_result and operation_intent and self._cfg.vision_enabled:
            verification = await self.simple_result_verification(operation_intent)
            result += f"\nOperation verification: {verification}"

        return result

    async def capture_image(self, title: str | None = None) -> str:
        """Capture ImageJ Image."""
        return await self.apis.capture_image(title)

    async def label_image(self) -> str:
        """Execute the human-in-the-loop mask export macro for current image set."""
        macro_script = label_macro()
        await self.run_macro(macro_script, timeout=300.0)
        return "The label_image macro has been executed successfully."

    async def capture_screen(self, query: str | None = None) -> str:
        """Capture ImageJ Screen."""
        if not self._cfg.vision_enabled:
            window_info = await self.imagej_windowInfo()
            return f"[Vision disabled] Basic ImageJ window info:\n{window_info}"

        if not self._cfg.vision_available:
            window_info = await self.imagej_windowInfo()
            return (
                "[Vision not available] The configured model does not support image input. "
                "Please configure a vision-capable model (COPILOTJ_VLM_MODEL) "
                "or use a model with vision support.\n"
                f"Basic ImageJ window info:\n{window_info}"
            )

        vlm = new_vlm_model_client(
            model=self._cfg.vlm_model or None,
            api_key=self._cfg.vlm_api_key or None,
            base_url=self._cfg.vlm_base_url,
            cfg=self._cfg,
        )
        imagejCapture_resp = await self.apis.capture_screen()
        vision_prompt = """
You are an ImageJ Vision Perception tool.  
Your task is to analyze the screenshot of the current ImageJ environment and return a 
detailed Markdown report describing the open images, window states, and biological content.

Your output MUST follow the exact Markdown structure below.

# 🖼️ Image Status
- **Has image:** yes/no  
- **Active window:** (window title or "none")  
- **File path:** (absolute path if visible; if not, write "unknown")  
- **Bit depth:** (8-bit / 16-bit / RGB / other)  
- **Channels detected:** e.g., C1=Green, C2=Red, Brightfield, etc.  
- **Is stack:** yes/no  
- **Number of slices:** (if known)  

# 🪟 Screen & Window Information
- **Open ImageJ windows:** list all visible windows  
- **Active ROI:** rectangle / polygon / line / none  
- **Scale bar present:** yes/no  
- **Z-position:** (if a stack)  
- **Frame index / slice index:** (if visible)  

# 🔬 Biological & Visual Content Interpretation
Provide a cautious but helpful interpretation of what is visible in the image:
- **Objects present:** e.g., "elongated bacteria-like rods", "nuclei", "fibrous structures"  
- **Possible cell types or organism types:** Macrophage / RBC / bacteria / neurons / yeast (if recognizable)  
- **Morphology summary:** e.g., “elongated rods”, “round cells”, “clustered nuclei”  
- **Fluorescence interpretation:** e.g., “green channel shows rod-like fluorescent structures suggesting GFP-tagged bacteria”  

Be cautious: **If uncertain, explicitly say uncertain.**

# 📉 Image Quality Assessment
Describe the technical quality of the image:
- **Brightness:** low / normal / high  
- **Contrast:** low / normal / high  
- **Noise level:** low / medium / high / severe  
- **Uneven background:** yes/no  
- **Focus quality:** sharp / soft / blurry  

# 🔧 Preprocessing & Analysis Recommendations
Provide recommendations for ImageJ preprocessing:
- **Should invert:** yes/no (based on foreground polarity)  
- **Recommended denoising:** Gaussian / Median / NLM / none  
- **Background correction needed:** yes/no + suggested radius  +
- **Recommended contrast enhancement:** e.g., CLAHE  
- **Suggested threshold method:** Otsu, Triangle, Yen, Moments, Auto, or user manual
- **Segmentation difficulty:** easy / moderate / difficult  
- **Suggested follow-up operations:** Open/Close, Fill Holes, Watershed, Analyze Particles  

# 📝 Notes
Provide any additional warnings or clarifications:
- e.g., “The image appears very noisy; segmentation will need preprocessing.”
- e.g., “Bacteria-like rods are dense; watershed may be unreliable.”

---

## Rules
1. All ImageJ metadata (window titles, file path, channel names, bit depth) must be correct.  
2. If an image is open, the **file path must appear** in the output when visible on screen.  
3. Biological interpretation must be reasonable and cautious.  
"""

        if query:
            vision_prompt += query
        imagejCapture = await vlm.create(
            [ImageMessage(role="user", image=i.image) for i in imagejCapture_resp.screenshots]
            + [TextMessage(role="user", text=vision_prompt)]
        )
        return imagejCapture.content or "no response"

    async def imagej_perception(self, query: str | None = None) -> str:
        """ImageJ Perception Tool."""
        perception = await self.imagej_windowInfo()

        if not self._cfg.vision_enabled:
            return f"[Vision disabled] ImageJ Window Information:\n{perception}"

        if not self._cfg.vision_available:
            return (
                "[Vision not available] The configured model does not support image input. "
                "Please configure a vision-capable model (COPILOTJ_VLM_MODEL) "
                "or use a model with vision support.\n"
                f"ImageJ Window Information:\n{perception}"
            )

        PROMPT_TEMPLATE = """

This is the leader agent's request: {{QUERY}}

For your reference, Following is the perception of ImageJ,
imagejStatus summarizes the status of ImageJ.
imagejOperation is the monitor event history of ImageJ.

{{PERCEPTION}}
"""
        prompt = PROMPT_TEMPLATE.replace("{{QUERY}}", str(query)).replace("{{PERCEPTION}}", perception)
        perception_str = await self.capture_screen(prompt)
        return perception_str

    async def imagej_windowInfo(self) -> str:
        """ImageJ Window Information Tool."""
        imagejStatus = await self.apis.take_snapshot()
        imagejOperation = await self.apis.get_operation_history()

        # Format as readable text instead of JSON to avoid string replacement issues
        result = f"""ImageJ Window Information:

=== ImageJ Status ===
{imagejStatus}

=== ImageJ Operation History ===
{imagejOperation}
"""
        return result

    async def simple_result_verification(self, operation_intent: str, expected_outcome: str = None) -> str:
        """Simple perception-based result verification."""
        if not self._cfg.vision_enabled:
            return "[Vision disabled] Visual verification skipped. Operation assumed successful."

        if not self._cfg.vision_available:
            return (
                "[Vision not available] Visual verification skipped — "
                "the configured model does not support image input. "
                "Operation assumed successful."
            )

        vlm = new_vlm_model_client(
            model=self._cfg.vlm_model or None,
            api_key=self._cfg.vlm_api_key or None,
            base_url=self._cfg.vlm_base_url,
            cfg=self._cfg,
        )
        imagejCapture_resp = await self.apis.capture_screen()

        verification_prompt = f"""
You are an ImageJ Result Verification assistant. 

OPERATION INTENT: {operation_intent}
EXPECTED OUTCOME: {expected_outcome or "Not specified"}

Please analyze the current ImageJ screen and provide a BRIEF assessment:

1. **Operation Success**: Did the intended operation appear to complete successfully?
2. **Visual Result**: Briefly describe what you see (e.g., "threshold applied", "new window opened", "ROI selected")
3. **Potential Issues**: Any obvious problems or suboptimal results?
4. **Recommendation**: Should we proceed, retry, or adjust the approach?

Keep your response concise and focused on actionable feedback.
"""

        imagej_verification = await vlm.create(
            [ImageMessage(role="user", image=i.image) for i in imagejCapture_resp.screenshots]
            + [TextMessage(role="user", text=verification_prompt)]
        )
        return imagej_verification.content or "no verification response"


async def folder_summary(folder_path: str):
    if folder_path.strip() == ".":
        return "The current directory is not a valid folder path, try to ask the user to provide a specific folder path or confirm the current directory."

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return f"The path '{folder_path}' is not a valid directory. Please ignore this tool and continue."

    summary = []
    MAX_FILES = 300

    for path in folder.rglob("*"):
        if len(summary) >= MAX_FILES:
            break
        relative_path = path.relative_to(folder)
        if path.is_file():
            summary.append(f"{relative_path}")
        elif path.is_dir():
            summary.append(f"Directory: {relative_path}")

    folder_summary = []
    for index, item in enumerate(summary, start=1):
        folder_summary.append(f"File {index}: {item}")

    files_summary = ",".join(folder_summary)

    total_files_msg = ""
    if len(summary) >= MAX_FILES:
        total_files_msg = f" (Showing first {MAX_FILES} items, more files may exist, need to ask user to check the folder directly if necessary)"

    prompt = f"""\
The folder path is set to: {folder_path}.\
The provided folder and uploaded files contain the following items{total_files_msg}:
{files_summary}.\
"""

    return prompt


async def system_info(add_python_info: bool = False):
    """
    Returns a string with information about the system: machine, operating system (all details), version of Python, etc.
    """

    info = {
        "Machine": f"{platform.machine()} ({platform.processor()} {platform.architecture()[0]})",
        "Version": platform.version().split(":")[0],
        "Platform": platform.platform(),
        "System": platform.system(),
    }

    if add_python_info:
        info |= {
            "Python Version": sys.version,
            "Python Compiler": platform.python_compiler(),
            "Python Implementation": platform.python_implementation(),
            "Python Build": platform.python_build(),
        }

    info_str = "```system_info\n"
    info_str += "\n".join([f"{key}: {value}" for key, value in info.items()])
    info_str += "\n```\n"

    return info_str


# Define tools
class PyimagejRunner:
    """Handles ImageJ execution within a Jupyter Notebook."""

    _notebook: JupyterNotebook

    def __init__(self, path_fiji: str | None = None):
        super().__init__()

        self._notebook = JupyterNotebook()
        mode = "interactive"

        if sys.platform == "darwin":
            warnings.warn("Running in headless mode on macOS.")
            mode = "headless"

        if path_fiji is not None:
            python_code = textwrap.dedent(f"""
                import imagej
                ij = imagej.init("{path_fiji}", mode="{mode}")
                ij.ui().showUI()
            """)
        else:
            python_code = textwrap.dedent(f"""
                import imagej
                ij = imagej.init(mode="{mode}")
                ij.ui().showUI()
            """)

        _, ok = self._notebook.add_and_run(python_code)
        if not ok:
            self._notebook.close()
            raise RuntimeError("Failed to initialize ImageJ.")

    def run_macro(self, script: str) -> str:
        """Runs an ImageJ macro inside Jupyter Notebook."""
        pyscript = rf"""ij.py.run_macro(r'''{script}''')"""
        result, ok = self._notebook.add_and_run(pyscript)

        if not ok:
            return f"Error during execution:\n{result}"

        return result

    def close(self) -> None:
        """Closes the notebook session."""
        self._notebook.close()


# execute_python_script tool
class JupyterClient:
    """Handles Python code execution using Jupyter client with proper lifecycle management."""

    def __init__(self):
        """Initialize Jupyter client."""
        self._notebook = None
        self._initialize_notebook()

        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _initialize_notebook(self):
        """Initialize or reinitialize the notebook."""
        try:
            if self._notebook:
                self._notebook.close()
        except Exception:
            pass

        self._notebook = JupyterNotebook()

    def _signal_handler(self, signum, frame):
        """Handle system signals."""
        _log.warning("Received signal %d, cleaning up Jupyter client", signum)
        self.cleanup()

    def execute_code(self, code: str) -> str:
        """Execute Python code using Jupyter client with auto-recovery.

        Args:
            code: Python code to execute

        Returns:
            str: Execution result
        """
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if not self._notebook:
                    self._initialize_notebook()

                assert self._notebook is not None
                result, ok = self._notebook.add_and_run(code)
                if not ok:
                    # Check if it's a kernel issue
                    if "kernel" in result.lower() or "parent appears to have exited" in result.lower():
                        _log.warning("Kernel issue detected on attempt %d, reinitializing", attempt + 1)
                        self._initialize_notebook()
                        if attempt < max_retries - 1:
                            continue
                    return f"Error executing code: {result}"
                return result

            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1 and (
                    "kernel" in error_msg.lower() or "parent appears to have exited" in error_msg.lower()
                ):
                    _log.warning("Connection issue on attempt %d, retrying", attempt + 1)
                    self._initialize_notebook()
                    continue
                return f"Error executing code: {error_msg}"

        return "Error: Failed to execute code after multiple attempts"

    def cleanup(self):
        """Clean up Jupyter client resources."""
        if self._notebook:
            try:
                _log.info("Shutting down Jupyter kernel...")
                self._notebook.close()
                _log.info("Jupyter kernel shut down successfully")
            except Exception as e:
                _log.error("Error shutting down Jupyter kernel: %s", e)
            finally:
                self._notebook = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


# Global Jupyter client instance with proper cleanup
_jupyter_client = None


def get_jupyter_client() -> JupyterClient:
    """Get or create Jupyter client."""
    global _jupyter_client
    if _jupyter_client is None:
        _jupyter_client = JupyterClient()
    return _jupyter_client


def cleanup_jupyter_client():
    """Cleanup global Jupyter client."""
    global _jupyter_client
    if _jupyter_client:
        _jupyter_client.cleanup()
        _jupyter_client = None


# Register global cleanup
atexit.register(cleanup_jupyter_client)


async def execute_python_script(script: Annotated[str, "Python script to execute"]) -> str:
    """
    Execute Python scripts for image processing with deep learning models.
    Enhanced with kernel recovery and better error handling.

    Args:
        script: Python script to execute

    Returns:
        str: Result of the script execution
    """
    try:
        # Get Jupyter client
        client = get_jupyter_client()

        # Add a small delay to ensure kernel is ready
        await asyncio.sleep(0.1)

        try:
            # Execute the script using Jupyter with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(client.execute_code, script),
                timeout=300,  # 5 minutes timeout
            )

            # Check for kernel shutdown warning
            if "parent appears to have exited" in result.lower():
                _log.warning("Jupyter kernel shutdown detected, attempting recovery...")
                # Try once more with a fresh client
                cleanup_jupyter_client()
                client = get_jupyter_client()
                await asyncio.sleep(0.5)  # Wait for initialization
                result = await asyncio.to_thread(client.execute_code, script)

            # Check if there's an error in the result
            if result.startswith("Error executing code:"):
                return f"❌ Script execution failed:\n{result}"

            # Additional output length limit protection
            max_output_length = 8000
            if len(result) > max_output_length:
                truncated_length = max_output_length - 200
                result = (
                    result[:truncated_length]
                    + f"\n\n... [Output truncated, original length: {len(result)} characters] ..."
                )

            # Return successful result
            return (
                f"✅ Script executed successfully:\n{result}"
                if result
                else "✅ Script executed successfully (no output)"
            )

        except asyncio.TimeoutError:
            return "❌ Error executing code: Timeout after 2 minutes"

        except Exception as e:
            error_msg = str(e)
            if "kernel" in error_msg.lower() or "parent appears to have exited" in error_msg.lower():
                _log.warning("Kernel issue detected, attempting recovery...")
                cleanup_jupyter_client()
                return "⚠️ Jupyter kernel restarted due to connection issue. Please try again."

            return f"❌ Error executing code: {error_msg}"

    except Exception as e:
        return f"❌ Failed to get Jupyter client: {str(e)}"


def label_macro():
    """Returns the macro script for human-in-the-loop mask export for current image set."""
    macro_script = textwrap.dedent("""
    macro "CopilotJ Save Each ROI Mask (English)" {

    Dialog.create("CopilotJ HITL Mask Export");
    Dialog.addDirectory("Images directory", "");
    Dialog.addDirectory("Masks output directory", "");
    Dialog.addCheckbox("Export full-size masks (Edit>Fill style)", true);
    Dialog.addCheckbox("Use ROI name for file naming (if available)", true);
    Dialog.show();

    imgsDir   = Dialog.getString();
    masksDir  = Dialog.getString();
    fullSize  = Dialog.getCheckbox();
    useRoiName= Dialog.getCheckbox();

    if (imgsDir=="") imgsDir=getDirectory("Choose images directory");
    if (masksDir=="") masksDir=getDirectory("Choose masks output directory");
    if (imgsDir=="" || masksDir=="") exit("Cancelled.");

    imgsDir  = ensureSlash(imgsDir);
    masksDir = ensureSlash(masksDir);

    File.makeDirectory(masksDir);
    list = getFileList(imgsDir);
    if (list.length==0) exit("No image files found in the directory.");

    exts = newArray(".tif",".tiff",".png",".jpg",".jpeg",
                    ".TIF",".TIFF",".PNG",".JPG",".JPEG");

    for (k=0; k<list.length; k++) {
        if (!endsWithAny(list[k], exts)) continue;

        inPath = imgsDir + list[k];
        if (!File.exists(inPath)) {
            print("⚠️ File not found: " + inPath);
            continue;
        }

        open(inPath);
        title = getTitle();
        getDimensions(W,H,C,Z,T);

        if (!isOpen("ROI Manager")) run("ROI Manager...");
        roiManager("reset");
        roiManager("Show All");

        waitForUser("Current image: " + title + 
            "\n\nUse ROI tools to annotate objects.\n" +
            "Click OK to save masks and continue to the next image.\n" +
            "Click Cancel to stop.");

        if (roiManager("count")==0) { close(); continue; }

        // --- Per-ROI export ---
        imgStem = stripExt(title);
        outDir  = masksDir + imgStem + File.separator;
        File.makeDirectory(outDir);

        n = roiManager("count");
        print("Exporting " + n + " ROI masks → " + outDir);

        selectWindow(title);
        for (i=0; i<n; i++) {
            roiManager("Select", i);

            // ---- filename: ROI name if available, else roi_XXX ----
            idxName = "roi_" + zeroPad3(i+1);
            rname   = Roi.getName();
            if (useRoiName && rname != "") {
                safe = sanitize(rname);
                if (safe != "") idxName = safe;
            }

            if (fullSize) {
                // --- Full-size mask (Edit>Fill on blank canvas) ---
                newImage("__tmp__", "8-bit black", W, H, 1);
                tmpID = getImageID();              // lock the new window
                selectImage(tmpID);
                run("Restore Selection");
                setForegroundColor(255,255,255);
                run("Fill", "slice");

                // enforce strict binary + fill interior on this exact window
                enforce_binary_and_fill_windowID(tmpID);

                // save & close
                selectImage(tmpID);
                saveAs("Tiff", outDir + idxName + ".tif");
                close();
            } else {
                // --- Cropped mask (Create Mask) ---
                run("Restore Selection");
                run("Create Mask");                // opens a new small window and activates it
                maskID = getImageID();             // lock the mask window by ID
                selectImage(maskID);
                rename("__roi_mask__" + zeroPad3(i+1));  // optional: stable name

                // enforce strict binary + fill interior on this exact window
                enforce_binary_and_fill_windowID(maskID);

                // save & close
                selectImage(maskID);
                saveAs("Tiff", outDir + idxName + ".tif");
                close();
            }
        }

        close(); // close original image
        print("✅ Saved ROI masks to: " + outDir);
    }

    showMessage("Done", "All ROI masks have been exported to:\n" + masksDir);
}

// ---- Utility functions ----
function stripExt(s) {
    dot = lastIndexOf(s, ".");
    if (dot>=0) return substring(s, 0, dot);
    else return s;
}
function endsWithAny(s, arr) {
    for (i=0; i<arr.length; i++)
        if (endsWith(s, arr[i])) return true;
    return false;
}
function ensureSlash(path) {
    if (!endsWith(path, "/") && !endsWith(path, "\\")) return path + File.separator;
    return path;
}
function zeroPad3(n) {
    if (n < 10)  return "00"+n;
    if (n < 100) return "0"+n;
    return ""+n;
}
function sanitize(s) {
    s = replace(s, " ", "_");
    s = replace(s, "[/\\\\:\\*\\?\"<>\\|]", "-");
    return s;
}

// Make a specific window strictly binary (0 background, 255 foreground) and solid-fill interior
function enforce_binary_and_fill_windowID(winID) {
    selectImage(winID);
    run("8-bit");
    setOption("BlackBackground", true);
    run("Make Binary");        // -> 0/255
    // run("Close");           // uncomment only if outlines have small gaps
    run("Fill Holes");         // solid white inside
    run("Grays");
}
""")
    return macro_script


if __name__ == "__main__":
    from copilotj.core import load_config

    load_config()

    # Run the test
    # async def test_deep_research():
    #     result = await deep_research("how to analyze time-lapse images of cell migration")
    #     print(result)

    # Run the async test
    # asyncio.run(test_deep_research())
