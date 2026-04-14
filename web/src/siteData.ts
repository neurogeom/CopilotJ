/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

export type Tag =
  | "All"
  | "ImageJ/Fiji"
  | "Python DL"
  | "Segmentation"
  | "3D/Visualization"
  | "Tracking"
  | "Spatial-omics"
  | "Research/Web";

export const GITHUB_REPO = "https://github.com/neurogeom/copilotj";
export const DISCORD_LINK = "https://discord.gg/32wpPq5dBV";

export const SLOGAN =
  "CopilotJ turns natural-language requests into executable, verifiable bioimage analysis workflows by coordinating ImageJ/Fiji ecosystem, Python scientific libraries and deep-learning tools";

export const FEATURES = [
  {
    label: "Intelligent",
    description:
      "Users only need to describe analysis tasks in conversation, and CopilotJ iteratively generates structured, actionable steps to complete end-to-end analyses",
  },
  {
    label: "Efficient",
    description:
      "Compared with external chatbot–assisted ImageJ workflows, CopilotJ improves efficiency by up to 4.3-fold and reduces user interactions by 8.1-fold, while substantially enhancing analytical capability and narrowing the performance gap between beginners and experienced users",
  },
  {
    label: "Reproducible",
    description:
      "Successful runs can be exported as reusable workflows and replayed to reproduce results and scale analyses across datasets with minimal user intervention",
  },
];

export const BACKGROUND = [
  "Quantitative bioimage analysis is indispensable to modern life science research, but ImageJ workflows remain difficult to scale beyond experience-reliant, menu-driven interaction and macro scripting",
  "The ImageJ and Fiji ecosystem offers rich plugin coverage, yet researchers still face fragmented documentation, parameter-heavy tools, and limited access to modern Python and deep-learning workflows",
  "CopilotJ addresses this gap with an LLM-based conversational interface that maps high-level intent to executable ImageJ-centric analysis procedures.",
];

export const CONTRIBUTIONS = [
  {
    title: "Human-in-the-loop (HITL)",
    description:
      "CopilotJ supports human-in-the-loop interaction, allowing users to intervene at critical decision points through familiar GUI operations while maintaining workflow continuity",
  },
  {
    title: "Vision-assisted self-reflection",
    description:
      "CopilotJ integrates feedback from tool outcomes, software states, and visual observations to perform reflective reasoning and generate corrective scripts or revised procedures when needed",
  },
  {
    title: "Self-evolving",
    description:
      "CopilotJ maintains a self-evolving knowledge bank that captures and serializes successful workflows, reusing prior experience to guide planning and tool selection for new tasks",
  },
];

export const SUPPLEMENTARY_VIDEOS = [
  {
    id: 1,
    group: "Foundational operations",
    title: "Basic manipulation",
    youtubeUrl: "https://www.youtube.com/watch?v=QdeXvtg9u_o",
    summary:
      "Executes a multi-step ImageJ workflow including duplication, rotation, contrast adjustment, blur, Otsu thresholding, and edge detection",
  },
  {
    id: 2,
    group: "Foundational operations",
    title: "Video generation",
    youtubeUrl: "https://www.youtube.com/watch?v=9r3M0G6EI0g",
    summary:
      "Generates a 3D visualization video from neuron image data with automatic denoising and contrast enhancement when visibility is poor",
  },
  {
    id: 3,
    group: "Foundational operations",
    title: "Batch post-processing",
    youtubeUrl: "https://www.youtube.com/watch?v=JLcoPdPfYEI",
    summary:
      "Processes folders of microscopy images in batch by adjusting contrast, adding scale bars, and exporting standardized outputs",
  },
  {
    id: 4,
    group: "Core computational imaging",
    title: "3D deconvolution",
    youtubeUrl: "https://www.youtube.com/watch?v=Nc6xSoQmJWI",
    summary:
      "Performs image deconvolution on a Drosophila stack using a provided PSF and exports the restored volume for analysis",
  },
  {
    id: 5,
    group: "Core computational imaging",
    title: "Human-in-the-loop segmentation",
    youtubeUrl: "https://www.youtube.com/watch?v=3yVhDykL6fs",
    summary:
      "Surveys candidate segmentation strategies, lets the user refine the threshold inside ImageJ, and produces the final segmentation under transparent HITL control",
  },
  {
    id: 6,
    group: "Core computational imaging",
    title: "Object tracking",
    youtubeUrl: "https://www.youtube.com/watch?v=6DNCgnTNy48",
    summary:
      "Detects and tracks objects in time-lapse microscopy movies and exports motion statistics together with trajectory visualizations",
  },
  {
    id: 7,
    group: "Domain-specific quantification",
    title: "Colocalization analysis",
    youtubeUrl: "https://www.youtube.com/watch?v=r1N9ctI-5qA",
    summary: "Runs red-green channel colocalization analysis and exports standardized metrics plus analysis outputs",
  },
  {
    id: 8,
    group: "Domain-specific quantification",
    title: "DNA comet assay quantification",
    youtubeUrl: "https://www.youtube.com/watch?v=RyVIedlpw6k",
    summary:
      "Segments comet heads and tails, quantifies tail length and DNA percentage in the tail, and writes overlays and measurements",
  },
  {
    id: 9,
    group: "Domain-specific quantification",
    title: "Gel electrophoresis analysis",
    youtubeUrl: "https://www.youtube.com/watch?v=Y88URkSEi8Y",
    summary:
      "Quantifies lane-wise signal from SDS-PAGE imagery and automatically summarizes results as an interpretable bar chart",
  },
  {
    id: 10,
    group: "Domain-specific quantification",
    title: "Cell scratch assay measurement",
    youtubeUrl: "https://www.youtube.com/watch?v=YHg8DLboKdw",
    summary:
      "Measures wound area at multiple time points and computes wound-closure percentage with a maximum-gap strategy",
  },
  {
    id: 11,
    group: "Multi-stage pipelines",
    title: "Glomerulus quantification and workflow management",
    youtubeUrl: "https://www.youtube.com/watch?v=CY-3c3HCMjo",
    summary:
      "Builds a hybrid ImageJ-plus-Python pipeline for tissue morphology analysis, then saves and replays it as a reusable batch workflow",
  },
  {
    id: 12,
    group: "Multi-stage pipelines",
    title: "Automated calcium imaging analysis",
    youtubeUrl: "https://www.youtube.com/watch?v=q53SQrylGpE",
    summary:
      "Runs motion correction, neuron segmentation, ROI extraction, response alignment, response classification, and summary export for calcium imaging stacks",
  },
  {
    id: 13,
    group: "Benchmarking",
    title: "Comparison with interactively used ImageJ",
    youtubeUrl: "https://www.youtube.com/watch?v=7l5iEjIt3rg",
    summary:
      "Contrasts CopilotJ’s prompt-driven automation with manually executed ImageJ operations in a side-by-side workflow comparison",
  },
  {
    id: 14,
    group: "Deep-learning extensions",
    title: "Image super-resolution",
    youtubeUrl: "https://www.youtube.com/watch?v=kjzNdAof_oY",
    summary:
      "Invokes a BiaPy-based super-resolution model to enhance image clarity and structural detail for downstream analysis",
  },
  {
    id: 15,
    group: "Deep-learning extensions",
    title: "Cell classification",
    youtubeUrl: "https://www.youtube.com/watch?v=sCf8_dEHC1Q",
    summary: "Runs pretrained deep-learning inference to classify cell types and returns structured prediction outputs",
  },
  {
    id: 16,
    group: "Deep-learning extensions",
    title: "Cell profiling with Cellpose",
    youtubeUrl: "https://www.youtube.com/watch?v=nYcLd9dMnow",
    summary:
      "Combines Cellpose segmentation with classical processing steps to generate quantitative cell profiling outputs",
  },
] as const;

export const ORGS = [
  {
    name: "Guangdong Institute of Intelligence Science and Technology",
    logo: "/imgs/logo_gdiist.png",
    link: "https://www.gdiist.cn/",
  },
  { name: "The Hong Kong Polytechnic University", logo: "/imgs/logo_polyu_bme.png", link: "https://www.polyu.edu.hk/" },
  { name: "Beijing Normal University", logo: "/imgs/logo_bnu.png", link: "https://www.bnu.edu.cn/" },
];

export const TOOLS = [
  // Root directory tools
  { name: "Bio-Formats", logo: new URL("./assets/tools/logo_bioformats.png", import.meta.url).href, link: "https://www.openmicroscopy.org/bio-formats/" },
  { name: "BioImage.IO", logo: new URL("./assets/tools/logo_bioimage-io-icon.svg", import.meta.url).href, link: "https://bioimage.io/" },
  { name: "BoneJ", logo: new URL("./assets/tools/logo_bonej.png", import.meta.url).href, link: "https://bonej.org/" },
  { name: "Cellpose", logo: new URL("./assets/tools/logo_cellpose.png", import.meta.url).href, link: "https://www.cellpose.org/" },
  { name: "CLIJ2", logo: new URL("./assets/tools/clij2_logo.png", import.meta.url).href, link: "https://clij.github.io/" },
  { name: "CSBDeep", logo: new URL("./assets/tools/logo_csbdeep.png", import.meta.url).href, link: "https://csbdeep.bioimagecomputing.com/" },
  { name: "DeepImageJ", logo: new URL("./assets/tools/logo_deepimagej.png", import.meta.url).href, link: "https://deepimagej.github.io/" },
  { name: "FLIMLib", logo: new URL("./assets/tools/logo_flimlib.png", import.meta.url).href, link: "https://flimlib.github.io/" },
  { name: "Image.sc Forum", logo: new URL("./assets/tools/logo_imagesc.png", import.meta.url).href, link: "https://forum.image.sc/" },
  { name: "ImgLib2", logo: new URL("./assets/tools/logo_imglib2.png", import.meta.url).href, link: "https://imglib2.net/" },
  { name: "OMERO", logo: new URL("./assets/tools/logo_omero.png", import.meta.url).href, link: "https://www.openmicroscopy.org/omero/" },
  { name: "StarDist", logo: new URL("./assets/tools/logo_stardist.png", import.meta.url).href, link: "https://github.com/stardist/stardist" },
  { name: "TrackMate", logo: new URL("./assets/tools/logo_trackmate.png", import.meta.url).href, link: "https://imagej.net/plugins/trackmate/" },

  // Python tools
  { name: "BiaPy", logo: new URL("./assets/tools/python/logo_biapy.png", import.meta.url).href, link: "https://biapy.readthedocs.io/" },
  { name: "Python", logo: new URL("./assets/tools/python/logo_python.png", import.meta.url).href, link: "https://www.python.org/" },
  { name: "PyTorch", logo: new URL("./assets/tools/python/logo_pytorch.svg", import.meta.url).href, link: "https://pytorch.org/" },
  { name: "TensorFlow", logo: new URL("./assets/tools/python/logo_tensorflow.svg", import.meta.url).href, link: "https://www.tensorflow.org/" },
  { name: "OpenCV", logo: new URL("./assets/tools/python/logo_opencv.png", import.meta.url).href, link: "https://opencv.org/" },

  // Research tools
  { name: "DuckDuckGo", logo: new URL("./assets/tools/research/logo_ddg.svg", import.meta.url).href, link: "https://duckduckgo.com/" },
  { name: "Google", logo: new URL("./assets/tools/research/logo_google.png", import.meta.url).href, link: "https://www.google.com/" },
  { name: "Tavily", logo: new URL("./assets/tools/research/logo_tavily.svg", import.meta.url).href, link: "https://tavily.com/" },
  { name: "Wikipedia", logo: new URL("./assets/tools/research/logo_wikipedia.png", import.meta.url).href, link: "https://www.wikipedia.org/" },

  // // LLM tools
  // { name: "DeepSeek", logo: new URL("./assets/tools/llms/logo_deepseek.png", import.meta.url).href, link: "https://www.deepseek.com/" },
  // { name: "Gemini", logo: new URL("./assets/tools/llms/logo_gemini.jpg", import.meta.url).href, link: "https://gemini.google.com/" },
  // { name: "GLM", logo: new URL("./assets/tools/llms/logo_glm.svg", import.meta.url).href, link: "https://zhipuai.cn/" },
  // { name: "GPT", logo: new URL("./assets/tools/llms/logo_gpt.png", import.meta.url).href, link: "https://openai.com/" },
  // { name: "Ollama", logo: new URL("./assets/tools/llms/logo_ollama.webp", import.meta.url).href, link: "https://ollama.com/" },
];
