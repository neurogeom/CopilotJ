# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
import json
from typing import Annotated, Iterable, Mapping, MutableMapping
from pathlib import Path

import bs4
from ddgs import DDGS
import requests
import selenium.webdriver as webdriver
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tavily import TavilyClient

from copilotj.core import load_env, new_langchain_openai_embeddings
from copilotj.multiagent.py_tools import get_project_temp_dir
from copilotj.multiagent.tools import execute_python_script

__all__ = [
    "ddg_search",
    "tavily_search",
    "wikipedia_search",
    "imagesc_search",
    "biii_search",
    "ImageJRetriever",
    "deep_research",
    "download_resource",
    "bioimage_search_models",
    "bioimage_get_model_info",
    "bioimage_download_model",
]

# BioImage Model Zoo constants
DEFAULT_COLLECTION_URL = os.getenv(
    "BIOIMAGE_MODEL_ZOO_URL",
    "https://bioimage-io.github.io/collection-bioimage-io/collection.json",
)
FALLBACK_COLLECTION_URLS = [
    "https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/main/collection.json",
    "https://bioimage.io/collection.json",
]
DEFAULT_CACHE_DIR = Path(os.getenv("BIOIMAGE_MODEL_ZOO_CACHE", Path(__file__).resolve().parent.parent.parent / "temp" / "bioimage_model_zoo")).resolve()
DEFAULT_CACHE_TTL = int(os.getenv("BIOIMAGE_MODEL_ZOO_CACHE_TTL", "86400"))


# BioImage Model Zoo helper functions
def _unwrap_collection(payload: object) -> list[Mapping]:
    """Extract the list of resources from various collection layouts."""
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, Mapping):
        raise ValueError("Unexpected collection payload type")

    candidate_keys = [
        "resources",
        "collection",
        "models",
        "items",
        "entries",
    ]
    for key in candidate_keys:
        if key in payload:
            data = payload[key]
            if isinstance(data, list):
                return data
            if isinstance(data, Mapping):
                nested = data.get("resources") or data.get("items")
                if isinstance(nested, list):
                    return nested

    raise ValueError("No resources found in collection payload")


def _load_cached_collection(cache_file: Path, ttl_seconds: int) -> list[Mapping] | None:
    if not cache_file.exists():
        return None
    age = time.time() - cache_file.stat().st_mtime
    if age > ttl_seconds:
        return None
    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return _unwrap_collection(payload)
    except Exception:
        pass
    return None


def _fetch_collection(
    *,
    base_url: str | Iterable[str] = DEFAULT_COLLECTION_URL,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    ttl_seconds: int = DEFAULT_CACHE_TTL,
    force_refresh: bool = False,
    session: requests.Session | None = None,
) -> list[Mapping]:
    """Return the BioImage Model Zoo collection as a list of resource dicts."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "collection.json"

    if not force_refresh:
        cached = _load_cached_collection(cache_file, ttl_seconds)
        if cached is not None:
            return cached

    urls: list[str] = []
    if isinstance(base_url, str):
        urls.append(base_url)
    else:
        urls.extend(list(base_url))
    urls.extend([u for u in FALLBACK_COLLECTION_URLS if u not in urls])

    last_error: Exception | None = None
    sess = session or requests.Session()
    for url in urls:
        try:
            resp = sess.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            with cache_file.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            return _unwrap_collection(payload)
        except Exception as exc:
            last_error = exc

    if last_error:
        raise last_error
    raise RuntimeError("Failed to fetch collection: no URLs available")


def _text_matches(haystack: Iterable[str], needle: str) -> bool:
    needle_lower = needle.lower()
    return any(needle_lower in (item or "").lower() for item in haystack)


def _normalize_list(values: Iterable[str] | None) -> list[str]:
    return [str(v).strip() for v in values or [] if str(v).strip()]


def _get_model(models: list[Mapping], model_id_or_name: str) -> Mapping | None:
    """Locate a model by id or name from a collection list."""
    target = model_id_or_name.lower()
    for entry in models:
        if str(entry.get("id", "")).lower() == target:
            return entry
        if str(entry.get("name", "")).lower() == target:
            return entry
    return None


def _extract_download_urls(entry: Mapping) -> list[str]:
    """Pull likely download URLs from heterogeneous RDF entries."""
    urls: list[str] = []
    candidate_keys = [
        "download_url",
        "downloadUrl",
        "download",
        "source",
        "sources",
        "rdf_source",
        "rdfSource",
    ]

    for key in candidate_keys:
        value = entry.get(key)
        if isinstance(value, str):
            if value.startswith("http"):
                urls.append(value)
        elif isinstance(value, list):
            urls.extend([v for v in value if isinstance(v, str) and v.startswith("http")])
        elif isinstance(value, Mapping):
            for item in value.values():
                if isinstance(item, str) and item.startswith("http"):
                    urls.append(item)

    weights = entry.get("weights")
    if isinstance(weights, Mapping):
        for fmt in weights.values():
            if isinstance(fmt, Mapping):
                source = fmt.get("source") or fmt.get("url")
                if isinstance(source, str) and source.startswith("http"):
                    urls.append(source)

    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


# search tool
def ddg_search(
    query: Annotated[str, "Search query string describing what information you need"],
    *,
    max_results: Annotated[int, "Maximum number of results to return"] = 3,
    timeout: Annotated[int, "Timeout for search request (s)"] = 10,
    retries: Annotated[int, "Number of retry attempts"] = 2,
    images: Annotated[bool, "Set True to return image results instead of text"] = False,
    public_only: Annotated[bool, "Bias towards Wikimedia/Flickr/Unsplash if True"] = False,
    safesearch: Annotated[str, "DuckDuckGo safesearch: off | moderate | strict"] = "moderate",
) -> str:
    attempt = 0
    while attempt <= retries:
        try:
            with DDGS(timeout=timeout) as ddgs:
                if images:
                    sites = [
                        "site:commons.wikimedia.org",
                        "site:upload.wikimedia.org",
                        "site:wikimedia.org",
                        "site:flickr.com",
                        "site:unsplash.com",
                        "site:staticflickr.com",
                    ]
                    q = f"{query} (photo OR image) {' '.join(sites)}" if public_only else f"{query} (photo OR image)"
                    items = list(ddgs.images(q, max_results=max_results, safesearch=safesearch))
                    if not items:
                        return "No image results."
                    out = []
                    for i, it in enumerate(items, 1):
                        img = it.get("image") or it.get("thumbnail")
                        out.append(f"{i}. {it.get('title')}\nPage: {it.get('url')}\nImage: {img}")
                    return "\n\n".join(out)
                else:
                    results = ddgs.text(query, max_results=max_results)
                    if not results:
                        return "No text results."
                    out = []
                    for i, r in enumerate(results, 1):
                        out.append(f"{i}. {r.get('title')}\n{r.get('body')}\nSource: {r.get('href')}")
                    return "\n\n".join(out)
        except Exception as e:
            attempt += 1
            if attempt > retries:
                return f"DDG search failed: {e}"
            time.sleep(1 * attempt)


def tavily_search(
    query: Annotated[str, "Search query string describing what information you need"],
    *,
    max_results: Annotated[int, "Maximum number of search results to return"] = 5,
    include_answer: Annotated[bool, "Whether to include AI-generated summary"] = True,
    include_raw_content: Annotated[bool, "Whether to include raw content from sources"] = False,
) -> str | list[dict[str, str]]:
    try:
        load_env()
        tavily_api_key = os.getenv("COPILOTJ_TAVILY_API_KEY")
        if not tavily_api_key:
            return "Tavily API key not found. Please set COPILOTJ_TAVILY_API_KEY environment variable."

        client = TavilyClient(api_key=tavily_api_key)

        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            search_depth="advanced",
        )

        if include_answer and response.get("answer"):
            result = f"**Summary:** {response['answer']}\n\n"
        else:
            result = ""

        if response.get("results"):
            result += "**Sources:**\n"
            for i, item in enumerate(response["results"], 1):
                title = item.get("title", "No title")
                url = item.get("url", "No URL")
                content = (
                    item.get("content", "No content")[:500] + "..."
                    if len(item.get("content", "")) > 500
                    else item.get("content", "")
                )
                result += f"{i}. **{title}**\n   {content}\n   Source: {url}\n\n"

        return result or "No results found for the Tavily search query."

    except Exception as e:
        return f"Tavily search failed: {str(e)}"


def wikipedia_search(query: Annotated[str, "Search query string describing what information you need"]):
    """Wikipedia Search Tool."""
    try:
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        page = wikipedia.run(query)
        return page if page else "No Wikipedia page found for query."
    except Exception as e:
        return f"Error accessing Wikipedia: {str(e)}"


def imagesc_search(
    query: Annotated[str, "Search query string describing what information you need"],
    timeout: Annotated[int, "Timeout for direct forum crawling before Tavily fallback"] = 15,
    limit: int = 3,
) -> str:
    base_url = "https://forum.image.sc/search?q="
    search_url = base_url + query.replace(" ", "%20")

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(timeout)

        print(f"🔍 Attempting direct crawl of Image.sc forum (timeout: {timeout}s)...")
        driver.get(search_url)

        time.sleep(2)

        soup = bs4.BeautifulSoup(driver.page_source, "html.parser")

        results = []
        for result in soup.find_all("a", class_="search-link", limit=limit):
            title = result.get_text(strip=True)
            link = "https://forum.image.sc" + result["href"]
            results.append({"title": title, "link": link})

        if not results:
            return "No direct crawl results found."

        detailed_results = []
        for idx, result in enumerate(results, start=1):
            try:
                driver.get(result["link"])
                time.sleep(1)

                page_soup = bs4.BeautifulSoup(driver.page_source, "html.parser")
                content_div = page_soup.find("div", class_="cooked")
                if content_div:
                    content = content_div.get_text(strip=True)
                    detailed_results.append(
                        f"Result {idx}: {result['title']}\n{content[:500]}...\nLink: {result['link']}"
                    )
                else:
                    detailed_results.append(
                        f"Result {idx}: {result['title']}\nUnable to extract content.\nLink: {result['link']}"
                    )
            except Exception:
                detailed_results.append(
                    f"Result {idx}: {result['title']}\nError loading content.\nLink: {result['link']}"
                )

        return "\n\n".join(detailed_results)

    except Exception as e:
        raise Exception(f"Crawling failed: {str(e)}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def biii_search(query: str, timeout: int = 10) -> str:
    return "biii search tool is not available, skip this tool."  # TODO
    search_url = "https://biii.eu/search?search_api_fulltext=" + query.replace(" ", "%20")

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(timeout)

        print(f"🔍 Attempting direct crawl of bio-image.io (timeout: {timeout}s)...")
        driver.get(search_url)
        time.sleep(2)

        soup = bs4.BeautifulSoup(driver.page_source, "html.parser")

        # Find search results - try multiple selectors for bio-image.io
        results = []
        search_elements = soup.find_all("a", class_="tool-link", limit=3)

        if not search_elements:
            # Try alternative selectors for different site structures
            search_elements = soup.find_all("a", href=True, limit=5)
            search_elements = [
                elem for elem in search_elements if query.lower().replace(" ", "-") in elem.get("href", "").lower()
            ][:3]

        for element in search_elements:
            title = element.get_text(strip=True) or "No title"
            href = element.get("href", "")

            # Handle relative URLs
            if href.startswith("/"):
                link = f"https://biii.eu{href}"
            elif href.startswith("http"):
                link = href
            else:
                continue

            results.append({"title": title, "link": link})

        if not results:
            return "No direct crawl results found on bio-image.io."

        detailed_results = []
        for idx, result in enumerate(results, start=1):
            try:
                driver.get(result["link"])
                time.sleep(1)

                page_soup = bs4.BeautifulSoup(driver.page_source, "html.parser")

                # Try multiple content selectors for bio-image.io
                content_classes = [
                    "tool-description",
                    "content",
                    "cooked",
                    "post-content",
                    "article-content",
                    "main-content",
                    "entry-content",
                ]

                content = "Unable to extract content."
                for content_class in content_classes:
                    content_div = page_soup.find("div", class_=content_class)
                    if content_div:
                        content = content_div.get_text(strip=True)[:500] + "..."
                        break

                detailed_results.append(f"Result {idx}: {result['title']}\n{content}\nLink: {result['link']}")

            except Exception:
                detailed_results.append(
                    f"Result {idx}: {result['title']}\nError loading content.\nLink: {result['link']}"
                )

        return "\n\n".join(detailed_results)

    except Exception as e:
        raise Exception(f"Crawling bio-image.io failed: {str(e)}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


# RAG tool
class ImageJRetriever:
    def __init__(self):
        self._retriever = _load_kb_retriever()

    def retrieve(self, query: str) -> str:
        try:
            retriever = self._retriever
            try:
                relevant_docs = retriever.invoke(query)

                contents = [doc.page_content for doc in relevant_docs]
                if contents:
                    vect = TfidfVectorizer()
                    X = vect.fit_transform([query] + contents)
                    sims = cosine_similarity(X[0], X[1:]).ravel()
                else:
                    sims = []

                reference_texts = []
                for i, doc in enumerate(relevant_docs):
                    score = doc.metadata.get("score")
                    if score is None and i < len(sims):
                        score = sims[i]
                    if score is None:
                        score = 0.0

                    reference_texts.append(
                        f"Document {i + 1} (Relevance Score: {score:.2f}):\n"
                        f"Content: {doc.page_content}\n"
                        f"Metadata: {doc.metadata}"
                    )

                reference_text = "\n\n".join(reference_texts)
                return (
                    f"Found {len(reference_texts)} most relevant documents from the knowledge base:\n\n{reference_text}"
                )

            except Exception as e:
                return f"Error searching knowledge base: {str(e)}"

        except Exception as e:
            return f"Failed to retrieve relevant documents: {str(e)}"


def _load_kb_retriever(*, max_results: int = 10) -> VectorStoreRetriever:
    try:
        kb_path = "./assets/knowledge_base"
        pkl_files = []
        for root, _, files in os.walk(kb_path):
            for file in files:
                if file.endswith(".pkl"):
                    pkl_files.append(os.path.join(root, file))

        if not pkl_files:
            raise Exception("No .pkl files found in knowledge base directory")

        retrievers = []
        for pkl_file in pkl_files:
            try:
                collection_name = os.path.splitext(os.path.basename(pkl_file))[0]
                vector_store = _load_docs_store(os.path.dirname(pkl_file), collection_name)
                retriever = vector_store.as_retriever(search_kwargs={"k": max_results})
                retrievers.append((retriever, collection_name))
            except Exception:
                continue

        if not retrievers:
            raise Exception("No valid collections found in knowledge base")

        if len(retrievers) == 1:
            return retrievers[0][0]

        return retrievers

    except Exception:
        raise


def _load_docs_store(db_path: str, collection_name: str) -> FAISS:
    try:
        store_path = os.path.join(db_path, f"{collection_name}")

        if not os.path.exists(f"{store_path}.faiss") or not os.path.exists(f"{store_path}.pkl"):
            raise FileNotFoundError(f"Docs store {collection_name} not found. Run `python scripts/rag.py` first.")

        embeddings = new_langchain_openai_embeddings(api_key=os.getenv("COPILOTJ_RAG_API_KEY"))

        try:
            vector_store = FAISS.load_local(
                index_name=collection_name,
                folder_path=db_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            return vector_store
        except Exception as e:
            raise Exception("Failed to load docs store.") from e
    except Exception:
        raise


async def imagej_retriever(query: Annotated[str, "Search query string describing what information you need"]) -> str:
    """Retrieve relevant documents from the ImageJ knowledge base.

    Args:
        query (str): The search query to find relevant documents.

    Returns:
        str: Retrieved documents and their metadata.
    """
    retriever = ImageJRetriever()
    return retriever.retrieve(query)


# deep research tool
async def deep_research(query: Annotated[str, "The research question to investigate thoroughly"]) -> str:
    research_report = {
        "query": query,
        "sources_consulted": [],
        "findings": {},
        "cross_validation": {},
        "synthesis": "",
        "recommendations": [],
    }

    try:
        kb_results = await imagej_retriever(query)
        research_report["sources_consulted"].append("ImageJ Knowledge Base")
        research_report["findings"]["knowledge_base"] = kb_results

        tavily_results = tavily_search(query, max_results=5, include_answer=True)
        research_report["sources_consulted"].append("Tavily AI Search")
        research_report["findings"]["web_current"] = tavily_results

        imagesc_results = imagesc_search(query)
        research_report["sources_consulted"].append("Image.sc Forum")
        research_report["findings"]["community"] = imagesc_results

        wiki_results = wikipedia_search(query)
        research_report["sources_consulted"].append("Wikipedia")
        research_report["findings"]["reference"] = wiki_results

        ddg_results = ddg_search(query, max_results=5)
        research_report["sources_consulted"].append("DuckDuckGo Search")
        research_report["findings"]["web_alternative"] = ddg_results

        return _format_research_with_prompt(research_report)

    except Exception as e:
        return f"❌ Deep research failed: {str(e)}, please skip this tool and continue with other tools."


def _format_research_with_prompt(research_report: dict) -> str:
    findings = research_report["findings"]
    sources = research_report["sources_consulted"]

    research_data = f"""
## Raw Research Data for: "{research_report["query"]}"
**Sources Consulted**: {", ".join(sources)} ({len(sources)} total sources)
### Research Findings by Source:
"""
    if "knowledge_base" in findings:
        research_data += f"""
**📚 Local Knowledge Base:**
{findings["knowledge_base"]}
"""
    if "web_current" in findings:
        research_data += f"""
**🌐 Current Web Information (Tavily AI):**
{findings["web_current"]}
"""
    if "community" in findings:
        research_data += f"""
**💬 Community Discussions (Image.sc Forum):**
{findings["community"]}
"""
    if "reference" in findings:
        research_data += f"""
**📖 Reference Information (Wikipedia):**
{findings["reference"]}
"""
    if "web_alternative" in findings:
        research_data += f"""
**🔍 Additional Web Search (DuckDuckGo):**
{findings["web_alternative"]}
"""
    formatting_prompt = f"""
Please focus on the Query: {research_report["query"]} to extract the most related information and insights. Cite the links and sources in the response.
"""
    return research_data + formatting_prompt


# download tool
async def download_resource(code: Annotated[str, "The code to download the resource"]) -> str:
    """Download a resource from the given URL using execute_python_script."""
    try:
        download_dir = get_project_temp_dir("downloads")
        download_dir.mkdir(parents=True, exist_ok=True)

        result = await execute_python_script(code)

        return f"🔗 Download attempt for: {code}\n📁 Target directory: {download_dir}\n\nScript Output:\n{result}"

    except Exception as e:
        return f"❌ Error downloading with code: {code}\n{str(e)}"


# BioImage Model Zoo tools
def bioimage_search_models(
    query: Annotated[str | None, "Free-text search query for model name, description, or keywords"] = None,
    tags: Annotated[list[str] | None, "List of tags to filter by (e.g., ['denoising', 'segmentation'])"] = None,
    authors: Annotated[list[str] | None, "List of author names to filter by"] = None,
    limit: Annotated[int, "Maximum number of results to return"] = 10,
) -> str:
    """Search BioImage Model Zoo for pre-trained models.
    
    Returns formatted list of models with name, ID, version, description, tags, and authors.
    Use this to find models for specific tasks like denoising, segmentation, detection, etc.
    """
    try:
        # Fetch collection
        models = _fetch_collection(
            base_url=DEFAULT_COLLECTION_URL,
            cache_dir=DEFAULT_CACHE_DIR,
            ttl_seconds=DEFAULT_CACHE_TTL,
            force_refresh=False,
        )

        query_text = query.lower().strip() if query else None
        tags_set = {t.lower() for t in _normalize_list(tags)}
        author_set = {a.lower() for a in _normalize_list(authors)}

        def matches(entry: Mapping) -> bool:
            name = str(entry.get("name", ""))
            description = str(entry.get("description", ""))
            entry_tags = [str(t) for t in entry.get("tags", [])]
            entry_authors = []
            raw_authors = entry.get("authors") or entry.get("maintainers")
            if isinstance(raw_authors, list):
                for author in raw_authors:
                    if isinstance(author, Mapping) and "name" in author:
                        entry_authors.append(str(author.get("name")))
                    else:
                        entry_authors.append(str(author))

            if query_text and not _text_matches([name, description, " ".join(entry_tags)], query_text):
                return False
            if tags_set and not tags_set.issubset({t.lower() for t in entry_tags}):
                return False
            if author_set and not author_set.intersection({a.lower() for a in entry_authors}):
                return False
            return True

        results: list[MutableMapping] = []
        for entry in models:
            if len(results) >= limit:
                break
            if not matches(entry):
                continue
            summary: MutableMapping[str, object] = {
                "id": entry.get("id"),
                "name": entry.get("name"),
                "version": entry.get("version") or entry.get("latest_version"),
                "description": (entry.get("description") or "").strip(),
                "tags": entry.get("tags", []),
                "authors": entry.get("authors") or entry.get("maintainers"),
            }
            results.append(summary)
        
        if not results:
            return "No models found matching the search criteria."
        
        output = []
        for idx, model in enumerate(results, 1):
            output.append(
                f"[{idx}] {model.get('name', 'Unknown')}\n"
                f"    ID: {model.get('id', 'N/A')}\n"
                f"    Version: {model.get('version', 'N/A')}\n"
                f"    Description: {model.get('description', 'No description')}\n"
                f"    Tags: {', '.join(str(t) for t in model.get('tags', []))}\n"
            )
        
        return "\n".join(output)
    except Exception as e:
        return f"Error searching BioImage Model Zoo: {str(e)}"


def bioimage_get_model_info(
    model_id: Annotated[str, "Model ID or name to get detailed information for"],
) -> str:
    """Get detailed metadata for a specific BioImage Model Zoo model.
    
    Returns comprehensive information including description, authors, tags, download URLs, etc.
    """
    try:
        models = _fetch_collection(force_refresh=False)
        entry = _get_model(models, model_id)
        if not entry:
            return f"Model '{model_id}' not found in BioImage Model Zoo."
        
        # Format the detailed model information
        info = []
        info.append(f"Name: {entry.get('name', 'Unknown')}")
        info.append(f"ID: {entry.get('id', 'N/A')}")
        info.append(f"Version: {entry.get('version') or entry.get('latest_version') or 'N/A'}")
        info.append(f"Description: {entry.get('description', 'No description')}")
        info.append(f"Tags: {', '.join(str(t) for t in entry.get('tags', []))}")
        
        # Authors
        authors = entry.get('authors') or entry.get('maintainers')
        if authors:
            author_names = []
            for author in authors:
                if isinstance(author, dict) and 'name' in author:
                    author_names.append(author['name'])
                else:
                    author_names.append(str(author))
            info.append(f"Authors: {', '.join(author_names)}")
        
        # Download URLs
        urls = _extract_download_urls(entry)
        if urls:
            info.append(f"Download URL: {urls[0]}")
        
        return "\n".join(info)
    except Exception as e:
        return f"Error getting model info: {str(e)}"


def bioimage_download_model(
    model_id: Annotated[str, "Model ID or name to download"],
    dest_dir: Annotated[str | None, "Optional destination directory path (defaults to project assets/bioimage_models)"] = None,
) -> str:
    """Download a BioImage Model Zoo model archive.
    
    Returns the local file path where the model was downloaded.
    """
    try:
        if dest_dir:
            dest_path = Path(dest_dir)
        else:
            dest_path = Path(__file__).resolve().parent.parent.parent / "assets" / "bioimage_models"
        
        dest_path.mkdir(parents=True, exist_ok=True)

        models = _fetch_collection(
            base_url=DEFAULT_COLLECTION_URL,
            cache_dir=DEFAULT_CACHE_DIR,
            ttl_seconds=DEFAULT_CACHE_TTL,
            force_refresh=False,
        )

        entry = _get_model(models, model_id)
        if entry is None:
            raise ValueError(f"Model '{model_id}' not found in collection")

        urls = _extract_download_urls(entry)
        if not urls:
            raise ValueError(f"Model '{model_id}' does not expose a download URL")

        chosen = urls[0]
        filename_guess = Path(chosen.split("?")[0]).name or f"{entry.get('id', 'model')}.zip"
        result_path = dest_path / filename_guess

        sess = requests.Session()
        with sess.get(chosen, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with result_path.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    handle.write(chunk)
        
        return f"✅ Model downloaded successfully to: {result_path}"
    except Exception as e:
        return f"❌ Error downloading model: {str(e)}"


if __name__ == "__main__":
    from copilotj.core import load_env

    load_env()

    result = ddg_search("images of albert einstein", images=True)
    print(result)
