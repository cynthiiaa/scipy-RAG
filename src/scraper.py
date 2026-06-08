"""
SciPy Documentation Scraper

Utilities for scraping SciPy documentation from the official website for use in a RAG system.

Workshop goals:
- Keep the code readable and reasonably robust (retries, timeouts, polite rate limiting)
- Capture provenance (retrieved_at, scipy_doc_version) so datasets can be refreshed later
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urldefrag
import urllib.robotparser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


@dataclass
class ScrapedDocument:
    """Represents a scraped documentation page."""
    url: str
    title: str
    module: str
    function_name: str
    signature: str
    description: str
    parameters: str
    returns: str
    examples: str
    full_text: str
    doc_type: str  # "function", "class", "module", "tutorial"
    retrieved_at: str = ""
    scipy_doc_version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SciPyDocsScraper:
    """
    Scraper for SciPy official documentation.

    Notes:
    - Uses robots.txt checks and a polite delay.
    - Uses lightweight retry/backoff for transient network errors (429/5xx).
    """

    BASE_URL = "https://docs.scipy.org/doc/scipy/reference/"

    MODULES = [
        "cluster",
        "constants",
        "fft",
        "integrate",
        "interpolate",
        "io",
        "linalg",
        "ndimage",
        "optimize",
        "signal",
        "sparse",
        "spatial",
        "special",
        "stats",
    ]

    def __init__(
        self,
        delay: float = 0.5,
        output_dir: str = "data/raw",
        user_agent: str = "SciPyRAGWorkshop/1.0 (Educational; contact: workshop@example.com)",
        timeout_s: float = 10.0
    ):
        """
        Args:
            delay: Seconds to wait between requests (be polite!)
            output_dir: Directory to save scraped data
            user_agent: User agent string for requests
            timeout_s: Per-request timeout
        """
        self.delay = float(delay)
        self.timeout_s = float(timeout_s)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Requests session + retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        retry = Retry(
            total=5,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Robots.txt
        self._robot_parser = urllib.robotparser.RobotFileParser()
        self._robot_parser.set_url(urljoin(self.BASE_URL, "../robots.txt"))
        try:
            self._robot_parser.read()
        except Exception:
            # If robots.txt fetch fails, we still scrape politely, but don't claim robots compliance.
            self._robot_parser = None

    def _allowed_by_robots(self, url: str) -> bool:
        if self._robot_parser is None:
            return True
        try:
            return self._robot_parser.can_fetch(self.session.headers.get("User-Agent", "*"), url)
        except Exception:
            return True

    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a page; returns BeautifulSoup or None."""
        if not self._allowed_by_robots(url):
            print(f"Blocked by robots.txt: {url}")
            return None

        try:
            resp = self.session.get(url, timeout=self.timeout_s)
            resp.raise_for_status()
            time.sleep(self.delay)
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    from urllib.parse import urldefrag, urljoin, urlparse

    def get_module_function_urls(self, module: str) -> list[str]:
        """Get all generated function/class URLs for a SciPy module page."""
        module_url = f"{self.BASE_URL}{module}.html"
        soup = self.get_page(module_url)
        if not soup:
            return []

        urls: list[str] = []
        seen: set[str] = set()

        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Remove fragments like #scipy.integrate.quad
            href, _fragment = urldefrag(href)

            full_url = urljoin(module_url, href)
            path = urlparse(full_url).path

            is_generated_doc = "/generated/" in path
            is_same_module = f"scipy.{module}." in path
            is_html = path.endswith(".html")

            if is_generated_doc and is_same_module and is_html and full_url not in seen:
                seen.add(full_url)
                urls.append(full_url)

        return urls

    def _extract_doc_version(self, soup: BeautifulSoup) -> str:
        """
        Best-effort extraction of SciPy doc version from the page.
        SciPy's HTML structure can change; treat this as a helpful hint, not a guarantee.
        """
        # Common places: version switcher / footer / meta
        # Try meta first
        meta = soup.find("meta", attrs={"name": "doc-version"})
        if meta and meta.get("content"):
            return meta["content"].strip()

        # Try visible "version" labels
        for selector in [
            ("span", {"class": re.compile(r"version", re.I)}),
            ("div", {"class": re.compile(r"version", re.I)}),
        ]:
            el = soup.find(selector[0], attrs=selector[1])
            if el:
                txt = el.get_text(" ", strip=True)
                # Extract something like 1.17.0
                m = re.search(r"\b\d+\.\d+(\.\d+)?\b", txt)
                if m:
                    return m.group(0)

        return ""

    def parse_function_page(self, url: str) -> Optional[ScrapedDocument]:
        """Parse an individual generated function page."""
        soup = self.get_page(url)
        if not soup:
            return None

        # Extract function name from URL: scipy.optimize.minimize.html -> scipy.optimize.minimize
        path = urlparse(url).path
        func_full_name = path.split("/")[-1].replace(".html", "")
        parts = func_full_name.split(".")
        if len(parts) < 3:
            return None

        module = parts[1]
        function_name = ".".join(parts[2:])

        # Find main content (Sphinx theme varies)
        main_content = soup.find("div", class_="body") or soup.find("main") or soup

        # Title
        title_elem = main_content.find("h1")
        title = title_elem.get_text(strip=True) if title_elem else func_full_name

        # Signature (Sphinx)
        signature = ""
        sig_elem = main_content.find("dt", class_="sig")
        if sig_elem:
            signature = sig_elem.get_text(strip=True)
        else:
            # Fallback: first <pre> containing name
            sig_pre = main_content.find("pre")
            if sig_pre and func_full_name in sig_pre.get_text():
                signature = sig_pre.get_text(strip=True)

        # Description (first paragraph in definition)
        description = ""
        dd_elem = main_content.find("dd")
        if dd_elem:
            first_p = dd_elem.find("p")
            if first_p:
                description = first_p.get_text(strip=True)

        parameters = self._extract_section(main_content, ["Parameters", "Args"])
        returns = self._extract_section(main_content, ["Returns", "Return"])
        examples = self._extract_section(main_content, ["Examples", "Example"])

        # Clean text while preserving code block formatting
        full_text = self._clean_text_preserve_code(main_content)

        doc_type = "function"
        if "class" in title.lower() or signature.startswith("class "):
            doc_type = "class"

        retrieved_at = datetime.now(timezone.utc).isoformat()
        scipy_doc_version = self._extract_doc_version(soup)

        return ScrapedDocument(
            url=url,
            title=title,
            module=f"scipy.{module}",
            function_name=function_name,
            signature=signature,
            description=description,
            parameters=parameters,
            returns=returns,
            examples=examples,
            full_text=full_text[:10000],  # keep workshop runs light
            doc_type=doc_type,
            retrieved_at=retrieved_at,
            scipy_doc_version=scipy_doc_version,
        )

    def _clean_text_preserve_code(self, soup: BeautifulSoup) -> str:
        """
        Extract text from soup, preserving code block formatting.

        - Code blocks (<pre>, .highlight) keep their original whitespace
        - Other text gets whitespace normalized
        """
        import copy
        soup_copy = copy.copy(soup)

        # Find all code blocks and replace with placeholders
        code_blocks = []
        for i, code_elem in enumerate(soup_copy.find_all(['pre', 'div'], class_=lambda c: c and ('highlight' in c if isinstance(c, str) else any('highlight' in cls for cls in c)))):
            # Preserve original formatting in code blocks
            code_text = code_elem.get_text()
            code_blocks.append(code_text)
            placeholder = f"__CODE_BLOCK_{i}__"
            code_elem.replace_with(placeholder)

        # Also handle <pre> tags not caught above
        for i, pre in enumerate(soup_copy.find_all('pre'), start=len(code_blocks)):
            code_text = pre.get_text()
            code_blocks.append(code_text)
            placeholder = f"__CODE_BLOCK_{i}__"
            pre.replace_with(placeholder)

        # Extract remaining text with whitespace normalization
        text = soup_copy.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text)

        # Restore code blocks
        for i, code in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            text = text.replace(placeholder, f"\n```\n{code}\n```\n")

        return text.strip()

    def _extract_section(self, soup: BeautifulSoup, headers: list[str]) -> str:
        """Extract a named section from Sphinx docs (best effort)."""
        for header in headers:
            section = soup.find("p", class_="rubric", string=re.compile(header, re.I))
            if section:
                content: list[str] = []
                for sibling in section.find_next_siblings():
                    if sibling.name == "p" and "rubric" in (sibling.get("class", []) or []):
                        break
                    # Preserve code formatting in sections
                    if sibling.name == "pre" or (sibling.get("class") and any("highlight" in c for c in sibling.get("class", []))):
                        content.append(sibling.get_text())
                    else:
                        text = sibling.get_text(separator=" ", strip=True)
                        content.append(re.sub(r'\s+', ' ', text))
                return "\n".join(content)

            dt = soup.find("dt", string=re.compile(f"^{header}", re.I))
            if dt:
                dd = dt.find_next_sibling("dd")
                if dd:
                    return self._clean_text_preserve_code(dd)

        return ""

    def scrape_module(self, module: str) -> list[ScrapedDocument]:
        print(f"\nScraping scipy.{module}...")
        urls = self.get_module_function_urls(module)
        print(f"Found {len(urls)} function pages")

        documents: list[ScrapedDocument] = []
        for url in tqdm(urls, desc=f"scipy.{module}"):
            doc = self.parse_function_page(url)
            if doc:
                documents.append(doc)
        return documents

    def scrape_all(self, modules: Optional[list[str]] = None) -> list[ScrapedDocument]:
        modules = modules or self.MODULES
        all_documents: list[ScrapedDocument] = []

        for module in modules:
            docs = self.scrape_module(module)
            all_documents.extend(docs)
            self.save_documents(docs, f"scipy_{module}.json")

        self.save_documents(all_documents, "scipy_all.json")
        return all_documents

    def save_documents(self, documents: list[ScrapedDocument], filename: str) -> None:
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([doc.to_dict() for doc in documents], f, indent=2)
        print(f"Saved {len(documents)} documents to {output_path}")

    def load_documents(self, filename: str) -> list[ScrapedDocument]:
        input_path = self.output_dir / filename
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ScrapedDocument(**doc) for doc in data]


def create_sample_dataset() -> list[dict]:
    """
    Create a sample dataset for workshop/testing without scraping.
    """
    return [
        {
            "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html",
            "title": "scipy.optimize.minimize",
            "module": "scipy.optimize",
            "function_name": "minimize",
            "signature": "scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)",
            "description": "Minimization of scalar function of one or more variables.",
            "parameters": "fun : callable\n    The objective function to be minimized.\n"
                         "x0 : ndarray, shape (n,)\n    Initial guess.\n"
                         "method : str or callable, optional\n    Type of solver.\n"
                         "bounds : sequence or Bounds, optional\n    Bounds on variables.",
            "returns": "res : OptimizeResult\n    The optimization result as an OptimizeResult object.",
            "examples": ">>> from scipy.optimize import minimize\n>>> from scipy.optimize import rosen\n>>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]\n>>> res = minimize(rosen, x0, method=\"Nelder-Mead\", tol=1e-6)\n>>> res.x\narray([1., 1., 1., 1., 1.])",
            "full_text": "scipy.optimize.minimize - Minimization of scalar function of one or more variables...",
            "doc_type": "function",
            "retrieved_at": "",
            "scipy_doc_version": "",
        },
        {
            "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html",
            "title": "scipy.optimize.curve_fit",
            "module": "scipy.optimize",
            "function_name": "curve_fit",
            "signature": "scipy.optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, bounds=(-inf, inf), method=None, jac=None, *, full_output=False, nan_policy=None)",
            "description": "Use non-linear least squares to fit a function, f, to data.",
            "parameters": "f : callable\n    The model function.\n"
                         "xdata : array_like\n    The independent variable.\n"
                         "ydata : array_like\n    The dependent data.\n"
                         "p0 : array_like, optional\n    Initial guess.\n"
                         "bounds : 2-tuple, optional\n    Lower and upper bounds.",
            "returns": "popt : array\n    Optimal values.\npcov : 2-D array\n    Estimated covariance.",
            "examples": ">>> import numpy as np\n>>> from scipy.optimize import curve_fit\n>>> def func(x, a, b, c):\n...     return a * np.exp(-b * x) + c\n>>> xdata = np.linspace(0, 4, 50)\n>>> ydata = func(xdata, 2.5, 1.3, 0.5)\n>>> popt, pcov = curve_fit(func, xdata, ydata)\n>>> popt\narray([2.5..., 1.3..., 0.5...])",
            "full_text": "scipy.optimize.curve_fit - Use non-linear least squares to fit a function to data...",
            "doc_type": "function",
            "retrieved_at": "",
            "scipy_doc_version": "",
        },
        {
            "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html",
            "title": "scipy.integrate.quad",
            "module": "scipy.integrate",
            "function_name": "quad",
            "signature": "scipy.integrate.quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50)",
            "description": "Compute a definite integral.",
            "parameters": "func : function\n    A Python callable to integrate.\n"
                         "a : float\n    Lower limit.\n"
                         "b : float\n    Upper limit.\n"
                         "args : tuple, optional\n    Extra arguments.",
            "returns": "y : float\n    The integral.\nabserr : float\n    Absolute error estimate.",
            "examples": ">>> from scipy import integrate\n>>> f = lambda x: x**2\n>>> integrate.quad(f, 0, 1)\n(0.33333333333333337, 3.700743415417189e-15)",
            "full_text": "scipy.integrate.quad - Compute a definite integral...",
            "doc_type": "function",
            "retrieved_at": "",
            "scipy_doc_version": "",
        },
        {
            "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html",
            "title": "scipy.interpolate.CubicSpline",
            "module": "scipy.interpolate",
            "function_name": "CubicSpline",
            "signature": "scipy.interpolate.CubicSpline(x, y, axis=0, bc_type=\"not-a-knot\", extrapolate=None)",
            "description": "Piecewise cubic interpolator to fit values on a 1-D grid.",
            "parameters": "x : array_like\n    1-D array of x-coordinates.\n"
                         "y : array_like\n    Array of y-coordinates.\n"
                         "bc_type : str or 2-tuple, optional\n    Boundary conditions.\n"
                         "extrapolate : bool, optional\n    Whether to extrapolate outside x.",
            "returns": "CubicSpline\n    A callable spline object.",
            "examples": ">>> import numpy as np\n>>> from scipy.interpolate import CubicSpline\n>>> x = np.arange(0, 10)\n>>> y = np.exp(-x/3.0)\n>>> cs = CubicSpline(x, y)\n>>> cs(2.5)\n0.43...",
            "full_text": "scipy.interpolate.CubicSpline - Piecewise cubic interpolator...",
            "doc_type": "class",
            "retrieved_at": "",
            "scipy_doc_version": "",
        },
        {
            "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html",
            "title": "scipy.linalg.solve",
            "module": "scipy.linalg",
            "function_name": "solve",
            "signature": "scipy.linalg.solve(a, b, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a=\"gen\", transposed=False)",
            "description": "Solve the linear equation set a @ x = b for x.",
            "parameters": "a : (M, M) array_like\n    Square matrix.\n"
                         "b : (M,) or (M, N) array_like\n    Right-hand side.\n"
                         "assume_a : str, optional\n    Matrix structure hint.",
            "returns": "x : ndarray\n    Solution with shape matching b.",
            "examples": ">>> import numpy as np\n>>> from scipy import linalg\n>>> a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])\n>>> b = np.array([2, 4, -1])\n>>> x = linalg.solve(a, b)\n>>> x\narray([ 2., -2.,  9.])",
            "full_text": "scipy.linalg.solve - Solves the linear equation set a @ x = b...",
            "doc_type": "function",
            "retrieved_at": "",
            "scipy_doc_version": "",
        },
    ]


if __name__ == "__main__":
    print("Creating sample dataset...")
    samples = create_sample_dataset()

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "scipy_sample.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} sample documents to data/processed/scipy_sample.json")
