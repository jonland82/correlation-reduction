"""Build a responsive HTML edition from the current LaTeX manuscript.

Requires pandoc and Beautiful Soup 4. Run: python build_html.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString


ROOT = Path(__file__).resolve().parent
TEX = ROOT / "dependence_work_revised.tex"
OUTPUT = ROOT / "dependence_work.html"


def pandoc(source: str | None = None, standalone: bool = False) -> str:
    command = ["pandoc", "-f", "latex", "-t", "html5", "--mathjax"]
    if standalone:
        command.append("--standalone")
    result = subprocess.run(
        command + ([str(TEX)] if source is None else []),
        input=source,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        cwd=ROOT,
    )
    return result.stdout


def fragment(soup: BeautifulSoup, latex: str):
    return BeautifulSoup(pandoc(latex), "html.parser")


def add_fragment(target, snippet: BeautifulSoup):
    for child in list(snippet.contents):
        target.append(child)


def roadmap_steps(source: str):
    def brace_group(start: int):
        if source[start] != "{":
            raise ValueError("Expected opening brace in roadmap step")
        depth = 0
        for position in range(start, len(source)):
            if source[position] == "{":
                depth += 1
            elif source[position] == "}":
                depth -= 1
                if depth == 0:
                    return source[start + 1:position], position + 1
        raise ValueError("Unclosed roadmap step")

    for match in re.finditer(r"\\roadmapstep(?=\{)", source):
        heading, next_position = brace_group(match.end())
        detail, _ = brace_group(next_position)
        yield heading, detail


def main() -> None:
    source = TEX.read_text(encoding="utf-8")
    soup = BeautifulSoup(pandoc(standalone=True), "html.parser")
    soup.html["lang"] = "en"
    soup.html.attrs.pop("xmlns", None)
    soup.html.attrs.pop("xml:lang", None)
    soup.head.style.decompose()
    for script in list(soup.head.find_all("script")):
        script.decompose()
    css = soup.new_tag("link", rel="stylesheet", href="paper.css")
    soup.head.append(css)
    config = soup.new_tag("script")
    config.string = (
        "window.MathJax = {tex: {packages: {'[+]': ['ams']}, "
        "inlineMath: [['\\\\(', '\\\\)']]}, "
        "options: {skipHtmlTags: ['script','noscript','style','textarea','pre','code']}};"
    )
    soup.head.append(config)
    mathjax = soup.new_tag(
        "script",
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js",
        defer=True,
    )
    soup.head.append(mathjax)
    description = soup.new_tag("meta", attrs={"name": "description", "content": "A workshop paper on the work needed to change dependence between coupled oscillators and how to test it with cavity-coupled membranes."})
    soup.head.append(description)

    body = soup.body
    empty_roadmap = body.find("div", class_="center")
    if empty_roadmap:
        empty_roadmap.decompose()
    title = body.find("header", id="title-block-header")
    title["id"] = "top"
    title["class"] = "paper-heading"

    abstract_source = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.S)
    if not abstract_source:
        raise ValueError("Missing abstract")
    abstract = soup.new_tag("section", id="abstract", attrs={"class": "abstract"})
    label = soup.new_tag("h2")
    label.string = "Abstract"
    abstract.append(label)
    add_fragment(abstract, fragment(soup, abstract_source.group(1)))
    title.insert_after(abstract)

    roadmap = soup.new_tag("section", attrs={"class": "roadmap", "aria-label": "Narrative arc"})
    roadmap_title = soup.new_tag("h2")
    roadmap_title.string = "From a mathematical dependence change to an experimental test"
    roadmap.append(roadmap_title)
    steps = soup.new_tag("div", attrs={"class": "roadmap-steps"})
    for index, (heading, detail) in enumerate(roadmap_steps(source)):
        step = soup.new_tag("div", attrs={"class": "roadmap-step"})
        number = soup.new_tag("span", attrs={"class": "step-number"})
        number.string = f"{index + 1:02d}"
        step.append(number)
        strong = soup.new_tag("strong")
        strong.string = heading
        step.append(strong)
        small = soup.new_tag("span", attrs={"class": "step-detail"})
        small.string = re.sub(r"\$([^$]+)\$", r"\\(\1\\)", detail)
        step.append(small)
        steps.append(step)
    if len(steps.find_all("div", class_="roadmap-step")) != 4:
        raise ValueError("Expected four roadmap steps")
    roadmap.append(steps)
    abstract.insert_after(roadmap)

    # Pandoc leaves LaTeX equation labels in the math and renders links as
    # [eq:name]. Give every displayed equation a stable HTML anchor/number.
    equation_numbers: dict[str, int] = {}
    for number, math in enumerate(body.select("span.math.display"), 1):
        latex = math.get_text()
        match = re.search(r"\\label\{([^}]+)\}", latex)
        if match:
            equation_numbers[match.group(1)] = number
            math["id"] = match.group(1)
            latex = latex.replace(match.group(0), "")
            math.string = latex
        math["class"] = ["math", "display", "equation"]
        math["data-number"] = f"({number})"
    if len(body.select("span.math.display")) != 15:
        raise ValueError("Expected all 15 numbered equations")

    for link in body.select('a[data-reference-type="eqref"]'):
        key = link.get("data-reference")
        if key not in equation_numbers:
            raise ValueError(f"Missing equation {key}")
        link.string = f"({equation_numbers[key]})"
        link["class"] = "cross-reference"

    section_ids = ["sec:definition", "sec:equilibrium", "sec:membranes", "sec:test", "conclusion"]
    appendix_ids = ["app:equilibrium", "app:steady", "app:dynamics"]
    for number, key in enumerate(section_ids, 1):
        heading = body.find("h1", id=key)
        heading["class"] = "section-heading"
        prefix = soup.new_tag("span", attrs={"class": "section-number", "aria-hidden": "true"})
        prefix.string = f"{number:02d}"
        heading.insert(0, prefix)
    appendix_title = soup.new_tag("h2", id="appendix-title", attrs={"class": "group-heading"})
    appendix_title.string = "Appendix"
    body.find("h1", id=appendix_ids[0]).insert_before(appendix_title)
    for letter, key in zip("ABC", appendix_ids):
        heading = body.find("h1", id=key)
        heading["class"] = "section-heading appendix-heading"
        prefix = soup.new_tag("span", attrs={"class": "section-number", "aria-hidden": "true"})
        prefix.string = letter
        heading.insert(0, prefix)
    for link in body.select('a[data-reference-type="ref"]'):
        key = link.get("data-reference")
        if key in appendix_ids:
            link.string = "ABC"[appendix_ids.index(key)]
        elif key.startswith("fig:"):
            link.string = str(["fig:ratio", "fig:work"].index(key) + 1)
        link["class"] = "cross-reference"

    for number, figure in enumerate(body.find_all("figure"), 1):
        image = figure.find("img")
        image["loading"] = "lazy"
        image["decoding"] = "async"
        original_id = image.attrs.pop("id")
        figure["id"] = original_id
        figure["class"] = "paper-figure"
        picture = soup.new_tag("picture")
        mobile = "pointwise_ratio_mobile.svg" if number == 1 else "work_vs_duration_mobile.svg"
        picture.append(soup.new_tag("source", media="(max-width: 700px)", srcset=f"optical_spring_results/{mobile}", type="image/svg+xml"))
        image.wrap(picture)
        caption = figure.find("figcaption")
        caption.attrs.pop("aria-hidden", None)
        caption.insert(0, NavigableString(f"Figure {number}. "))
        # A full-size file remains useful if a reader wants to inspect a panel.
        figure_link = soup.new_tag("a", href=image["src"], attrs={"class": "figure-open", "aria-label": f"Open Figure {number} at full size"})
        figure_link.string = "Open full size ↗"
        figure.append(figure_link)

    bib = body.find("div", class_="thebibliography")
    bib.clear()
    bib["id"] = "references"
    bib["class"] = "references"
    refs_heading = soup.new_tag("h2", attrs={"class": "group-heading"})
    refs_heading.string = "References"
    bib.append(refs_heading)
    ref_list = soup.new_tag("ol")
    items = re.findall(r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem\{|\\end\{thebibliography\})", source, re.S)
    if len(items) != 2:
        raise ValueError("Expected two references")
    for key, entry in items:
        item = soup.new_tag("li", id=f"ref:{key}")
        add_fragment(item, fragment(soup, entry))
        ref_list.append(item)
    bib.append(ref_list)
    citation_numbers = {key: i for i, (key, _) in enumerate(items, 1)}
    for citation in body.select("span.citation"):
        key = citation["data-cites"]
        if key not in citation_numbers:
            raise ValueError(f"Missing reference {key}")
        citation.name = "a"
        citation["href"] = f"#ref:{key}"
        citation["class"] = "citation"
        citation.string = f"[{citation_numbers[key]}]"

    topbar = soup.new_tag("div", attrs={"class": "topbar"})
    topbar_name = soup.new_tag("a", href="#top", attrs={"class": "brand"})
    topbar_name.string = "J. R. Landers  /  Paper"
    topbar.append(topbar_name)
    pdf_link = soup.new_tag("a", href="dependence_work_revised.pdf", attrs={"class": "pdf-link", "download": ""})
    pdf_link.string = "Download PDF ↗"
    topbar.append(pdf_link)

    shell = soup.new_tag("div", attrs={"class": "site-shell"})
    left = soup.new_tag("nav", attrs={"class": "left-rail", "aria-label": "Paper sections"})
    left_label = soup.new_tag("span", attrs={"class": "rail-label"})
    left_label.string = "CONTENTS"
    left.append(left_label)
    for key, text in [
        ("abstract", "Abstract"), ("sec:definition", "Define dependence"),
        ("sec:equilibrium", "Equilibrium work"), ("sec:membranes", "Membranes"),
        ("sec:test", "Experimental test"), ("conclusion", "Conclusion"),
        ("appendix-title", "Appendix"), ("references", "References"),
    ]:
        link = soup.new_tag("a", href=f"#{key}")
        link.string = text
        left.append(link)

    article = soup.new_tag("main", id="paper", attrs={"class": "paper"})
    for node in list(body.contents):
        article.append(node)
    right = soup.new_tag("aside", attrs={"class": "right-rail", "aria-label": "Paper overview"})
    right_label = soup.new_tag("span", attrs={"class": "rail-label"})
    right_label.string = "THE ARC"
    right.append(right_label)
    for number, text in enumerate(("Define the change", "Calculate the work", "Model the membranes", "Test both together"), 1):
        p = soup.new_tag("p")
        num = soup.new_tag("span")
        num.string = f"0{number}"
        p.append(num)
        p.append(NavigableString(text))
        right.append(p)
    shell.extend((left, article, right))
    body.extend((topbar, shell))

    ids = {node.get("id") for node in body.select("[id]")}
    broken = [node["href"] for node in body.select("a[href]")
              if node["href"].startswith("#") and node["href"][1:] not in ids]
    if broken:
        raise ValueError(f"Broken internal links: {broken}")

    OUTPUT.write_text("<!doctype html>\n" + str(soup.html), encoding="utf-8")
    print(f"Wrote {OUTPUT.name}: {len(equation_numbers)} labeled equations, 15 displayed equations, 2 figures, 2 references")


if __name__ == "__main__":
    main()
