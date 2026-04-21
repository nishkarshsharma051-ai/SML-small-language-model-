"""
Build a small instruction-tuning dataset for Hugging Face fine-tuning.

The goal here is not to train a foundation model from scratch. Instead, we turn
the existing study knowledge in this repo into instruction/response pairs that
can be used to fine-tune a pretrained causal LM into a more useful local chat
model.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List

from study_data import HISTORY, ENGLISH, MATH_CONCEPTS, SCIENCE, CODING


OUTPUT_DIR = "data"
TRAIN_PATH = os.path.join(OUTPUT_DIR, "hf_sft_train.jsonl")
VAL_PATH = os.path.join(OUTPUT_DIR, "hf_sft_val.jsonl")
SEED = 42


def _add_example(examples: List[Dict[str, str]], prompt: str, response: str) -> None:
    examples.append({
        "prompt": prompt.strip(),
        "response": response.strip(),
    })


def _add_variants(examples: List[Dict[str, str]], prompts: List[str], response: str) -> None:
    for prompt in prompts:
        _add_example(examples, prompt, response)


def _join_lines(*lines: str) -> str:
    return "\n".join(line for line in lines if line)


def build_examples() -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []

    ww2 = HISTORY["world war 2"]
    _add_variants(
        examples,
        [
            "Explain World War 2.",
            "Give me a summary of World War 2.",
            "What happened during World War 2?",
        ],
        _join_lines(
            ww2["summary"],
            "",
            "Key causes: " + ww2["causes"],
            "",
            "Important dates:",
            *[f"- {item}" for item in ww2["key_dates"]],
            "",
            "Impact: " + ww2["impact"],
        ),
    )
    _add_variants(examples, ["What caused World War 2?", "Why did World War 2 start?"], ww2["causes"])
    _add_variants(examples, ["Give me the key dates of World War 2.", "List the major World War 2 dates."], "\n".join(f"- {item}" for item in ww2["key_dates"]))

    french = HISTORY["french revolution"]
    _add_variants(examples, ["Explain the French Revolution.", "Summarize the French Revolution."], french["summary"])
    _add_variants(examples, ["What caused the French Revolution?", "Why did the French Revolution happen?"], french["causes"])
    _add_variants(examples, ["What was the result of the French Revolution?", "What changed after the French Revolution?"], french["result"])

    rome = HISTORY["ancient rome"]
    _add_variants(examples, ["Summarize Ancient Rome.", "Explain Ancient Rome."], rome["summary"])
    _add_variants(examples, ["What is the legacy of Ancient Rome?", "Why is Ancient Rome important?"], rome["legacy"])

    english_figures = ENGLISH["figures of speech"]
    _add_variants(examples, ["What is a simile?", "Define simile."], english_figures["simile"])
    _add_variants(examples, ["What is a metaphor?", "Define metaphor."], english_figures["metaphor"])
    _add_variants(examples, ["What is personification?", "Define personification."], english_figures["personification"])
    _add_variants(examples, ["What is alliteration?", "Define alliteration."], english_figures["alliteration"])
    _add_variants(examples, ["What is an oxymoron?", "Define oxymoron."], english_figures["oxymoron"])

    lit = ENGLISH["literary analysis"]
    _add_variants(examples, ["What is a thesis in an essay?", "Explain the thesis of an essay."], lit["thesis"])
    _add_variants(examples, ["What is a theme in literature?", "Explain theme in literature."], lit["theme"])
    _add_variants(examples, ["What is symbolism?", "Define symbolism in literature."], lit["symbolism"])

    math = MATH_CONCEPTS["pythagorean theorem"]
    _add_variants(examples, ["Explain the Pythagorean theorem.", "What is the Pythagorean theorem?"], _join_lines(math["formula"], math["explanation"], math["applications"]))
    quadratic = MATH_CONCEPTS["quadratic equations"]
    _add_variants(examples, ["What is the quadratic formula?", "Give me the quadratic formula."], quadratic["quadratic_formula"])
    _add_variants(examples, ["Explain the discriminant of a quadratic equation.", "What does the discriminant tell us?"], quadratic["discriminant"])
    calculus = MATH_CONCEPTS["calculus basics"]
    _add_variants(examples, ["What is a derivative?", "Explain derivative."], calculus["derivative"])
    _add_variants(examples, ["What is an integral?", "Explain integral."], calculus["integral"])
    _add_variants(examples, ["What is Euler's identity?", "Explain Euler's identity."], MATH_CONCEPTS["euler's identity"])
    _add_variants(
        examples,
        ["Explain the chain rule.", "What is the chain rule in calculus?"],
        "If y = f(g(x)), then $\\frac{dy}{dx} = f'(g(x)) \\cdot g'(x)$. In symbols: $\\frac{d}{dx} f(g(x)) = f'(g(x))g'(x)$.",
    )
    _add_variants(
        examples,
        ["Explain limits in calculus.", "What is a limit?"],
        "A limit describes what a function approaches as $x$ approaches a value. We write $\\lim_{x \\to a} f(x) = L$ if the output gets arbitrarily close to $L$.",
    )
    _add_variants(
        examples,
        ["What is a Taylor series?", "Explain Taylor series."],
        "A Taylor series writes a function as an infinite sum of derivatives around a point: $f(x) = \\sum_{n=0}^{\\infty} \\frac{f^{(n)}(a)}{n!}(x-a)^n$.",
    )
    _add_variants(
        examples,
        ["Explain matrices.", "What is a matrix?"],
        "A matrix is a rectangular array of numbers used for linear transformations, solving systems of equations, and representing data. For example, $A \\in \\mathbb{R}^{m\\times n}$.",
    )
    _add_variants(
        examples,
        ["Explain vectors and dot product.", "What is a dot product?"],
        "A vector has magnitude and direction. The dot product is $\\mathbf{a}\\cdot\\mathbf{b}=\\sum_i a_i b_i$, and it measures alignment between vectors.",
    )
    _add_variants(
        examples,
        ["Explain probability.", "What is probability?"],
        "Probability measures how likely an event is, with $0 \\le P(E) \\le 1$. For independent events, $P(A \\cap B)=P(A)P(B)$.",
    )

    science = SCIENCE["quantum mechanics"]
    _add_variants(examples, ["Explain quantum mechanics.", "What is quantum mechanics?"], science["summary"])
    _add_variants(examples, ["What is the uncertainty principle?", "Explain Heisenberg's uncertainty principle."], science["key_principles"][1])
    _add_variants(examples, ["What is superposition in quantum mechanics?", "Explain quantum superposition."], science["key_principles"][2])
    thermo = SCIENCE["thermodynamics"]
    _add_variants(examples, ["Explain the laws of thermodynamics.", "List the laws of thermodynamics."], "\n".join(f"- {law}" for law in thermo["laws"]))
    _add_variants(examples, ["What is photosynthesis?", "Explain photosynthesis."], SCIENCE["photosynthesis"])

    coding = CODING["python basics"]
    _add_variants(
        examples,
        ["Explain Python variables with an example.", "What are Python variables?"],
        _join_lines(
            coding["variables"],
            "",
            "Example:",
            "```python",
            "x = 5",
            "name = 'Ting Ling Ling'",
            "```",
        ),
    )
    _add_variants(examples, ["Explain Python loops.", "How do loops work in Python?"], coding["loops"])
    _add_variants(examples, ["How do I define a function in Python?", "What is a Python function?"], coding["functions"])
    _add_variants(examples, ["What data structures should I know in Python?", "Explain Python data structures."], coding["data_structures"])

    js = CODING["javascript"]
    _add_variants(examples, ["Explain JavaScript variables.", "What are let and const in JavaScript?"], js["variables"])
    _add_variants(examples, ["What are JavaScript promises?", "Explain async JavaScript promises."], js["promises"])
    _add_variants(examples, ["How do I access the DOM in JavaScript?", "What is the DOM?"], js["dom"])
    _add_variants(examples, ["What ES6 features should I know?", "Explain ES6 features."], js["es6_features"])

    web = CODING["web development"]
    _add_variants(examples, ["How do I center content with Flexbox?", "Teach me Flexbox centering."], web["css_flexbox"])
    _add_variants(examples, ["What is CSS Grid?", "Explain CSS Grid."], web["css_grid"])
    _add_variants(examples, ["How do I make a responsive website?", "What is responsive design?"], web["responsive_design"])

    alg = CODING["algorithms"]
    _add_variants(examples, ["Explain Big O notation.", "What is Big O notation?"], alg["big_o"])
    _add_variants(examples, ["What sorting algorithm should I use?", "Compare common sorting algorithms."], alg["sorting"])
    _add_variants(examples, ["How does binary search work?", "Explain binary search."], alg["search"])
    _add_variants(
        examples,
        ["Explain recursion in programming.", "What is recursion?"],
        "Recursion is when a function calls itself on a smaller version of the problem. A recursive solution needs a base case and a recursive step.\n\nExample:\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```",
    )
    _add_variants(
        examples,
        ["Explain object-oriented programming.", "What is OOP?"],
        "Object-oriented programming organizes code around objects and classes. Core ideas: encapsulation, inheritance, polymorphism, and abstraction.",
    )
    _add_variants(
        examples,
        ["Explain async and await in JavaScript.", "How does async/await work?"],
        "async/await lets you write asynchronous code that reads like synchronous code. Use try/catch for errors and await on promises.",
    )
    _add_variants(
        examples,
        ["Explain memoization.", "What is memoization in algorithms?"],
        "Memoization stores the results of expensive function calls so repeated inputs can return instantly. It is a common optimization for recursion and dynamic programming.",
    )
    _add_variants(
        examples,
        ["How do I structure a clean Python project?", "Give me Python project best practices."],
        "A clean Python project usually separates source code, tests, configuration, and documentation. Use modules, type hints when useful, and keep functions small and focused.",
    )
    _add_variants(
        examples,
        ["Explain SQL joins.", "What are SQL joins?"],
        "SQL joins combine rows from multiple tables. Common types are INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN.",
    )

    # General assistant style examples help the model answer more naturally.
    _add_example(examples, "Be concise and answer in plain English.", "Sure. I can answer directly and keep it concise when you ask for brevity.")
    _add_example(examples, "Write code in a clean style.", "I will use clear structure, comments when helpful, and fenced code blocks with the correct language tag.")
    _add_example(examples, "What should you do if you are unsure?", "I should say I am not sure, avoid inventing facts, and ask a clarifying question if needed.")
    _add_example(examples, "Use universal signs in math.", "I should prefer symbols like ∑, ∫, ∂, ⇒, ∀, ∃, ≤, ≥, and ≈ when they make the explanation clearer.")
    _add_variants(
        examples,
        ["Help me write a professional email.", "Draft a polite email for me."],
        "Sure. I can help with that. A strong email usually has a clear subject, a brief greeting, the request or purpose in the first sentence, and a polite closing.",
    )
    _add_variants(
        examples,
        ["Summarize this text in a few bullets.", "Give me a concise summary."],
        "Absolutely. I can turn long text into a short summary, key bullets, or a plain-English explanation depending on what you need.",
    )
    _add_variants(
        examples,
        ["Explain this concept like I'm a beginner.", "Teach me from the basics."],
        "Of course. I will start with the basics, define the terms, then build up to the advanced idea with a simple example.",
    )
    _add_variants(
        examples,
        ["Explain this concept at an advanced level.", "Go deep on this topic."],
        "I will give a deeper explanation with precise terminology, important edge cases, and the mathematical or technical structure where relevant.",
    )
    _add_variants(
        examples,
        ["Brainstorm ideas for a project.", "Give me creative ideas."],
        "Here are a few directions we could take: a practical build, a creative twist, a minimal version, and a more ambitious version.",
    )
    _add_variants(
        examples,
        ["Help me debug this code.", "Find the bug in my code."],
        "I can help debug it. Please share the code, the error message, and what you expected to happen, and I will trace the issue step by step.",
    )
    _add_variants(
        examples,
        ["Write me a study plan.", "Help me make a revision schedule."],
        "Sure. I can create a structured study plan with daily goals, revision blocks, and practice checkpoints.",
    )
    _add_variants(
        examples,
        ["Answer clearly and directly.", "Be conversational and helpful."],
        "Understood. I will answer clearly, stay conversational, and adapt the depth to your question.",
    )

    return examples


def write_jsonl(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    examples = build_examples()
    random.Random(SEED).shuffle(examples)

    split_idx = max(1, int(len(examples) * 0.9))
    train_rows = examples[:split_idx]
    val_rows = examples[split_idx:]

    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(VAL_PATH, val_rows)

    print(f"Wrote {len(train_rows)} training rows to {TRAIN_PATH}")
    print(f"Wrote {len(val_rows)} validation rows to {VAL_PATH}")


if __name__ == "__main__":
    main()
