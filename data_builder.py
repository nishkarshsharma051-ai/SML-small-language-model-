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

from study_data import HISTORY, ENGLISH, MATH_CONCEPTS, ADVANCED_MATH, MATH_PROBLEMS, SCIENCE, CODING, ADVANCED_CODING


OUTPUT_DIR = "data"
TRAIN_PATH = os.path.join(OUTPUT_DIR, "hf_sft_train.jsonl")
VAL_PATH = os.path.join(OUTPUT_DIR, "hf_sft_val.jsonl")
TEACHER_LOG_PATH = os.path.join(OUTPUT_DIR, "teacher_log.jsonl")
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


def _add_curated_instruction_examples(examples: List[Dict[str, str]]) -> None:
    curated = [
        {
            "prompt": "Write a Python function that checks whether a string is a palindrome.",
            "response": (
                "Here is a clean solution:\n\n"
                "```python\n"
                "def is_palindrome(text: str) -> bool:\n"
                "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
                "    return cleaned == cleaned[::-1]\n"
                "```\n\n"
                "This normalizes case, removes non-alphanumeric characters, and compares the text with its reverse."
            ),
        },
        {
            "prompt": "Explain object-oriented programming with a Python example.",
            "response": (
                "Object-oriented programming organizes code around classes and objects. "
                "A class defines behavior and structure, and an object is an instance of that class.\n\n"
                "```python\n"
                "class Dog:\n"
                "    def __init__(self, name: str):\n"
                "        self.name = name\n"
                "\n"
                "    def speak(self) -> str:\n"
                "        return f\"{self.name} says woof\"\n"
                "\n"
                "pet = Dog(\"Milo\")\n"
                "print(pet.speak())\n"
                "```\n\n"
                "This example shows encapsulation because the `Dog` object stores its own data and behavior together."
            ),
        },
        {
            "prompt": "Explain the difference between a list and a tuple in Python.",
            "response": (
                "A list is mutable, which means you can change its contents after creation. "
                "A tuple is immutable, which means it cannot be changed after it is created.\n\n"
                "Use a list when you expect updates, and use a tuple when the data should stay fixed."
            ),
        },
        {
            "prompt": "How do I reverse a string in Python?",
            "response": (
                "The shortest way is slicing:\n\n"
                "```python\n"
                "text = \"hello\"\n"
                "reversed_text = text[::-1]\n"
                "```\n\n"
                "This returns `olleh`."
            ),
        },
        {
            "prompt": "Write a SQL query to get the second highest salary from an Employee table.",
            "response": (
                "One common approach is:\n\n"
                "```sql\n"
                "SELECT MAX(salary) AS second_highest_salary\n"
                "FROM Employee\n"
                "WHERE salary < (SELECT MAX(salary) FROM Employee);\n"
                "```\n\n"
                "The inner query finds the maximum salary, and the outer query finds the largest salary below it."
            ),
        },
        {
            "prompt": "Explain REST APIs in simple terms.",
            "response": (
                "A REST API is a way for applications to communicate over HTTP using predictable rules. "
                "A client sends requests such as GET, POST, PUT, or DELETE to resource URLs, and the server returns structured data, often JSON."
            ),
        },
        {
            "prompt": "What is the difference between GET and POST?",
            "response": (
                "GET is usually used to retrieve data, and POST is usually used to send data that creates or triggers something on the server. "
                "GET requests are often safe and cacheable, while POST requests usually change server state."
            ),
        },
        {
            "prompt": "Explain the chain rule in calculus and why it works.",
            "response": (
                "The chain rule is used when one function is inside another. If $y = f(g(x))$, then\n\n"
                "$$\\frac{dy}{dx} = f'(g(x)) \\cdot g'(x).$$\n\n"
                "Step by step:\n"
                "1. Let $u = g(x)$ so that $y = f(u)$.\n"
                "2. Differentiate the outer function with respect to $u$, giving $\\frac{dy}{du} = f'(u)$.\n"
                "3. Differentiate the inner function with respect to $x$, giving $\\frac{du}{dx} = g'(x)$.\n"
                "4. Multiply them: $\\frac{dy}{dx} = \\frac{dy}{du} \\cdot \\frac{du}{dx}$.\n\n"
                "It works because a change in $x$ first changes the inside function, and that inside change then affects the outside function."
            ),
        },
        {
            "prompt": "Differentiate y = sin(x^2).",
            "response": (
                "Use the chain rule.\n\n"
                "Let $u = x^2$. Then $y = \\sin(u)$.\n"
                "The derivative of the outer function is $\\cos(u)$, and the derivative of the inner function is $2x$.\n\n"
                "So,\n\n"
                "$$\\frac{dy}{dx} = \\cos(x^2) \\cdot 2x = 2x\\cos(x^2).$$"
            ),
        },
        {
            "prompt": "Solve 2x + 5 = 17.",
            "response": (
                "Start with:\n\n"
                "$$2x + 5 = 17$$\n\n"
                "Subtract 5 from both sides:\n\n"
                "$$2x = 12$$\n\n"
                "Divide by 2:\n\n"
                "$$x = 6$$"
            ),
        },
        {
            "prompt": "What is a matrix in linear algebra?",
            "response": (
                "A matrix is a rectangular array of numbers arranged in rows and columns. "
                "Matrices are used to represent linear transformations, systems of equations, and structured data. "
                "For example, a matrix can be written as $A \\in \\mathbb{R}^{m \\times n}$."
            ),
        },
        {
            "prompt": "Explain vectors and the dot product with symbols.",
            "response": (
                "A vector is an ordered list of numbers, such as $\\mathbf{v} = (v_1, v_2, \\dots, v_n)$. "
                "The dot product of two vectors is\n\n"
                "$$\\mathbf{a} \\cdot \\mathbf{b} = \\sum_{i=1}^{n} a_i b_i.$$"
                "\n\nIt measures how aligned the two vectors are."
            ),
        },
        {
            "prompt": "Who was Nikola Tesla and why is he important?",
            "response": (
                "Nikola Tesla was an inventor and electrical engineer best known for his work on alternating current systems. "
                "He helped shape modern power transmission, motors, and electrical engineering. "
                "His ideas were important because they supported efficient long-distance electricity distribution."
            ),
        },
        {
            "prompt": "What was the Battle of Waterloo?",
            "response": (
                "The Battle of Waterloo was fought on June 18, 1815, and it marked the final defeat of Napoleon Bonaparte. "
                "Napoleon's forces were defeated by a coalition led by the Duke of Wellington and Gebhard Leberecht von Blucher. "
                "The battle ended Napoleon's rule and reshaped European politics."
            ),
        },
        {
            "prompt": "How do black holes form?",
            "response": (
                "Black holes usually form when a very massive star runs out of fuel and its core collapses under gravity. "
                "If the remaining core is massive enough, the collapse continues until an event horizon forms. "
                "Black holes can also grow later by accreting matter or merging with other compact objects."
            ),
        },
        {
            "prompt": "Explain photosynthesis simply.",
            "response": (
                "Photosynthesis is the process by which plants use sunlight to convert water and carbon dioxide into glucose and oxygen. "
                "In simple terms, plants use light energy to make their own food."
            ),
        },
        {
            "prompt": "Help me write a polite professional email asking for a project update.",
            "response": (
                "Here is a professional draft:\n\n"
                "Subject: Project Update Request\n\n"
                "Dear [Name],\n\n"
                "I hope you are doing well. I wanted to check in regarding the current status of the project and see whether there are any updates you can share. Please let me know if there are any blockers or if I can help with anything on my side.\n\n"
                "Best regards,\n"
                "[Your Name]"
            ),
        },
        {
            "prompt": "Rewrite this sentence more professionally: i need the file asap",
            "response": "Could you please send me the file at your earliest convenience?",
        },
        {
            "prompt": "Summarize this text in three bullets.",
            "response": (
                "Sure. I will identify the main argument, the supporting points, and the conclusion, then turn them into three concise bullets."
            ),
        },
        {
            "prompt": "Explain recursion like I'm a beginner.",
            "response": (
                "Recursion means a function solves a problem by calling itself on a smaller version of the same problem. "
                "A recursive solution needs two parts: a base case that stops the recursion and a recursive step that reduces the problem."
            ),
        },
        {
            "prompt": "Give me a 7-day study plan for calculus.",
            "response": (
                "Here is a simple 7-day plan:\n\n"
                "Day 1: Limits and continuity\n"
                "Day 2: Derivatives and rules\n"
                "Day 3: Chain rule and implicit differentiation\n"
                "Day 4: Applications of derivatives\n"
                "Day 5: Integrals and antiderivatives\n"
                "Day 6: Definite integrals and area\n"
                "Day 7: Mixed review and timed practice\n\n"
                "Spend part of each day on examples and part on problem solving."
            ),
        },
        {
            "prompt": "I have 3 days left before my math exam. Help me make a revision plan.",
            "response": (
                "Here is a focused 3-day revision plan:\n\n"
                "Day 1: Review core concepts and formulas. Work through easy and medium problems.\n"
                "Day 2: Practice mixed problem sets under time pressure. Review mistakes carefully.\n"
                "Day 3: Do a final recap of weak areas, key formulas, and one short mock test.\n\n"
                "Keep your revision active by solving problems, not only rereading notes."
            ),
        },
        {
            "prompt": "My code throws an IndexError. How should I debug it?",
            "response": (
                "An `IndexError` usually means you are trying to access a position that does not exist in a list or sequence. "
                "Check the sequence length, print the index value before access, and verify loop bounds such as `range(len(items))`. "
                "If you share the code and error line, I can trace it step by step."
            ),
        },
        {
            "prompt": "My Python code raises an IndexError. How should I debug it?",
            "response": (
                "An `IndexError` means your code is trying to access an element outside the valid range of a list, tuple, or string.\n\n"
                "A good debugging checklist is:\n"
                "1. Print the sequence and its length with `print(len(items))`.\n"
                "2. Print the index right before the failing line.\n"
                "3. Check loop bounds such as `range(len(items))` versus `range(len(items) + 1)`.\n"
                "4. Check whether the sequence might be empty.\n"
                "5. Read the traceback to find the exact line that failed.\n\n"
                "Example:\n\n"
                "```python\n"
                "items = [10, 20, 30]\n"
                "for i in range(len(items) + 1):\n"
                "    print(i, len(items))\n"
                "    print(items[i])\n"
                "```\n\n"
                "The bug is `range(len(items) + 1)`, because the last index attempted is `3`, but valid indices are `0`, `1`, and `2`."
            ),
        },
        {
            "prompt": "How do I debug an IndexError step by step in Python?",
            "response": (
                "Step 1: Read the traceback and locate the exact failing line.\n"
                "Step 2: Print the index value and the sequence length.\n"
                "Step 3: Check whether the index can become negative or exceed `len(sequence) - 1`.\n"
                "Step 4: Inspect loops, slices, and any logic that changes the sequence size.\n"
                "Step 5: Fix the boundary condition and test again.\n\n"
                "In short, an `IndexError` is usually a bounds problem, so compare the index you use with the valid index range."
            ),
        },
        {
            "prompt": "The app is slow. What should I check first?",
            "response": (
                "Start with simple measurements. Check load time, first-token latency, tokens per second, and memory usage. "
                "Then identify whether the slowdown comes from model loading, generation settings, the web server, or the frontend."
            ),
        },
        {
            "prompt": "What should you do if you are unsure about a fact?",
            "response": (
                "I should avoid inventing facts. I should say I am not fully sure, give the most careful answer I can, and ask a clarifying question when needed."
            ),
        },
        {
            "prompt": "If a user asks a harmless science question, should you refuse?",
            "response": (
                "No. I should answer harmless science questions directly and clearly. I should only refuse if the request is unsafe or unrelated to legitimate educational help."
            ),
        },
        {
            "prompt": "User: I do not understand derivatives.\nAssistant: No problem. A derivative measures how fast something changes.\nUser: Can you explain it with an example?",
            "response": (
                "Sure. Imagine distance changing over time. If a car's position is given by $s(t) = t^2$, then the derivative is $s'(t) = 2t$. "
                "That derivative tells you how fast the position is changing at each moment. For example, at $t = 3$, the rate of change is $6$."
            ),
        },
        {
            "prompt": "User: I need to write a formal email.\nAssistant: Sure. What is it about?\nUser: I want to ask my professor for an extension because I was sick.",
            "response": (
                "Here is a respectful draft:\n\n"
                "Subject: Request for Assignment Extension\n\n"
                "Dear Professor [Name],\n\n"
                "I hope you are doing well. I am writing to request a short extension on the assignment because I have been unwell and it has affected my ability to complete the work on time. I would be grateful if an extension could be considered. Thank you for your understanding.\n\n"
                "Sincerely,\n"
                "[Your Name]"
            ),
        },
        {
            "prompt": "Compare breadth-first search and depth-first search.",
            "response": (
                "Breadth-first search explores level by level, while depth-first search explores one path as far as possible before backtracking. "
                "BFS is useful for shortest paths in unweighted graphs, while DFS is often used for traversal, cycle detection, and topological-style exploration."
            ),
        },
        {
            "prompt": "Explain Dijkstra's algorithm simply.",
            "response": (
                "Dijkstra's algorithm finds the shortest path from a starting node to all other nodes in a graph with non-negative edge weights. "
                "It repeatedly picks the unvisited node with the smallest current distance, relaxes its outgoing edges, and updates better distances when found."
            ),
        },
        {
            "prompt": "What is overfitting in machine learning?",
            "response": (
                "Overfitting happens when a model learns the training data too specifically, including noise or accidental patterns, and then performs poorly on new data. "
                "A model that overfits usually has low training error but weaker generalization."
            ),
        },
        {
            "prompt": "Explain the difference between training loss and validation loss.",
            "response": (
                "Training loss measures how well the model fits the training data, while validation loss measures how well it performs on separate held-out data. "
                "If training loss improves but validation loss gets worse, the model may be overfitting."
            ),
        },
        {
            "prompt": "What makes a good answer in a coding assistant?",
            "response": (
                "A good coding answer is correct, readable, and practical. It should explain the idea briefly, provide clean code when needed, and mention edge cases or tradeoffs when they matter."
            ),
        },
        {
            "prompt": "Explain this topic in beginner, intermediate, and advanced terms: recursion.",
            "response": (
                "Beginner: Recursion is when a function calls itself to solve a smaller version of the same problem.\n\n"
                "Intermediate: Recursive solutions need a base case and a recursive step, and they are useful for trees, divide-and-conquer, and combinatorial search.\n\n"
                "Advanced: Recursion can be analyzed through recurrence relations, stack depth, and subproblem structure, and it is often optimized with memoization or converted into iterative dynamic programming."
            ),
        },
        {
            "prompt": "Explain matrices and vectors in linear algebra using symbols.",
            "response": (
                "In linear algebra, a matrix is a rectangular array of numbers and a vector is an ordered list of numbers.\n\n"
                "A matrix can be written as $A \\in \\mathbb{R}^{m \\times n}$ and a vector as $x \\in \\mathbb{R}^n$. "
                "Matrix-vector multiplication is written as\n\n"
                "$$A x = b,$$\n\n"
                "where $b \\in \\mathbb{R}^m$.\n\n"
                "For example,\n\n"
                "$$A = \\begin{pmatrix}1 & 2\\\\3 & 4\\end{pmatrix}, \\quad x = \\begin{pmatrix}x_1\\\\x_2\\end{pmatrix}$$\n\n"
                "gives\n\n"
                "$$Ax = \\begin{pmatrix}x_1 + 2x_2\\\\3x_1 + 4x_2\\end{pmatrix}.$$"
            ),
        },
        {
            "prompt": "Use symbols to explain matrices and vectors.",
            "response": (
                "A vector is an element of $\\mathbb{R}^n$, so we write $x = (x_1, x_2, \\dots, x_n)^T$. "
                "A matrix is an element of $\\mathbb{R}^{m \\times n}$, so we write $A = [a_{ij}]$. "
                "When a matrix acts on a vector, the result is another vector:\n\n"
                "$$Ax = b.$$"
                "\n\nThe dot product of two vectors $u, v \\in \\mathbb{R}^n$ is\n\n"
                "$$u \\cdot v = \\sum_{i=1}^{n} u_i v_i.$$"
            ),
        },
        {
            "prompt": "Give a clear symbolic explanation of matrix-vector multiplication.",
            "response": (
                "Let $A = [a_{ij}] \\in \\mathbb{R}^{m \\times n}$ and let $x \\in \\mathbb{R}^n$. "
                "Then the product $Ax$ is a vector in $\\mathbb{R}^m$ whose $i$-th entry is\n\n"
                "$$(Ax)_i = \\sum_{j=1}^{n} a_{ij} x_j.$$"
                "\n\nThis means each output entry is a linear combination of the entries of $x$, weighted by the corresponding row of $A$."
            ),
        },
    ]

    for row in curated:
        _add_example(examples, row["prompt"], row["response"])


def _dedupe_examples(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped = []
    for row in rows:
        key = (row["prompt"], row["response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


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
        ["Explain the chain rule in symbols and step by step.", "Why does the chain rule work?"],
        "Step 1: Let $y=f(u)$ and $u=g(x)$. Step 2: Differentiate with respect to $u$ and $x$, so $\\frac{dy}{du}=f'(u)$ and $\\frac{du}{dx}=g'(x)$. Step 3: Multiply the rates of change: $\\frac{dy}{dx}=\\frac{dy}{du}\\cdot\\frac{du}{dx}=f'(g(x))g'(x)$. This works because a tiny change in $x$ creates a tiny change in $u$, which then creates a tiny change in $y$.",
    )
    _add_variants(
        examples,
        ["What is the chain rule formula?", "Give the chain rule in one line."],
        "For a composite function $y=f(g(x))$, the chain rule is $\\frac{dy}{dx}=f'(g(x))\\cdot g'(x)$. In Leibniz notation, $\\frac{dy}{dx}=\\frac{dy}{du}\\cdot\\frac{du}{dx}$ when $u=g(x)$.",
    )
    _add_variants(
        examples,
        ["Show the chain rule on y = sin(x^2).", "Differentiate sin(x^2) using the chain rule."],
        "Let $y=\\sin(x^2)$. The outer derivative is $\\cos(u)$ and the inner derivative is $2x$, so $\\frac{dy}{dx}=\\cos(x^2)\\cdot 2x = 2x\\cos(x^2)$.",
    )
    _add_variants(
        examples,
        ["What is a common mistake with the chain rule?", "What should I not do with the chain rule?"],
        "A common mistake is to divide derivatives or mix up inner and outer functions. The correct rule multiplies them: $\\frac{dy}{dx}=f'(g(x))\\cdot g'(x)$, not a ratio.",
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
        ["Explain matrices and vectors in linear algebra using symbols.", "Use symbols to explain matrices and vectors."],
        "A matrix $A \\in \\mathbb{R}^{m\\times n}$ is a rectangular array of numbers, and a vector $x \\in \\mathbb{R}^n$ is a list of coordinates. Matrix-vector multiplication is written $Ax=b$, where $A$ transforms the vector $x$ into the vector $b$. In symbols: $A\\mathbf{x}=\\mathbf{b}$ and $\\mathbf{a}\\cdot\\mathbf{b}=\\sum_i a_i b_i$.",
    )
    _add_variants(
        examples,
        ["Show me a simple linear algebra example.", "Give a matrix-vector example."],
        "Let $A=\\begin{pmatrix}1 & 2\\\\3 & 4\\end{pmatrix}$ and $x=\\begin{pmatrix}x_1\\\\x_2\\end{pmatrix}$. Then $Ax=\\begin{pmatrix}x_1+2x_2\\\\3x_1+4x_2\\end{pmatrix}$. This shows how matrices transform vectors.",
    )
    _add_variants(examples, ["Explain probability.", "What is probability?"], _join_lines(
        "Probability measures how likely an event is, with $0 \\le P(E) \\le 1$. For independent events, $P(A \\cap B)=P(A)P(B)$.",
        "Key theorems include the Law of Large Numbers and the Central Limit Theorem."
    ))

    # Advanced Math
    la = ADVANCED_MATH["linear algebra"]
    _add_variants(examples, ["What are eigenvalues?", "Explain eigenvalues and eigenvectors."], _join_lines(la["eigenvalues"], la["eigenvectors"]))
    _add_variants(examples, ["What is SVD?", "Explain Singular Value Decomposition."], la["svd"])
    _add_variants(examples, ["What does orthogonality mean in math?", "Define orthogonal vectors."], la["orthogonality"])

    calc = ADVANCED_MATH["advanced calculus"]
    _add_variants(examples, ["Explain Stokes' Theorem.", "Give me the formula for Stokes' Theorem."], calc["stokes theorem"])
    _add_variants(examples, ["Explain the Divergence Theorem.", "What is the Gauss Divergence Theorem?"], calc["divergence theorem"])
    _add_variants(examples, ["What is a PDE?", "Explain Partial Differential Equations."], calc["pde"])

    discrete = ADVANCED_MATH["discrete math"]
    _add_variants(examples, ["Explain Graph Theory.", "What is graph theory used for?"], discrete["graph theory"])
    _add_variants(examples, ["What is the formula for combinations?", "Explain combinatorics."], discrete["combinatorics"])
    _add_variants(examples, ["Explain Set Theory basics.", "What are union and intersection in sets?"], discrete["set theory"])

    # Math Problems
    diff_eq = MATH_PROBLEMS["differential equations"]
    _add_variants(examples, ["Solve dy/dx + 2y = e^x.", "How do I solve a first-order linear ODE?"], _join_lines(
        f"Problem: {diff_eq['problem']}",
        "",
        "Solution:",
        diff_eq["solution"]
    ))
    
    matrix_prob = MATH_PROBLEMS["linear algebra"]
    _add_variants(examples, ["Find the eigenvalues of [[1, 2], [2, 1]].", "Show me a step-by-step eigenvalue calculation."], _join_lines(
        f"Problem: {matrix_prob['problem']}",
        "",
        "Solution:",
        matrix_prob["solution"]
    ))

    science = SCIENCE["quantum mechanics"]
    _add_variants(examples, ["Explain quantum mechanics.", "What is quantum mechanics?"], science["summary"])
    _add_variants(examples, ["What is the uncertainty principle?", "Explain Heisenberg's uncertainty principle."], science["key_principles"][1])
    _add_variants(examples, ["What is superposition in quantum mechanics?", "Explain quantum superposition."], science["key_principles"][2])
    thermo = SCIENCE["thermodynamics"]
    _add_variants(examples, ["Explain the laws of thermodynamics.", "List the laws of thermodynamics."], "\n".join(f"- {law}" for law in thermo["laws"]))
    _add_variants(examples, ["What is photosynthesis?", "Explain photosynthesis."], SCIENCE["photosynthesis"])
    _add_variants(
        examples,
        ["How do black holes form?", "Explain how black holes form."],
        "Black holes usually form when a very massive star exhausts its nuclear fuel and its core collapses under gravity. If the core mass is high enough, nothing can stop the collapse, and an event horizon forms. Black holes can also grow by mergers and accretion. In astronomy, this is a standard physics question, so I should answer it directly.",
    )
    _add_variants(
        examples,
        ["What is a black hole?", "Explain black holes simply."],
        "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape. It forms when matter is compressed into an extremely small and dense region. In simple terms, it is one of the densest objects in the universe.",
    )
    _add_variants(
        examples,
        ["Why can't light escape a black hole?", "Explain the event horizon in simple terms."],
        "Light cannot escape because the escape velocity at the event horizon exceeds the speed of light. The event horizon is the boundary where escape becomes impossible. That boundary is what makes the black hole look black.",
    )
    _add_variants(
        examples,
        ["What is an event horizon?", "Explain the event horizon of a black hole."],
        "The event horizon is the boundary around a black hole beyond which nothing can escape, not even light. It marks the point where the escape velocity becomes greater than the speed of light.",
    )
    _add_variants(
        examples,
        ["Give a simple explanation of black holes with symbols.", "Explain a black hole in one short paragraph."],
        "A black hole is a collapsed region of space with extremely strong gravity. If an object crosses the event horizon, it cannot escape, so we can write this idea as $v_{escape} > c \\Rightarrow$ no light escapes. I should answer astronomy questions directly and clearly.",
    )

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

    # Advanced Coding
    adv_coding = ADVANCED_CODING["dynamic programming"]
    _add_variants(examples, ["Explain Dynamic Programming.", "What is DP in coding?"], adv_coding["concept"])
    for prob in adv_coding["problems"]:
        _add_variants(examples, [f"Solve {prob['name']}.", f"Write a solution for {prob['name']}."], _join_lines(prob["problem"], "", prob["solution"]))

    graphs = ADVANCED_CODING["graph algorithms"]
    _add_variants(examples, ["Explain Graph Algorithms.", "What are graphs in coding?"], graphs["concept"])
    for algo in graphs["algorithms"]:
        _add_variants(examples, [f"Explain {algo['name']}.", f"How does {algo['name']} work?"], _join_lines(algo["description"], "", algo["implementation"]))

    sys_design = ADVANCED_CODING["system design"]
    _add_variants(examples, ["Explain Scalability in System Design.", "Vertical vs Horizontal Scaling."], sys_design["scalability"])
    _add_variants(examples, ["What is Load Balancing?", "Explain load balancer algorithms."], sys_design["load_balancing"])
    _add_variants(examples, ["Explain Caching strategies.", "What are write-through and write-back caches?"], sys_design["caching"])
    _add_variants(examples, ["What are Microservices?", "Explain microservices architecture."], sys_design["microservices"])

    adv_ds = ADVANCED_CODING["advanced data structures"]
    _add_variants(examples, ["What is a Trie data structure?", "Explain prefix trees."], adv_ds["trie"])
    _add_variants(examples, ["Explain Segment Trees.", "What is a segment tree used for?"], adv_ds["segment_tree"])
    _add_variants(examples, ["How to implement an LRU Cache?", "Explain LRU Cache."], adv_ds["lru_cache"])
    _add_variants(examples, ["What is a Fenwick Tree?", "Explain Binary Indexed Tree (BIT)."], adv_ds["fenwick_tree"])

    sys_patterns = ADVANCED_CODING["system design patterns"]
    _add_variants(examples, ["What are consistency patterns in system design?", "Explain eventual consistency."], sys_patterns["consistency_patterns"])
    _add_variants(examples, ["What is database sharding?", "How to scale a database?"], sys_patterns["database_sharding"])
    _add_variants(examples, ["Explain Event-Driven Architecture.", "What is Kafka used for in system design?"], sys_patterns["event_driven"])

    complexity = ADVANCED_CODING["complexity analysis"]
    _add_variants(examples, ["Explain Time Complexity.", "What is O(n log n)?"], complexity["time_complexity"])
    _add_variants(examples, ["What is Space Complexity?", "Explain memory usage of algorithms."], complexity["space_complexity"])
    _add_variants(examples, ["What is Amortized Analysis?", "Explain average cost of operations."], complexity["amortized_analysis"])

    concurrency = ADVANCED_CODING["concurrency"]
    _add_variants(examples, ["Explain Python GIL.", "Why is Python slow for multi-threading?"], concurrency["python_gil"])
    _add_variants(examples, ["What is asyncio in Python?", "Explain cooperative multitasking."], concurrency["asyncio"])
    _add_variants(examples, ["What are deadlocks?", "How to prevent deadlocks?"], concurrency["deadlocks"])

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
    _add_variants(
        examples,
        ["If I ask a science question, should you refuse?", "How should you answer astronomy questions?"],
        "No. For science and astronomy questions, I should answer directly, explain the concept clearly, and only refuse if the request is unsafe or unrelated to the science topic.",
    )

    _add_curated_instruction_examples(examples)

    # Optional: incorporate teacher answers collected from Grok fallback.
    # This is opt-in by existence of the log file, and is best used with your consent
    # because it may contain your private prompts.
    if os.path.exists(TEACHER_LOG_PATH):
        try:
            with open(TEACHER_LOG_PATH, "r", encoding="utf-8") as f:
                rows = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    p = str(obj.get("prompt", "")).strip()
                    r = str(obj.get("response", "")).strip()
                    if p and r:
                        rows.append({"prompt": p, "response": r})
            # Cap so accidental large logs don't dominate the curated dataset.
            random.Random(SEED).shuffle(rows)
            for row in rows[:200]:
                _add_example(examples, row["prompt"], row["response"])
        except Exception:
            pass

    return _dedupe_examples(examples)


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

    print(f"Built {len(examples)} unique instruction examples")
    print(f"Wrote {len(train_rows)} training rows to {TRAIN_PATH}")
    print(f"Wrote {len(val_rows)} validation rows to {VAL_PATH}")


if __name__ == "__main__":
    main()
