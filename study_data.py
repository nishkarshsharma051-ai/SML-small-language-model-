"""
study_data.py — Knowledge Base for Ting Ling Ling Study AI
Contains structured data for History, English, and Science.
"""

# ─── HISTORY KNOWLEDGE ────────────────────────────────────────────────────────
HISTORY = {
    "world war 2": {
        "summary": "World War II (1939–1945) was a global conflict that involved the vast majority of the world's countries and all of the great powers—organized into two opposing military alliances: the Allies and the Axis.",
        "causes": "Long-term causes included the harsh Treaty of Versailles, the failure of the League of Nations, the Great Depression, and the aggressive expansionism of Nazi Germany (Lebensraum), Fascist Italy, and Imperial Japan.",
        "key_dates": [
            "Sept 1, 1939: Germany invades Poland, triggering the war.",
            "June 1941: Operation Barbarossa begins (Invasion of USSR).",
            "Dec 7, 1941: Pearl Harbor attack brings the USA into the conflict.",
            "June 6, 1944: D-Day (Invasion of Normandy).",
            "May 8, 1945: V-E Day (Victory in Europe).",
            "Aug 1945: Atomic bombings of Hiroshima and Nagasaki.",
            "Sept 2, 1945: V-J Day (Victory over Japan)."
        ],
        "impact": "The war led to the rise of the United States and the Soviet Union as global superpowers, the beginning of the Cold War, the decolonization of Africa and Asia, and the establishment of the United Nations to prevent future conflicts."
    },
    "french revolution": {
        "summary": "The French Revolution (1789–1799) was a period of far-reaching social and political upheaval in France that fundamentally changed the course of modern history, replacing absolute monarchy with democratic principles.",
        "causes": "Stagnant economy, extreme social inequality between the three Estates, the influence of Enlightenment philosophers (Voltaire, Rousseau), and enormous national debt.",
        "key_events": [
            "July 14, 1789: Storming of the Bastille, symbolic start of the revolution.",
            "Aug 1789: Declaration of the Rights of Man and of the Citizen.",
            "1792: Proclamation of the First French Republic.",
            "Jan 1793: Execution of King Louis XVI.",
            "1793-1794: Reign of Terror led by Maximilien Robespierre and the Committee of Public Safety.",
            "1799: Napoleon Bonaparte's coup d'état ends the revolution."
        ],
        "result": "The revolution spread liberal and democratic ideas throughout Europe and introduced the Napoleonic Code, which influenced civil law systems worldwide."
    },
    "ancient rome": {
        "summary": "Ancient Rome progressed from a small town on the Tiber River into an expansive empire that dominated the Mediterranean world for centuries.",
        "periods": "Kingdom (753–509 BC), Republic (509–27 BC), and Empire (27 BC–476 AD).",
        "legacy": "Development of Latin (the base of Romance languages), Roman Law (Twelve Tables), engineering marvels (aqueducts, roads, Colosseum), and the spread of Christianity."
    }
}

# ─── ENGLISH KNOWLEDGE ────────────────────────────────────────────────────────
ENGLISH = {
    "figures of speech": {
        "simile": "A comparison using 'like' or 'as'. Example: 'The mystery of his mind is as deep as the ocean.'",
        "metaphor": "A direct comparison stating one thing IS another. Example: 'Hope is the thing with feathers.'",
        "personification": "Attributing human traits to non-human entities. Example: 'The old floorboards groaned under the weight of the past.'",
        "alliteration": "The repetition of consonant sounds at the beginning of words. Example: 'A big black bug bit a big black bear.'",
        "oxymoron": "A figure of speech in which apparently contradictory terms appear in conjunction. Example: 'Deafening silence' or 'Bitter sweet'."
    },
    "literary analysis": {
        "thesis": "A central statement or claim that an essay argues and supports with evidence.",
        "theme": "The underlying message or universal idea explored in a literary work (e.g., Man vs. Nature, The Loss of Innocence).",
        "symbolism": "The use of objects, characters, or colors to represent abstract ideas (e.g., a green light for hope in The Great Gatsby)."
    }
}

# ─── MATHEMATICS CONCEPTS ────────────────────────────────────────────────────
MATH_CONCEPTS = {
    "pythagorean theorem": {
        "formula": "a² + b² = c²",
        "explanation": "In a right-angled triangle, the square of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the other two sides.",
        "applications": "Used in navigation, construction, and geometry to find distances."
    },
    "quadratic equations": {
        "general_form": "ax² + bx + c = 0",
        "quadratic_formula": "x = [-b ± sqrt(b² - 4ac)] / 2a",
        "discriminant": "D = b² - 4ac. If D > 0, 2 real roots; If D = 0, 1 real root; If D < 0, 0 real roots (complex roots)."
    },
    "calculus basics": {
        "derivative": "f'(x) = lim(h->0) [f(x+h) - f(x)] / h. Represens the instantaneous rate of change or the slope of the tangent line.",
        "integral": "The anti-derivative. Represents the area under a curve. Fundamental Theorem of Calculus: ∫[a,b] f(x)dx = F(b) - F(a)."
    },
    "euler's identity": "e^(iπ) + 1 = 0. Often called the most beautiful equation in mathematics because it links five fundamental mathematical constants: 0, 1, e, i, and π."
}

# ─── ADVANCED MATHEMATICS ──────────────────────────────────────────────────
ADVANCED_MATH = {
    "linear algebra": {
        "eigenvalues": "Values λ such that Av = λv for a square matrix A and non-zero vector v. They represent scaling factors along eigenvectors.",
        "eigenvectors": "Vectors whose direction remains unchanged after a linear transformation, only scaled by their corresponding eigenvalue.",
        "svd": "Singular Value Decomposition. Decomposes any matrix A into UΣV*, useful for data compression, noise reduction, and pseudo-inverses.",
        "orthogonality": "Two vectors are orthogonal if their dot product is zero (a·b = 0), meaning they are perpendicular in n-dimensional space."
    },
    "advanced calculus": {
        "multiple integrals": "Integrals over regions in ℝ² or ℝ³. Used to calculate volumes, mass, and centers of gravity.",
        "stokes theorem": "Relates a surface integral of the curl of a vector field to a line integral of that field around the boundary. ∫∫_S (∇ × F)·dS = ∮_C F·dr.",
        "divergence theorem": "Relates a volume integral of divergence to a surface integral of the flux. ∫∫∫_V (∇ · F)dV = ∫∫_S F·ndS.",
        "pde": "Partial Differential Equations. Equations involving partial derivatives of multivariable functions, like the heat equation or wave equation."
    },
    "discrete math": {
        "graph theory": "The study of graphs (nodes and edges). Key concepts include Eulerian paths, Hamiltonian cycles, and graph coloring.",
        "combinatorics": "The study of counting, arrangement, and combination. Formula for combinations: nCr = n! / [r!(n-r)!].",
        "set theory": "The study of sets. Operations include union (∪), intersection (∩), and complement (A')."
    }
}

# ─── MATH PROBLEMS & SOLUTIONS ──────────────────────────────────────────────
MATH_PROBLEMS = {
    "differential equations": {
        "problem": "Solve the first-order linear differential equation: dy/dx + 2y = e^x.",
        "solution": "1. Find the integrating factor μ(x) = e^(∫2dx) = e^(2x).\n2. Multiply both sides: e^(2x)dy/dx + 2e^(2x)y = e^(3x).\n3. Recognize the LHS as d/dx(y * e^(2x)): d/dx(y e^(2x)) = e^(3x).\n4. Integrate both sides: y e^(2x) = ∫e^(3x)dx = (1/3)e^(3x) + C.\n5. Solve for y: y = (1/3)e^x + Ce^(-2x)."
    },
    "linear algebra": {
        "problem": "Find the eigenvalues of matrix A = [[1, 2], [2, 1]].",
        "solution": "1. Set up the characteristic equation: det(A - λI) = 0.\n2. det([[1-λ, 2], [2, 1-λ]]) = (1-λ)² - 4 = 0.\n3. λ² - 2λ + 1 - 4 = 0  =>  λ² - 2λ - 3 = 0.\n4. Factorize: (λ - 3)(λ + 1) = 0.\n5. Eigenvalues are λ₁ = 3 and λ₂ = -1."
    }
}

# ─── SCIENCE KNOWLEDGE ───────────────────────────────────────────────────────
SCIENCE = {
    "quantum mechanics": {
        "summary": "The branch of physics that deals with the behavior of matter and light on the atomic and subatomic scale.",
        "key_principles": [
            "Wave-Particle Duality: Light and matter exhibit properties of both waves and particles.",
            "Uncertainty Principle: It is impossible to know both the position and momentum of a particle simultaneously (Heisenberg).",
            "Superposition: A quantum system can be in multiple states at once until it is observed (Schrödinger's Cat)."
        ]
    },
    "thermodynamics": {
        "laws": [
            "0th Law: If two systems are in equilibrium with a third, they are in equilibrium with each other.",
            "1st Law: Energy cannot be created or destroyed, only transformed (Conservation of Energy).",
            "2nd Law: The total entropy of an isolated system always increases over time.",
            "3rd Law: As temperature approaches absolute zero, the entropy of a system approaches a constant minimum."
        ]
    },
    "photosynthesis": "6CO2 + 6H2O + Light -> C6H12O6 + 6O2. The process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."
}

# ─── CODING KNOWLEDGE ────────────────────────────────────────────────────────
CODING = {
    "python basics": {
        "variables": "Assign values with = sign. x = 5, y = 'Hello'.",
        "loops": "For loops iterate over sequences: 'for i in range(5): print(i)'. While loops continue as long as a condition is true.",
        "functions": "Defined using 'def keyword'. Example: 'def greet(name): return f\"Hello, {name}\"'",
        "data_structures": "Lists [1, 2, 3], Tuples (1, 2), Dictionaries {'key': 'value'}, Sets {1, 2}."
    },
    "javascript": {
        "variables": "Use 'let' for reassignable variables, 'const' for constants. Avoid 'var'.",
        "promises": "Handle asynchronous operations. Use .then() or async/await.",
        "dom": "Document Object Model. Access elements using 'document.getElementById()' or 'querySelector()'.",
        "es6_features": "Arrow functions () => {}, Template literals `${var}`, Destructuring, Spreads."
    },
    "web development": {
        "html": "HyperText Markup Language. Uses tags like <div>, <header>, <section>, <footer>.",
        "css_flexbox": "display: flex; justify-content: center; align-items: center; flex-direction: column.",
        "css_grid": "display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;",
        "responsive_design": "Use @media (max-width: 600px) { ... } for mobile optimization."
    },
    "algorithms": {
        "big_o": "Notation to describe complexity: O(1) Constant, O(n) Linear, O(log n) Logarithmic, O(n²) Quadratic.",
        "sorting": "Bubble Sort (O(n²)), Merge Sort (O(n log n)), Quick Sort (avg O(n log n)).",
        "search": "Binary Search requires a sorted list and has O(log n) complexity."
    }
}
