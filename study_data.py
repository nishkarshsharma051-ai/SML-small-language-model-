"""
study_data.py — Knowledge Base for Ting Ling Ling Study AI
Contains structured data for History, English, and Science.
"""

# ─── HISTORY KNOWLEDGE ────────────────────────────────────────────────────────
HISTORY = {
    "world war 2": {
        "summary": "World War II (1939–1945) was a global conflict between the Allies (UK, USA, USSR, China) and the Axis (Germany, Japan, Italy).",
        "causes": "Main causes included the Treaty of Versailles' harsh terms on Germany, the rise of fascism (Hitler/Mussolini), and Japanese expansionism.",
        "key_dates": [
            "1939: Germany invades Poland (Start)",
            "1941: Pearl Harbor (USA enters)",
            "1944: D-Day (Allied invasion of France)",
            "1945: Hiroshima/Nagasaki bombings and Axis surrender."
        ],
        "impact": "Led to the Cold War, the creation of the United Nations, and the end of European colonialism."
    },
    "french revolution": {
        "summary": "The French Revolution (1789–1799) overthrew the absolute monarchy and established a republic based on 'Liberty, Equality, Fraternity'.",
        "causes": "Social inequality (The Three Estates), financial crisis, and Enlightenment ideas.",
        "key_events": [
            "1789: Storming of the Bastille",
            "1793: Execution of King Louis XVI",
            "1793-94: The Reign of Terror (Robespierre)"
        ],
        "result": "Rise of Napoleon Bonaparte and the spread of democratic ideals across Europe."
    },
    "industrial revolution": {
        "summary": "The transition to new manufacturing processes in Europe and the US (1760–1840).",
        "innovations": "Steam engine (James Watt), Spinning Jenny, and the factory system.",
        "impact": "Rapid urbanization, rise of capitalism, and shift from agrarian to industrial societies."
    },
    "ancient egypt": {
        "summary": "A civilization in Northeast Africa along the Nile River, famous for pyramids, pharaohs, and hieroglyphs.",
        "key_features": "Pharaohs were god-kings; Pyramids served as tombs; Mummification preserved the body for the afterlife."
    }
}

# ─── ENGLISH KNOWLEDGE ────────────────────────────────────────────────────────
ENGLISH = {
    "simile": "A comparison using 'like' or 'as'. Example: 'He stands like a tower.'",
    "metaphor": "A direct comparison stating one thing IS another. Example: 'The world is a stage.'",
    "personification": "Giving human qualities to non-human things. Example: 'The wind whispered through the trees.'",
    "alliteration": "The repetition of consonant sounds at the beginning of words. Example: 'Peter Piper picked a peck...'",
    "hyperbole": "Extreme exaggeration for effect. Example: 'I've told you a million times.'",
    "essay structure": {
        "intro": "Hook, background info, and a clear thesis statement.",
        "body": "Paragraphs starting with topic sentences, followed by evidence and analysis.",
        "conclusion": "Restate thesis in a new way, summarize points, and provide a final thought."
    },
    "grammar rules": {
        "punctuation": "Commas are for pauses; semicolons join two independent clauses; colons introduce lists or explanations.",
        "tenses": "Past Simple (I walked), Present Simple (I walk), Future (I will walk)."
    }
}

# ─── MATHEMATICS CONCEPTS ────────────────────────────────────────────────────
MATH_CONCEPTS = {
    "pythagorean theorem": "In a right triangle, a² + b² = c², where c is the hypotenuse.",
    "quadratic formula": "x = [-b ± sqrt(b² - 4ac)] / 2a. Used to find roots of ax² + bx + c = 0.",
    "calculus basics": "Differentiation finds the rate of change (slope); Integration finds the area under a curve.",
    "circle area": "Area = πr², where r is the radius.",
    "circle circumference": "Circumference = 2πr."
}

# ─── SCIENCE KNOWLEDGE ───────────────────────────────────────────────────────
SCIENCE = {
    "photosynthesis": "The process by which plants use sunlight, water, and CO2 to create oxygen and energy (glucose).",
    "water cycle": "Evaporation, Condensation, Precipitation, and Collection.",
    "newtons laws": [
        "1. Inertia: An object stays at rest unless acted upon.",
        "2. F=ma: Force equals mass times acceleration.",
        "3. Action/Reaction: Every action has an equal and opposite reaction."
    ],
    "periodic table": "A display of chemical elements arranged by atomic number, electron configuration, and chemical properties."
}
