import os
import json
from study_data import HISTORY, ENGLISH, MATH_CONCEPTS, ADVANCED_MATH, MATH_PROBLEMS, SCIENCE, CODING, ASSISTANT_BEHAVIOR, ADVANCED_CODING

def format_item(key, val):
    if isinstance(val, dict):
        lines = [f"### {key.upper()}"]
        for k, v in val.items():
            field_name = k.replace('_', ' ').capitalize()
            if isinstance(v, list):
                lines.append(f"**{field_name}**:")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"**{field_name}**: {v}")
        return "\n".join(lines)
    return f"**{key.upper()}**: {val}"

def generate_scholarly_text():
    corpus = []
    
    # Inject Assistant Persona
    corpus.append("--- ASSISTANT SYSTEM GUIDELINES ---")
    corpus.append(format_item("Behavior", ASSISTANT_BEHAVIOR))
    
    # Process History
    corpus.append("--- SCHOLARLY HISTORY ---")
    for topic, data in HISTORY.items():
        corpus.append(format_item(topic, data))
    
    # Process English
    corpus.append("\n--- SCHOLARLY ENGLISH & LITERATURE ---")
    for topic, data in ENGLISH.items():
        corpus.append(format_item(topic, data))
        
    # Process Mathematics
    corpus.append("\n--- SCHOLARLY MATHEMATICS ---")
    for topic, data in MATH_CONCEPTS.items():
        corpus.append(format_item(topic, data))

    # Process Advanced Mathematics
    corpus.append("\n--- SCHOLARLY ADVANCED MATHEMATICS ---")
    for topic, data in ADVANCED_MATH.items():
        corpus.append(format_item(topic, data))

    # Process Math Problems
    corpus.append("\n--- SCHOLARLY MATH PROBLEMS ---")
    for topic, data in MATH_PROBLEMS.items():
        corpus.append(format_item(topic, data))
        
    # Process Science
    corpus.append("\n--- SCHOLARLY SCIENCE ---")
    for topic, data in SCIENCE.items():
        corpus.append(format_item(topic, data))

    # Process Coding
    corpus.append("\n--- SCHOLARLY CODING & PROGRAMMING ---")
    for topic, data in CODING.items():
        corpus.append(format_item(topic, data))
        
    # Process Advanced Coding
    corpus.append("\n--- SCHOLARLY ADVANCED CODING & SYSTEM DESIGN ---")
    for topic, data in ADVANCED_CODING.items():
        corpus.append(format_item(topic, data))
        
    base_text = "\n\n".join(corpus)
    
    # To help a small model learn, we repeat the facts multiple times (Augmentation)
    # This acts like "overfitting" on the core knowledge.
    final_text = (base_text + "\n\n") * 100
    
    output_path = os.path.join("data", "scholarly_input.txt")
    with open(output_path, "w") as f:
        f.write(final_text)
        
    print(f"Generated {len(final_text)} characters of scholarly data in {output_path}")

if __name__ == "__main__":
    generate_scholarly_text()
