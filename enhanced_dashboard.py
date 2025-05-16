# enhanced_dashboard_complete.py
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import shap
from itertools import combinations
from collections import Counter
from datetime import datetime, timedelta
from statsmodels.stats.contingency_tables import StratifiedTable
# ==================================================================
# 0. Configuration & Mock Data
# ==================================================================
PEER_TUTORS = {
    "Algebra": ["S101", "S102", "S105"],
    "Geometry": ["S103", "S104", "S106"],
    "Calculus": ["S107", "S108", "S109"]
}
PREREQUISITES = {
    'Geometry': ['Algebra'],
    'Calculus': ['Algebra', 'Geometry'],
    'Chemistry': ['Algebra'],
    'Biology': ['Chemistry'],
    'Chemical Bonding': ['Chemistry'],
    'Kinematics': ['Algebra'],
    'DNA Replication': ['Biology'],
    'Gas Laws': ['Chemistry'],
    'Derivatives': ['product_rule','Chain_rule']
}
TOPICS = [
    'Algebra', 'Geometry', 'Calculus', 'Chemistry', 'Biology',
    'Chemical Bonding', 'Kinematics', 'DNA Replication', 'Gas Laws','Derivatives'
]
SUBTOPICS = {
    'Algebra': ['Equations', 'Inequalities', 'Polynomials'],
    'Geometry': ['Angles', 'Shapes', 'Trigonometry'],
    'Calculus': ['Limits', 'Integrals'],
    'Chemistry': ['Elements', 'Reactions', 'Compounds','Chemical Bonding', 'Gas Laws'],
    'Biology': ['Cells', 'Genetics', 'Ecology', 'DNA Replication'],
    'Physics': ['Kinematics'],
    'Derivatives': ['Rules', 'Chain Rule', 'Applications'],
'DNA Replication': ['DNA Structure'],
    'Kinematics': ['Motion Equations']

}

# Application-level edges
APPLICATION_RELATIONS = {
    # Original entries
    'Biomolecules': {'base_topic': 'Chemical Bonding', 'type': 'application'},
    'Optimization': {'base_topic': 'Derivatives', 'type': 'application'},

    # New additions
    'Projectile Motion': {
        'base_topic': 'Kinematics',
        'type': 'application',
        'description': 'Application of kinematic equations to parabolic trajectories'
    },
    'Genetic Engineering': {
        'base_topic': 'DNA Replication',
        'type': 'application',
        'description': 'Practical applications of DNA manipulation techniques'
    },
    'Economic Modeling': {
        'base_topic': 'Derivatives',
        'type': 'application',
        'description': 'Using calculus to model market trends and optimization'
    },
    'Greenhouse Effect': {
        'base_topic': 'Gas Laws',
        'type': 'application',
        'description': 'Application of gas behavior principles to atmospheric science'
    }
}

FORMULA_QUIZ_BANK = {
    'Algebra': {
        'Quadratic Formula': [
            {
                "id": "alg_q1",
                "question": "The quadratic formula for ax² + bx + c = 0 is?",
                "type": "formula",
                "answer": "x = (-b ± √(b² - 4ac)) / 2a",
                "solution_steps": ["Recall the standard form of a quadratic equation and the derivation of the formula."]
            },
            {
                "id": "alg_q2",
                "question": "What is the discriminant of a quadratic equation and what does it indicate about the roots?",
                "type": "definition",
                "answer": "Discriminant = b² - 4ac. If > 0, two real roots; if = 0, one real root; if < 0, two complex roots.",
                "solution_steps": ["Define the discriminant and explain its relationship to the nature of the roots."]
            }
        ],
        'Logarithm Properties': [
            {
                "id": "alg_q3",
                "question": "State the product rule of logarithms: log_b(mn) = ?",
                "type": "formula",
                "answer": "log_b(m) + log_b(n)",
                "solution_steps": ["Recall the property that relates the logarithm of a product to the sum of logarithms."]
            },
            {
                "id": "alg_q4",
                "question": "State the power rule of logarithms: log_b(m^n) = ?",
                "type": "formula",
                "answer": "n * log_b(m)",
                "solution_steps": ["Recall the property that allows the exponent inside a logarithm to be brought out as a multiplier."]
            }
        ]
    },
    'Calculus': {
        'Integration Formulas': [
            {
                "id": "calc_q3",
                "question": "∫sin(x) dx = ?",
                "type": "formula",
                "answer": "-cos(x) + C",
                "solution_steps": ["Recall the integral of the sine function."]
            },
            {
                "id": "calc_q4",
                "question": "Formula for integration by substitution?",
                "type": "definition",
                "answer": "∫f(g(x))g'(x) dx = ∫f(u) du, where u = g(x)",
                "solution_steps": ["State the method for simplifying integrals by changing the variable."]
            }
        ]
    },
    'Chemistry': {
        'Gas Laws': [
            {
                "id": "chem_q1",
                "question": "State Boyle's Law.",
                "type": "definition",
                "answer": "For a fixed amount of gas at constant temperature, the pressure is inversely proportional to the volume (P₁V₁ = P₂V₂).",
                "solution_steps": ["Recall the relationship between pressure and volume of a gas at constant temperature."]
            },
            {
                "id": "chem_q2",
                "question": "The Ideal Gas Law formula is?",
                "type": "formula",
                "answer": "PV = nRT",
                "solution_steps": ["Recall the equation relating pressure, volume, number of moles, ideal gas constant, and temperature."]
            }
        ],
        'Chemical Formulas': [
            {
                "id": "chem_q3",
                "question": "The chemical formula for water is?",
                "type": "formula",
                "answer": "H₂O",
                "solution_steps": ["Recall the standard chemical formula for water."]
            },
            {
                "id": "chem_q4",
                "question": "The chemical formula for glucose is?",
                "type": "formula",
                "answer": "C₆H₁₂O₆",
                "solution_steps": ["Recall the standard chemical formula for glucose."]
            }
        ]
    },
    'Biology': {
        'Cell Biology Formulas': [
            {
                "id": "bio_q1",
                "question": "What is the formula for calculating magnification of a microscope?",
                "type": "formula",
                "answer": "Total Magnification = Magnification of Eyepiece Lens × Magnification of Objective Lens",
                "solution_steps": ["Recall how the magnifying powers of different lenses in a microscope combine."]
            },
            {
                "id": "bio_q2",
                "question": "What is the formula for calculating the growth rate of a population?",
                "type": "formula",
                "answer": "Growth Rate (r) = (Births - Deaths) / Initial Population",
                "solution_steps": ["Recall the basic factors that influence population growth."]
            }
        ],
        'Genetics Formulas': [
            {
                "id": "bio_q3",
                "question": "In Hardy-Weinberg equilibrium, what does the equation p + q = 1 represent?",
                "type": "definition",
                "answer": "It represents the sum of the frequencies of the two alleles for a particular trait in a population.",
                "solution_steps": ["Recall the basic principle of allele frequencies in a population."]
            },
            {
                "id": "bio_q4",
                "question": "In Hardy-Weinberg equilibrium, what does the equation p² + 2pq + q² = 1 represent?",
                "type": "definition",
                "answer": "It represents the sum of the frequencies of the homozygous dominant (p²), heterozygous (2pq), and homozygous recessive (q²) genotypes in a population.",
                "solution_steps": ["Recall how allele frequencies relate to genotype frequencies under Hardy-Weinberg equilibrium."]
            }
        ],
        'Photosynthesis Formulas': [
            {
                "id": "bio_q5",
                "question": "What is the overall balanced chemical equation for photosynthesis?",
                "type": "formula",
                "answer": "6CO₂ + 6H₂O + Light Energy → C₆H₁₂O₆ + 6O₂",
                "solution_steps": ["Recall the reactants and products of the photosynthesis process."]
            }
        ],
        'Respiration Formulas': [
            {
                "id": "bio_q6",
                "question": "What is the overall balanced chemical equation for aerobic respiration?",
                "type": "formula",
                "answer": "C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + Energy (ATP)",
                "solution_steps": ["Recall the reactants and products of the aerobic respiration process."]
            }
        ]
    },
    'Physics': {
        'Kinematics Formulas': [
            {
                "id": "phy_q1",
                "question": "State the first equation of motion.",
                "type": "formula",
                "answer": "v = u + at",
                "solution_steps": ["Recall the relationship between final velocity, initial velocity, acceleration, and time."]
            },
            {
                "id": "phy_q2",
                "question": "State the second equation of motion.",
                "type": "formula",
                "answer": "s = ut + (1/2)at²",
                "solution_steps": ["Recall the relationship between displacement, initial velocity, acceleration, and time."]
            },
            {
                "id": "phy_q3",
                "question": "State the third equation of motion.",
                "type": "formula",
                "answer": "v² = u² + 2as",
                "solution_steps": ["Recall the relationship between final velocity, initial velocity, acceleration, and displacement."]
            }
        ],
        'Newton\'s Laws of Motion': [
            {
                "id": "phy_q4",
                "question": "State Newton's second law of motion.",
                "type": "definition",
                "answer": "The acceleration of an object is directly proportional to the net force acting on the object and inversely proportional to its mass (F = ma).",
                "solution_steps": ["Recall the relationship between force, mass, and acceleration."]
            }
        ],
        'Optics Formulas': [
            {
                "id": "phy_q5",
                "question": "The lens formula is?",
                "type": "formula",
                "answer": "1/f = 1/v - 1/u",
                "solution_steps": ["Recall the relationship between focal length, image distance, and object distance for a lens."]
            },
            {
                "id": "phy_q6",
                "question": "The formula for magnification (m) produced by a lens is?",
                "type": "formula",
                "answer": "m = v/u",
                "solution_steps": ["Recall the relationship between image height/distance and object height/distance."]
            }
        ]
    }
}
if "quiz_progress" not in st.session_state:
    st.session_state.quiz_progress = {}
QUESTION_BANK = {
    'Algebra': [
        {
            "id": "alg_1",
            "text": "Solve for x: 2x + 5 = 13",
            "difficulty": 1,
            "type": "free_response",
            "answer": "4",
            "solution_steps": ["Subtract 5 from both sides: 2x = 8", "Divide by 2: x = 4"]
        },
        {
            "id": "alg_2",
            "text": "Factor: x² - 9",
            "difficulty": 2,
            "type": "multiple_choice",
            "options": ["(x+3)(x-3)", "(x+3)²", "(x-3)²", "Prime"],
            "correct_option": 0
        },
        {
            "id": "alg_3",
            "text": "Solve the system: 2x + y = 7, x - y = 1",
            "difficulty": 3,
            "type": "free_response",
            "answer": "x=2, y=3",
            "solution_steps": ["Add the two equations: 3x = 9 => x = 3", "Substitute x in the second equation: 3 - y = 1 => y = 2"]
        }
    ],
    'Geometry': [
        {
            "id": "geo_1",
            "text": "Find the area of a circle with radius 5",
            "difficulty": 1,
            "type": "free_response",
            "answer": "78.54",
            "solution_steps": ["Area of a circle = πr²", "Substitute r = 5: Area = π(5)² = 25π ≈ 78.54"]
        },
        {
            "id": "geo_2",
            "text": "Prove triangles ABC and DEF are similar",
            "difficulty": 3,
            "type": "essay",
            "answer_guidance": "Students should provide logical steps and justifications based on similarity postulates (e.g., AA, SAS, SSS).",
            "rubric": {
                "Understanding of Similarity": 2,
                "Correct Application of Postulates": 3,
                "Logical Reasoning": 3,
                "Clarity of Explanation": 2
            }
        },
        {
            "id": "geo_3",
            "text": "Calculate the volume of a cone with height 6 and radius 2",
            "difficulty": 2,
            "type": "free_response",
            "answer": "25.13",
            "solution_steps": ["Volume of a cone = (1/3)πr²h", "Substitute r = 2 and h = 6: Volume = (1/3)π(2)²(6) = 8π ≈ 25.13"]
        }
    ],
    'Calculus': [
        {
            "id": "calc_1",
            "text": "Find the derivative of f(x) = x³ + 2x²",
            "difficulty": 2,
            "type": "free_response",
            "answer": "3x² + 4x",
            "solution_steps": ["Apply the power rule: d/dx(xⁿ) = nxⁿ⁻¹", "d/dx(x³) = 3x²", "d/dx(2x²) = 4x", "So, f'(x) = 3x² + 4x"]
        },
        {
            "id": "calc_2",
            "text": "Evaluate ∫(2x + 1)dx from 0 to 3",
            "difficulty": 2,
            "type": "free_response",
            "answer": "12",
            "solution_steps": ["Find the antiderivative: ∫(2x + 1)dx = x² + x + C", "Evaluate at the limits: [(3)² + (3)] - [(0)² + (0)] = (9 + 3) - 0 = 12"]
        },
        {
            "id": "calc_3",
            "text": "Find the inflection points of f(x) = x³ - 6x²",
            "difficulty": 3,
            "type": "free_response",
            "answer": "x = 2",
            "solution_steps": ["Find the second derivative: f'(x) = 3x² - 12x, f''(x) = 6x - 12", "Set the second derivative to zero: 6x - 12 = 0 => x = 2", "Check the concavity change around x = 2"]
        }
    ],
    'Chemistry': [
        {
            "id": "chem_1",
            "text": "Balance: H₂ + O₂ → H₂O",
            "difficulty": 1,
            "type": "free_response",
            "answer": "2H₂ + O₂ → 2H₂O",
            "solution_steps": ["Count the number of atoms of each element on both sides.", "Adjust coefficients to balance the number of atoms."]
        },
        {
            "id": "chem_2",
            "text": "Calculate pH of 0.01M HCl solution",
            "difficulty": 2,
            "type": "free_response",
            "answer": "2",
            "solution_steps": ["HCl is a strong acid, so [H⁺] = concentration of HCl = 0.01M = 10⁻² M", "pH = -log₁₀[H⁺] = -log₁₀(10⁻²) = 2"]
        },
        {
            "id": "chem_3",
            "text": "Draw the Lewis structure for CO₂",
            "difficulty": 2,
            "type": "drawing",
            "answer_guidance": "Students should depict a central carbon atom double-bonded to two oxygen atoms, with lone pairs on the oxygen atoms.",
            "elements": ["C", "O"],
            "bonds": ["double", "double"],
            "lone_pairs": {"O": 4}
        }
    ],
    'Biology': [
        {
            "id": "bio_1",
            "text": "List the phases of mitosis in order",
            "difficulty": 1,
            "type": "ordering",
            "options": ["Prophase", "Metaphase", "Anaphase", "Telophase"],
            "correct_order": [0, 1, 2, 3]
        },
        {
            "id": "bio_2",
            "text": "Explain how DNA replication works",
            "difficulty": 2,
            "type": "essay",
            "answer_guidance": "Students should describe the roles of enzymes like helicase and polymerase, the concept of semi-conservative replication, and the directionality of synthesis.",
            "rubric": {
                "Identification of Key Enzymes": 2,
                "Explanation of Semi-Conservative Nature": 3,
                "Understanding of Directionality": 2,
                "Overall Clarity and Coherence": 3
            }
        },
        {
            "id": "bio_3",
            "text": "Calculate Hardy-Weinberg equilibrium for a population with allele frequencies p = 0.6 and q = 0.4 for a gene with two alleles.",
            "difficulty": 3,
            "type": "free_response",
            "answer": "p² = 0.36, 2pq = 0.48, q² = 0.16",
            "solution_steps": ["Hardy-Weinberg equations: p² + 2pq + q² = 1", "p² represents the frequency of the homozygous dominant genotype: (0.6)² = 0.36", "2pq represents the frequency of the heterozygous genotype: 2 * (0.6) * (0.4) = 0.48", "q² represents the frequency of the homozygous recessive genotype: (0.4)² = 0.16"]
        }
        ],
'Derivatives': [
        {
            "id": "deriv_1",
            "text": "Find the derivative of f(x) = 3x² + 2x",
            "difficulty": 2,
            "type": "free_response",
            "answer": "6x + 2",
            "solution_steps": ["Apply power rule to each term"]
        }
    ]
}
# Question response tracking
if "question_responses" not in st.session_state:
    st.session_state.question_responses = {}

# Recommendation dictionaries
BRIDGE_COURSES = {
    'Algebra': '/courses/algebra-fundamentals',
    'Geometry': '/courses/geometry-basics',
    'Calculus': '/courses/calculus-intro',
    'Chemistry': '/courses/chemistry-101',
    'Biology': '/courses/biology-essentials'
}
HOTS_QUESTIONS = {
    'Algebra': ['Polynomial Analysis', 'Equation Systems'],
    'Geometry': ['Geometric Proofs', 'Coordinate Geometry Applications'],
    'Calculus': ['Optimization Problems', 'Related Rates'],
    'Chemistry': ['Reaction Predictions', 'Equilibrium Analysis'],
    'Biology': ['Ecosystem Modeling', 'Genetic Expression Analysis']
}
PRACTICE_QUESTIONS = {
    'Algebra': {
        'recent': ['Factoring Polynomials', 'Solving Quadratics'],
        'historical': ['Linear Equations', 'Basic Operations'],
        'fundamental': ['Number Sense', 'Order of Operations']
    },
    'Geometry': {
        'recent': ['Triangle Congruence', 'Circle Theorems'],
        'historical': ['Angle Relationships', 'Pythagorean Theorem'],
        'fundamental': ['Basic Shapes', 'Area Formulas']
    },
    'Calculus': {
        'recent': ['Derivative Rules', 'Integration Techniques'],
        'historical': ['Limits', 'Continuity'],
        'fundamental': ['Function Behavior', 'Graphing']
    },
    'Chemistry': {
        'recent': ['Balancing Equations', 'Stoichiometry'],
        'historical': ['Periodic Table', 'Chemical Bonds'],
        'fundamental': ['States of Matter', 'Element Properties']
    },
    'Biology': {
        'recent': ['Cell Division', 'Heredity'],
        'historical': ['Cell Structure', 'Classification'],
        'fundamental': ['Life Processes', 'Scientific Method']
    }
}

MEDIA_LINKS = {
    'Algebra': {
        'videos': ['https://example.com/algebra-visual', 'https://example.com/equation-solving'],
        'analogies': 'Solving equations is like finding balance on a seesaw'
    },
    'Geometry': {
        'videos': ['https://example.com/geometry-shapes', 'https://example.com/trigonometry-circles'],
        'analogies': 'Geometric transformations are like moving furniture in a room'
    },
    'Calculus': {
        'videos': ['https://example.com/calculus-intro', 'https://example.com/derivative-explained'],
        'analogies': 'Derivatives are like speedometers that show rate of change'
    },
    'Chemistry': {
        'videos': ['https://example.com/chemistry-atoms', 'https://example.com/reactions-explained'],
        'analogies': 'Chemical reactions are like cooking recipes with precise ingredients'
    },
    'Biology': {
        'videos': ['https://example.com/biology-cells', 'https://example.com/genetics-basics'],
        'analogies': 'DNA is like a recipe book for building living organisms'
    }
}
EASY_TOPICS = {
    'Algebra': ['Real-life Equation Examples', 'Math Puzzles Using Algebra', 'Algebra in Games'],
    'Geometry': ['Sacred Geometry in Art', 'Geometry in Nature', 'Optical Illusions Explained'],
    'Calculus': ['Calculus in Sports', 'Real-world Optimization', 'Visual Calculus Games'],
    'Chemistry': ['Kitchen Chemistry Demos', 'Chemistry Magic Tricks', 'Everyday Reactions'],
    'Biology': ['Amazing Animal Adaptations', 'Biology Mysteries', 'Human Body Curiosities']
}

# Formula revision materials
FORMULA_REVISION = {
    'Algebra': ['Algebraic Identities Cheat Sheet', 'Common Formula Flashcards', 'Formula Quiz App'],
    'Geometry': ['Geometric Formula Cards', 'Interactive Shape Calculator', 'Trigonometry Formula Wheel'],
    'Calculus': ['Calculus Formula Reference', 'Derivative/Integral Pairs', 'Series Convergence Tests'],
    'Chemistry': ['Periodic Table Helper', 'Reaction Types Summary', 'Equilibrium Constants Guide'],
    'Biology': ['Genetic Inheritance Patterns', 'Ecological Formulas', 'Biochemical Pathways Summary']
}

# Motivation boosting quotes by subject
MOTIVATION_QUOTES = {
    'Algebra': "Algebra is not about numbers, equations, computations or algorithms; it's about understanding.",
    'Geometry': "Geometry is knowledge of the eternally existent.",
    'Calculus': "Calculus does to algebra what algebra did to arithmetic.",
    'Chemistry': "Chemistry is the bridge between physics of atoms and life.",
    'Biology': "Biology is the most powerful technology ever created. DNA is software, proteins are hardware."
}
# ==================================================================
# 1. Data Generation
# ==================================================================
@st.cache_data
def generate_student_data(num_students=500):
    np.random.seed(42)
    rows = []
    hardwired_students = [
        {'skill': 0.95, 'mot': 'High', 'correct_rate': 0.9, 'time_factor': 0.7},
        {'skill': 0.65, 'mot': 'Medium', 'correct_rate': 0.58, 'time_factor': 1.0},
        {'skill': 0.25, 'mot': 'Low', 'correct_rate': 0.35, 'time_factor': 1.5}
    ]

    # Helper function for safe subtopic sampling
    def get_subtopics(subtopic_list):
        try:
            if not subtopic_list or len(subtopic_list) == 0:
                return [None] * 3

            # Always return 3 elements, using replacement if needed
            replace = len(subtopic_list) < 3
            sampled = np.random.choice(
                subtopic_list,
                size=3,
                replace=replace
            )
            return list(sampled)
        except Exception as e:
            st.error(f"Subtopic sampling error: {str(e)}")
            return [None] * 3

    # Hardwired students generation
    for sid in range(3):
        hw = hardwired_students[sid]
        exam_date = datetime.now() - timedelta(days=30)
        done = ['Algebra', 'Geometry', 'Chemical Bonding'] if sid == 0 else ['Algebra']

        for topic in TOPICS:
            subtopics = SUBTOPICS.get(topic, [])
            for _ in range(5):  # Fixed interaction count
                subs = get_subtopics(subtopics)
                # Force performance characteristics
                corr = np.random.binomial(1, hw['correct_rate'])
                tkt = max(0.1, np.random.gamma(2, 1.5) * hw['time_factor'])
                rows.append({
                    'StudentID': sid,
                    'Topic': topic,
                    'Subtopic1': subs[0],
                    'Subtopic2': subs[1],
                    'Subtopic3': subs[2],
                    'Weight1': 0.4,
                    'Weight2': 0.3,
                    'Weight3': 0.3,
                    'Correct': corr,
                    'TimeTaken': tkt,
                    'ExamDate': exam_date,
                    'Completed': topic in done,
                    'MotivationLevel': hw['mot'],
                    'VideosWatched': 5 if sid == 0 else 3,
                    'QuizzesTaken': 4 if sid == 0 else 1,
                    'PracticeSessions': 6 if sid == 0 else 2,
                    'MediaClicks': 3,
                    'QuizProgress': 0
                })

    # Regular students generation
    num_students = max(1, num_students)
    for sid in range(num_students):
        exam_date = datetime.now() + timedelta(days=np.random.randint(7, 60))
        skill = np.clip(np.random.beta(2, 1.5), 0.01, 0.99)
        mot = np.random.choice(['High', 'Medium', 'Low'], p=[0.5, 0.3, 0.2])

        # Safe topic completion handling
        done = []
        if TOPICS:
            try:
                done = np.random.choice(
                    TOPICS,
                    size=np.random.randint(0, len(TOPICS) + 1),
                    replace=False
                ).tolist()
            except ValueError as e:
                st.error(f"Topic selection error: {str(e)}")
                done = []

        for topic in TOPICS:
            # Safe interaction count calculation
            interaction_count = min(max(np.random.poisson(3) + 1, 1), 10)

            for _ in range(interaction_count):
                subtopics = SUBTOPICS.get(topic, [])
                subs = get_subtopics(subtopics)

                # Dirichlet distribution safety
                try:
                    wts = np.random.dirichlet(np.ones(3))
                    wts /= wts.sum()
                except Exception as e:
                    wts = [0.4, 0.3, 0.3]
                    st.error(f"Weight generation error: {str(e)}")

                # Quiz progress validation
                quiz_bank = FORMULA_QUIZ_BANK.get(topic, {})
                qp_max = max(1, len(quiz_bank))
                qp = np.random.randint(0, qp_max) if qp_max > 0 else 0
                # Motivation factor with bounds
                mf = {'High': 1.5, 'Medium': 1.0, 'Low': 0.7}.get(mot, 1.0)

                # Binomial success with probability clamping
                success_prob = skill * mf
                corr = np.random.binomial(1, np.clip(success_prob, 0.01, 0.99))

                # Time taken with gamma distribution safeguards
                tkt = max(0.1, np.random.gamma(2, 1.5) * (2 - mf))

                # Usage metrics with validation
                vids = max(0, np.random.poisson(1))
                qz = max(0, np.random.binomial(1, 0.3))
                prac = max(0, np.random.poisson(2))
                media = max(0, np.random.poisson(1))
                rows.append({
                    'StudentID': sid,
                    'Topic': topic,
                    'Subtopic1': subs[0],
                    'Subtopic2': subs[1],
                    'Subtopic3': subs[2],
                    'Weight1': wts[0],
                    'Weight2': wts[1],
                    'Weight3': wts[2],
                    'Correct': corr,
                    'TimeTaken': tkt,
                    'ExamDate': exam_date,
                    'Completed': topic in done,
                    'MotivationLevel': mot,
                    'VideosWatched': vids,
                    'QuizzesTaken': qz,
                    'PracticeSessions': prac,
                    'MediaClicks': media,
                    'QuizProgress': qp
                })

    return pd.DataFrame(rows)


def validate_answer(question, student_answer):
    """Enhanced validation with error handling"""
    try:
        if question['type'] == 'free_response':
            correct = str(student_answer).strip() == str(question.get('answer', '')).strip()
            solution = question.get('solution_steps', 'No solution available')
            return correct, f"Solution: {solution}"

        elif question['type'] == 'multiple_choice':
            options = question.get('options', [])
            correct_index = question.get('correct_option', -1)
            if correct_index < 0 or correct_index >= len(options):
                return False, "Invalid question configuration"
            return student_answer == correct_index, f"Correct answer: {options[correct_index]}"

        return False, "Unsupported question type"

    except Exception as e:
        st.error(f"Validation error: {str(e)}")
        return False, "Error evaluating answer"
# ==================================================================
# 2. Student Segmentation & Peer Insights
# ==================================================================
def segment_students(df):
    perf = df.groupby('StudentID').agg(
        acc=('Correct', 'mean'),
        time=('TimeTaken', 'mean')
    ).reset_index()

    if perf.empty:
        return pd.DataFrame(columns=['StudentID', 'acc', 'time', 'cluster', 'label'])

    # Robust clustering initialization
    n_clusters = min(3, max(1, len(perf)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')

    # Handle edge cases
    if len(perf) == 1:
        return pd.DataFrame({
            'StudentID': perf.StudentID,
            'acc': perf.acc,
            'time': perf.time,
            'cluster': [0],
            'label': ['Average']
        })

    try:
        perf['cluster'] = kmeans.fit_predict(perf[['acc', 'time']])
    except Exception as e:
        perf['cluster'] = 0

    # Hard-coded demo student configuration
    demo_mapping = {
        0: ('Topper', 0),
        1: ('Average', 1),
        2: ('Poor', 2)
    }

    # Assign clusters and labels for demo students
    for sid, (label, cluster) in demo_mapping.items():
        if sid in perf.StudentID.values:
            perf.loc[perf.StudentID == sid, 'cluster'] = cluster
            perf.loc[perf.StudentID == sid, 'label'] = label

    # Label non-demo students using clustering results
    non_demo = perf[~perf.StudentID.isin([0, 1, 2])]
    if not non_demo.empty:
        cluster_stats = non_demo.groupby('cluster')['acc'].agg(['mean', 'count'])
        valid_clusters = cluster_stats[cluster_stats['mean'] > 0]

        if not valid_clusters.empty:
            ranked_clusters = valid_clusters.sort_values(
                ['mean', 'count'],
                ascending=[False, False]
            ).reset_index()

            cluster_label_map = {
                row['cluster']: ['Topper', 'Average', 'Poor'][i]
                for i, row in ranked_clusters.iterrows()
                if i < 3
            }

            perf.loc[non_demo.index, 'label'] = non_demo['cluster'].map(
                lambda x: cluster_label_map.get(x, 'Needs Help')
            )

    return perf.fillna({'label': 'Needs Help'})
def recommend_topper_resources(seg, df):
    # Return empty list if no data
    if seg.empty or df.empty:
        return []

    usage = df.groupby('StudentID').agg(Videos=('VideosWatched', 'sum'),
                                        Quizzes=('QuizzesTaken', 'sum'),
                                        Practice=('PracticeSessions', 'sum'),
                                        Media=('MediaClicks', 'sum')).reset_index()

    usage = usage.merge(seg[['StudentID', 'label']], on='StudentID', how='inner')

    # Check if we have toppers
    toppers = usage[usage.label == 'Topper']
    if toppers.empty:
        return ["No top performers identified yet."]

    topper_means = toppers.mean(numeric_only=True).sort_values(ascending=False)

    # Get top 3 metrics (excluding StudentID)
    if 'StudentID' in topper_means:
        topper_means = topper_means.drop('StudentID')

    top3 = topper_means.head(3)
    return [f"Top performers average {v:.1f} {k}" for k, v in top3.items()]

def calculate_odds_ratio(table):
    """Calculate odds ratio from a 2x2 contingency table."""
    try:
        a, b = table[0]
        c, d = table[1]
        # Apply Haldane-Anscombe correction (add 0.5 to all cells)
        # This handles zeros in the contingency table
        return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    except Exception:
        return float('nan')


def progression_summary(df, student1, student2, time_tolerance=0.15, perf_gap=0.2):
    """
    Compare two students with:
    - Similar start dates (±15 days)
    - Similar total time spent (±15%)
    - Academic performance gap >20%
    """
    # Get student data
    s1_data = df[df.StudentID == student1]
    s2_data = df[df.StudentID == student2]

    # Calculate first interaction dates
    s1_start = s1_data.ExamDate.min()
    s2_start = s2_data.ExamDate.min()

    # Calculate total time spent (in hours)
    s1_duration = s1_data.TimeTaken.sum()
    s2_duration = s2_data.TimeTaken.sum()

    # Calculate performance metrics
    s1_perf = s1_data.Correct.mean()
    s2_perf = s2_data.Correct.mean()

    insights = []

    # 1. Validate time alignment
    if abs((s1_start - s2_start).days) > 40:
        return ["Students started more than 15 days apart - not comparable"]

    # 2. Validate time investment
    duration_ratio = abs(s1_duration - s2_duration) / max(s1_duration, s2_duration)
    if duration_ratio > time_tolerance:
        return [f"Time spent differs by {duration_ratio:.0%} - beyond {time_tolerance:.0%} threshold"]

    # 3. Validate performance gap
    perf_diff = abs(s1_perf - s2_perf)
    if perf_diff < perf_gap:
        return [f"Performance difference {perf_diff:.0%} < {perf_gap:.0%} threshold"]

    # Identify better performer
    better_student = student1 if s1_perf > s2_perf else student2
    weaker_student = student2 if better_student == student1 else student1

    # Compare usage patterns
    metrics = ['VideosWatched', 'QuizzesTaken',
               'PracticeSessions', 'MediaClicks']

    comparisons = []
    for metric in metrics:
        s1_val = s1_data[metric].sum()
        s2_val = s2_data[metric].sum()
        diff = s1_val - s2_val

        comparisons.append({
            'metric': metric,
            'better': max(s1_val, s2_val),
            'weaker': min(s1_val, s2_val),
            'diff': abs(diff),
            'direction': 'higher' if diff > 0 else 'lower'
        })

    # Sort by largest absolute differences
    comparisons.sort(key=lambda x: x['diff'], reverse=True)

    # Generate insights
    insights.append(
        f"🏆 Better Performer: Student {better_student} ({s1_perf if better_student == student1 else s2_perf:.0%} vs {s2_perf if better_student == student1 else s1_perf:.0%})")

    for comp in comparisons[:3]:  # Top 3 differences
        insights.append(
            f"📊 {comp['metric']}: {comp['better']} vs {comp['weaker']} "
            f"({comp['direction']} by {comp['diff']})"
        )

    # Add strategic recommendations
    top_diff = comparisons[0]
    insights.append(
        f"🚀 Recommendation: Focus on increasing {top_diff['metric'].lower()} "
        f"activities by {top_diff['diff']} sessions/week"
    )

    return insights

# ==================================================================
# 3. Knowledge Graph Construction
# ==================================================================
def build_knowledge_graph(prereqs, df, topics, OR_thresh=2.0, SHAP_thresh=0.01,
                              min_count=20, application_relations=None):
    if application_relations is None:
        application_relations = APPLICATION_RELATIONS.copy()
    """
    Constructs a knowledge graph combining prerequisite relationships, application connections,
    and statistically validated topic relationships from student performance data.
    """
    G = nx.DiGraph()

    # Initialize all topic nodes
    for topic in topics:
        G.add_node(topic, type='topic')

    # Add application relationships with validation
    for app_key, app_info in application_relations.items():
        base_topic = app_info['base_topic']

        # Validate base topic exists in core topics
        if base_topic not in G.nodes:
            st.warning(f"⚠️ Missing base topic '{base_topic}' for application '{app_key}'. Skipping...")
            continue

        # Create application node if not exists
        if app_key not in G.nodes:
            G.add_node(app_key, type='application')

        # Add relationship with validation
        if G.has_node(base_topic) and G.has_node(app_key):
            G.add_edge(base_topic, app_key,
                       relation='application',
                       weight=2.5,
                       description=app_info.get('description', ''))
        else:
            st.error(f"❌ Failed to create application relationship between {base_topic} and {app_key}")

        # Add prerequisite relationships
        for target, requirements in prereqs.items():
            for req in requirements:
                if req in G.nodes and target in G.nodes:
                    G.add_edge(req, target, relation='prereq', weight=3.0)
                else:
                    st.warning(f"Missing prerequisite node(s) for {req} -> {target}")

    # Add application relationships
    for app_key, app_info in application_relations.items():
        G.add_edge(app_info['base_topic'], app_key,
                   relation='application', weight=2.5)

    if df.empty:
        return G

    try:
        # Student struggle analysis
        grp = df.groupby(['StudentID', 'Topic']).agg(
            attempts=('Correct', 'count'),
            correct=('Correct', 'sum'),
            time=('TimeTaken', 'mean')
        ).reset_index()
        grp['struggle'] = (grp.attempts >= 2) & (grp.correct / grp.attempts < 0.5)

        # Create aligned data structures
        struggle_matrix = grp.pivot(index='StudentID', columns='Topic',
                                    values='struggle').fillna(False)
        student_ids = struggle_matrix.index
        proficiency = (grp.groupby('StudentID').correct.sum() /
                       grp.groupby('StudentID').attempts.sum()).reindex(student_ids)
        avg_time = grp.groupby('StudentID').time.mean().reindex(student_ids)

        # Statistical relationship detection
        for topic_a, topic_b in combinations(topics, 2):
            if topic_a not in struggle_matrix.columns or topic_b not in struggle_matrix.columns:
                continue

            valid_mask = struggle_matrix[topic_a].notna() & struggle_matrix[topic_b].notna()
            valid_students = struggle_matrix[valid_mask].index

            if len(valid_students) < min_count:
                continue

            try:
                X = pd.DataFrame({
                    'topic_a': struggle_matrix.loc[valid_students, topic_a].astype(int),
                    'proficiency': proficiency.loc[valid_students],
                    'time': avg_time.loc[valid_students]
                }).dropna()

                y = struggle_matrix.loc[valid_students, topic_b].astype(int).loc[X.index]

                if len(X) < 10 or y.nunique() < 2:
                    continue

                # Adjusted odds ratio calculation
                or_value = np.nan
                if len(y) >= 50:  # Regularized logistic regression
                    lr = LogisticRegression(penalty='l2', C=0.1, max_iter=1000, random_state=42)
                    lr.fit(X[['topic_a', 'proficiency', 'time']], y)
                    or_value = np.exp(lr.coef_[0][0])
                else:  # Stratified analysis
                    try:
                        strata = pd.qcut(X.proficiency, q=3, duplicates='drop')
                        if len(strata.unique()) >= 2:
                            # Create contingency table
                            contingency_table = np.array([
                                [np.sum((X.topic_a == 1) & (y == 1)), np.sum((X.topic_a == 1) & (y == 0))],
                                [np.sum((X.topic_a == 0) & (y == 1)), np.sum((X.topic_a == 0) & (y == 0))]
                            ])
                            table = StratifiedTable.from_data(
                                var1='topic_a',
                                var2=y.name if isinstance(y, pd.Series) else 'topic_b',
                                strata='proficiency',
                                data=X.assign(topic_b=y)
                            )

                            or_value = table.oddsratio_pooled
                    except Exception as e:
                        st.error(f"Stratified analysis failed: {str(e)}")
                        or_value = np.nan

                if or_value > OR_thresh and not np.isnan(or_value):
                    G.add_edge(topic_a, topic_b, relation='odds_ratio',
                               weight=min(or_value, 5.0))

                # SHAP feature importance - fixed to ensure it runs
                if len(y) >= 100:  # Only run XGBoost on larger samples
                    try:
                        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                            random_state=42)
                        xgb.fit(X, y)

                        explainer = shap.TreeExplainer(xgb)
                        shap_values = explainer.shap_values(X)

                        a_importance = np.abs(shap_values[:, 0]).mean()
                        if a_importance > SHAP_thresh:
                            G.add_edge(topic_a, topic_b, relation='shap_importance',
                                       weight=min(a_importance * 10, 4.0))
                    except Exception as e:
                        st.error(f"SHAP analysis failed: {str(e)}")

            except Exception as e:
                st.error(f"Relationship analysis failed for {topic_a}-{topic_b}: {str(e)}")

    except Exception as e:
        st.error(f"Graph construction failed: {str(e)}")
        return G

    # Subtopic integration
    for topic in topics:
        topic_data = df[df.Topic == topic]
        if topic_data.empty:
            continue

        failures = topic_data[topic_data.Correct == 0]
        subtopic_weights = Counter()
        for _, row in failures.iterrows():
            for i in (1, 2, 3):
                subtopic = row.get(f'Subtopic{i}')
                weight = row.get(f'Weight{i}')
                if pd.notna(subtopic) and pd.notna(weight):
                    subtopic_weights[subtopic] += weight

            for subtopic, weight in subtopic_weights.most_common(2):
                if subtopic not in G.nodes:
                    G.add_node(subtopic, type='subtopic')

                G.add_edge(topic, subtopic, relation='subtopic',
                           weight=int(min(weight, 5)))

            # Connect to prerequisites
            for prereq in G.predecessors(topic):
                if prereq != subtopic:
                    G.add_edge(prereq, subtopic, relation='sub_prereq',
                               weight=1.5)

            # Connect to applications
            if subtopic in application_relations:
                app_info = application_relations[subtopic]
                if app_info['base_topic'] in G.nodes:
                    G.add_edge(subtopic, app_info['base_topic'],
                               relation='app_preparation', weight=2.0)

    return G
def apply_dual_tier_scoring(G):
    """
    Applies tiered scoring to edges based on relationship type and node connectivity.
    Maintains original tier logic as it effectively captures edge importance.
    """
    for source, target, data in G.edges(data=True):
        # Base score from relationship type
        base_scores = {
            'prereq': 3,
            'application': 1,
            'odds_ratio': 2,
            'shap_importance': 2.5,
            'subtopic': 1,
            'sub_prereq': 1,
            'app_preparation': 1
        }
        score = base_scores.get(data.get('relation', 'other'), 1)

        # Bonus for important nodes (number of incoming edges)
        incoming_edges = len([e for e in G.in_edges(target) if e[0] != source])
        bonus = min(2, incoming_edges)

        # Tier assignment
        total = score + bonus
        if total >= 4:
            data['tier'] = 1
        elif total >= 2:
            data['tier'] = 2
        else:
            data['tier'] = 3  # Lowest priority tier

    return G


def update_knowledge_graph_with_quiz(G, sid, topic):
    """Updates graph based on quiz performance"""
    responses = st.session_state.quiz_responses.get(sid, {}).get(topic, [])

    # Calculate subtopic weaknesses
    subtopic_scores = Counter()
    for response in responses:
        question = next(q for q in QUESTION_BANK[topic] if q['id'] == response['qid'])
        subtopic = question.get('subtopic', 'General')
        subtopic_scores[subtopic] += response['is_correct']

    # Update graph weights
    for subtopic, score in subtopic_scores.items():
        if G.has_edge(topic, subtopic):
            G[topic][subtopic]['weight'] = max(1, score * 0.5)  # Adjust edge weight
# ==================================================================
# 4. Collaborative Filtering & Peer Tutoring
# ==================================================================
def collaborative_filtering_recommendations(sid, df, seg):
    """Use collaborative filtering to recommend questions based on similar students"""
    # Skip if not enough data
    if df.empty or len(df.StudentID.unique()) < 5:
        return []

    # Get current student's data
    student_data = df[df.StudentID == sid]
    if student_data.empty:
        return []

    # Get student's performance profile
    student_profile = student_data.groupby('Topic').Correct.mean()

    # Find similar students (same segment but not the same student)
    if seg.empty:
        return []

    student_seg = seg[seg.StudentID == sid]
    if student_seg.empty:
        return []

    label = student_seg.iloc[0]['label']
    similar_students = seg[(seg.label == label) & (seg.StudentID != sid)].StudentID.tolist()

    if not similar_students:
        return []

    # Find topics that similar students do well in but current student struggles with
    similar_data = df[df.StudentID.isin(similar_students)]
    similar_perf = similar_data.groupby('Topic').Correct.mean()

    # Find topics where similar students do well (>70%) but current student does poorly (<50%)
    potential_topics = []
    for topic in similar_perf.index:
        if topic in student_profile and similar_perf[topic] > 0.7 and student_profile[topic] < 0.5:
            potential_topics.append(topic)

    # Get specific questions from those topics
    recommendations = []
    for topic in potential_topics[:2]:  # Limit to 2 topics
        if topic in QUESTION_BANK:
            # Find an appropriate question (medium difficulty)
            medium_questions = [q for q in QUESTION_BANK[topic] if q['difficulty'] == 2]
            if medium_questions:
                q = np.random.choice(medium_questions)
                recommendations.append(f"👥 Similar students do well with: {topic} - {q['text']}")

    return recommendations
def suggest_peer_tutoring(sid, df, seg):
    """Match students who can help each other based on complementary strengths/weaknesses"""
    if df.empty or seg.empty:
        return []

    # Get current student's strengths and weaknesses
    student_data = df[df.StudentID == sid]
    if student_data.empty:
        return []

    topic_perf = student_data.groupby('Topic').Correct.mean()

    # Define strengths and weaknesses
    strengths = topic_perf[topic_perf >= 0.7].index.tolist()
    weaknesses = topic_perf[topic_perf < 0.5].index.tolist()

    if not strengths or not weaknesses:
        return []

    # Find other students
    other_students = df[df.StudentID != sid].StudentID.unique()

    potential_matches = []
    for other_id in other_students:
        other_data = df[df.StudentID == other_id]
        if other_data.empty:
            continue

        other_perf = other_data.groupby('Topic').Correct.mean()

        # Find their strengths and weaknesses
        other_strengths = other_perf[other_perf >= 0.7].index.tolist()
        other_weaknesses = other_perf[other_perf < 0.5].index.tolist()

        # Check if they're a good match (they're strong where you're weak and vice versa)
        common_strengths = set(strengths).intersection(set(other_weaknesses))
        common_weaknesses = set(weaknesses).intersection(set(other_strengths))

        if common_strengths and common_weaknesses:
            # Calculate match score based on number of complementary topics
            match_score = len(common_strengths) + len(common_weaknesses)
            potential_matches.append((other_id, match_score, list(common_strengths), list(common_weaknesses)))

    # Sort by match score and get top match
    if potential_matches:
        potential_matches.sort(key=lambda x: x[1], reverse=True)
        best_match = potential_matches[0]

        return [
            f"👥 Student #{best_match[0]} would be a great study partner!",
            f"📚 They can help you with: {', '.join(best_match[3][:2])}",
            f"🔄 You can help them with: {', '.join(best_match[2][:2])}"
        ]

    return []

# ==================================================================
# 5. Quiz Helpers
# ==================================================================
def get_quiz_recommendations(sid, completed):
    rec = []

    # Only recommend topics the student has completed
    for t in completed:
        if t in FORMULA_QUIZ_BANK:
            # Get current progress or default to 0
            p = st.session_state.quiz_progress.setdefault(sid, {}).get(t, 0)
            subs = list(FORMULA_QUIZ_BANK[t].keys())

            # Recommend next subtopic if available
            if p < len(subs):
                rec.append(f"📝 Quiz Alert: {subs[p]} ({t})")

    return rec

# ==================================================================
# 6. Comprehensive Recommendations
# ==================================================================
def get_recommendations(sid, df, G, seg, mot='High'):
    # Handle empty dataframe
    if df.empty:
        return ["No student data available for recommendations."]

    # Get student data
    sd = df[df.StudentID == sid]
    if sd.empty:
        return [f"No data found for student {sid}."]

    # Calculate accuracy by topic
    acc = sd.groupby('Topic').Correct.mean()

    # Get completed topics
    comp = sd[sd.Completed].Topic.unique().tolist()

    rec = []

    # 1) Add a motivation quote
    if comp:
        selected_topic = np.random.choice(comp)
        if selected_topic in MOTIVATION_QUOTES:
            rec.append(f"💭 \"{MOTIVATION_QUOTES[selected_topic]}\"")

    # 2) Bridge course recommendations for low‐accuracy topics
    low_topics = acc[acc < 0.3].index.tolist()
    for t in low_topics:
        if t in BRIDGE_COURSES:
            rec.append(f"🚧 Bridge: {t} - {BRIDGE_COURSES[t]}")
        else:
            rec.append(f"🚧 Bridge: {t}")

    # 3) HOTS for high‐accuracy topics (unless motivation is Low)
    if mot != 'Low':
        high_topics = acc[acc > 0.7].index.tolist()
        for t in high_topics[:2]:
            if t in HOTS_QUESTIONS:
                rec.append(f"🧠 HOTS {t}: {', '.join(HOTS_QUESTIONS[t][:2])}")

    # 4) Practice recommendations for up to 3 completed topics
    for t in comp[:3]:
        if t in PRACTICE_QUESTIONS:
            pq = PRACTICE_QUESTIONS[t]
            seq = pq.get('recent', []) + pq.get('historical', []) + pq.get('fundamental', [])
            rec.append(f"📚 Practice {t}: {', '.join(seq[:3])}")

    # 5) Quiz recommendations
    quiz_recs = get_quiz_recommendations(sid, comp)
    rec.extend(quiz_recs[:2])

    # 6) Media & analogies
    hard_topics = acc[acc < 0.5].index.tolist()
    # first, videos for up to 2 completed topics
    for t in comp[:2]:
        m = MEDIA_LINKS.get(t, {})
        if m.get('videos'):
            rec.append(f"🎥 Media: {', '.join(m['videos'][:1])}")
            # then, analogies for all “hard” topics
    for t in hard_topics:
        m = MEDIA_LINKS.get(t, {})
        if m.get('analogies'):
            rec.append(f"🔗 Analogy: {m['analogies']}")

    # 7) Formula revision materials for low‐performing topics
    for t in low_topics[:2]:
        if t in FORMULA_REVISION:
            rec.append(f"📊 Formula Help {t}: {', '.join(FORMULA_REVISION[t][:2])}")

    # 8) Easy‐win topics if motivation is Low
    if mot == 'Low' and comp:
        easy_topic = np.random.choice(comp)
        if easy_topic in EASY_TOPICS:
            rec.append(f"👍 Easy Win: {easy_topic} - {EASY_TOPICS[easy_topic][0]}")

    # 9) Subtopic & application recommendations per completed topic
    for t in comp:
        # a) subtopics that need focus (only for low topics)
        if t in low_topics:
            subs = [
                v for _, v, d in G.out_edges(t, data=True)
                if d.get('relation') == 'subtopic'
            ]
            if subs:
                rec.append(f"🔧 Subtopics to review in {t}: {', '.join(subs[:3])}")

        # b) collaborative filtering suggestions
        collab_recs = collaborative_filtering_recommendations(sid, df, seg)
        rec.extend(collab_recs)

        # c) “future” topics (prereqs or odds_ratio)
        future_topics = [
            v for _, v, d in G.out_edges(t, data=True)
            if d.get('relation') in ('prereq', 'odds_ratio')
        ]
        if future_topics:
            rec.append(f"🔄 Apply {t} in: {', '.join(future_topics[:2])}")

        # d) direct applications
        apps = [
            v for _, v, d in G.out_edges(t, data=True)
            if d.get('relation') in ('app_preparation', 'application')
        ]
        if apps:
            rec.append(f"🔬 Real applications of {t}: {', '.join(apps[:2])}")
    return rec

# ==================================================================
# 7. Question‑Level Analytics
# ==================================================================
def analyze_item_level_performance(sid):
    if sid not in st.session_state.question_responses:
        return []

    responses = st.session_state.question_responses[sid]
    problem_questions = []

    for q_id, attempts in responses.items():
        if len(attempts) >= 2:
            success_rate = sum(attempts) / len(attempts)
            if success_rate < 0.5:
                # Find question details
                for topic, questions in QUESTION_BANK.items():
                    for q in questions:
                        if q['id'] == q_id:
                            # Add solution steps to the result
                            problem_questions.append((
                                q_id,
                                q['text'],
                                topic,
                                success_rate,
                                q.get('solution_steps', 'No feedback available')  # This line added
                            ))
                            break
                    if q_id in [q['id'] for q in questions]:  # More efficient check
                        break

    # Sort by worst performance first
    return sorted(problem_questions, key=lambda x: x[3])  # x[3] is success_rate


# ==================================================================
# 8. Streamlit UI
# ==================================================================
def main():
    st.set_page_config(layout="wide", page_title="Learning Dashboard", page_icon="🧠")

    # Add CSS for better styling
    st.markdown(
        """
        <style>
        .big-font {font-size:24px !important; font-weight:bold;}
        .medium-font {font-size:18px !important;}
        .highlight {background-color:#f0f2f6; padding:10px; border-radius:5px;}
        .recommendation {margin-bottom:10px; padding:5px;}
        .graph-container {border:1px solid #ddd; border-radius:5px; padding:10px;}
        </style>
        """, unsafe_allow_html=True)

    if 'quiz_progress' not in st.session_state:
        st.session_state.quiz_progress = {}
    if 'question_responses' not in st.session_state:
        st.session_state.question_responses = {}

    st.markdown('<p class="big-font">Enhanced Learning Dashboard 🧠</p>', unsafe_allow_html=True)

    # Initialize critical variables
    sid = None
    df = pd.DataFrame()
    G = nx.DiGraph()
    seg = pd.DataFrame()

    try:
        with st.spinner("Loading student data..."):
            df = generate_student_data()
            if df.empty:
                st.error("No student data generated")
                st.stop()

        # Build knowledge graph
        with st.spinner("Building knowledge graph..."):
            G = build_knowledge_graph(PREREQUISITES, df, TOPICS)
            G = apply_dual_tier_scoring(G)

        # Segment students
        with st.spinner("Analyzing student segments..."):
            seg = segment_students(df)
            if seg.empty:
                st.error("Student segmentation failed")
                st.stop()

        # Sidebar settings
        st.sidebar.markdown('# Settings', unsafe_allow_html=True)

        # Student selection
        all_students = sorted(df.StudentID.unique())
        hardwired = [0, 1, 2]
        other_students = [s for s in all_students if s not in hardwired]
        combined_students = hardwired + other_students

        options = []
        for s in combined_students:
            if s == 0:
                options.append(f"Student {s} (Demo Topper)")
            elif s == 1:
                options.append(f"Student {s} (Demo Average)")
            elif s == 2:
                options.append(f"Student {s} (Demo Poor)")
            else:
                options.append(f"Student {s}")

        def format_student(student_id):
            idx = combined_students.index(student_id)
            return options[idx]

        sid = st.sidebar.selectbox("Select Student", combined_students, format_func=format_student)

        # Performance tier display
        student_info = seg[seg.StudentID == sid]
        perf_tier = student_info['label'].iloc[0] if not student_info.empty else 'Average'
        tier_colors = {'Topper': 'green', 'Average': 'blue', 'Poor': 'red'}
        st.sidebar.markdown(
            f"### Student Details\n"
            f"**Performance Tier:** <span style='color:{tier_colors.get(perf_tier, 'black')}'>{perf_tier}</span>",
            unsafe_allow_html=True
        )

        # ── Motivation override ──
        st.sidebar.subheader("Student Mood Tracker")
        motivation_mapping = {
            'Topper': 'High',
            'Average': 'Medium',
            'Poor': 'Low'
        }
        current_mot = motivation_mapping.get(perf_tier, 'Medium')
        mood_options = ['High', 'Medium', 'Low']
        default_idx = mood_options.index(current_mot)
        override_mot = st.sidebar.selectbox(
            "Override Motivation Level",
            mood_options,
            index=default_idx,
            key="override_mot"
        )
        df.loc[df.StudentID == sid, 'MotivationLevel'] = override_mot

        # ── Peer Tutoring Section ──
        with st.expander("🔗 Peer Tutoring Matches", expanded=False):
            st.write("Students who complement your strengths/weaknesses:")
            matches = suggest_peer_tutoring(sid, df, seg)
            if matches:
                for line in matches:
                    st.markdown(f"- {line}")
            else:
                st.info("No suitable peer matches found right now.")

        # Main content layout
        col1, col2 = st.columns([1, 2])

        # Knowledge Graph Visualization
        with col1:
            st.markdown('<p class="medium-font">Knowledge Graph</p>', unsafe_allow_html=True)
            with st.container(height=400, border=True):
                try:
                    pos = nx.spring_layout(G, seed=42)
                    edge_colors = {
                        'prereq': '#FF0000', 'odds': '#00FF00',
                        'shap_importance': '#0000FF', 'application': '#800080',
                        'subtopic': '#FFA500', 'sub_prereq': '#FF69B4',
                        'app_preparation': '#008080'
                    }
                    traces = []
                    for rel, col in edge_colors.items():
                        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == rel]
                        if edges:
                            xs, ys = [], []
                            for u, v in edges:
                                if u in pos and v in pos:
                                    xs += [pos[u][0], pos[v][0], None]
                                    ys += [pos[u][1], pos[v][1], None]
                            traces.append(
                                go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=col, width=2), name=rel)
                            )
                    node_x = [pos[n][0] for n in G.nodes if n in pos]
                    node_y = [pos[n][1] for n in G.nodes if n in pos]
                    node_text = [n for n in G.nodes if n in pos]
                    acc_by_topic = df[df.StudentID == sid].groupby('Topic').Correct.mean()
                    node_colors = []
                    for node in node_text:
                        perf = acc_by_topic.get(node, np.nan)
                        if np.isnan(perf):
                            node_colors.append('#FFA500')
                        elif perf < 0.3:
                            node_colors.append('#FF0000')
                        elif perf < 0.7:
                            node_colors.append('#FFFF00')
                        else:
                            node_colors.append('#00FF00')
                    node_trace = go.Scatter(
                        x=node_x, y=node_y, mode='markers+text', text=node_text,
                        marker=dict(size=15, color=node_colors, line=dict(width=1, color='#000000')),
                        textposition="top center", name='Topics'
                    )
                    fig = go.Figure(data=traces + [node_trace], layout=go.Layout(
                        showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=350
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Topic colors: 🔴 Needs work | 🟡 Average | 🟢 Strong | 🟠 No data")
                except Exception as e:
                    st.error(f"Graph display error: {str(e)}")

        # Recommendations Panel
        with col2:
            st.markdown('<p class="medium-font">Personalized Learning Recommendations</p>', unsafe_allow_html=True)
            if sid is not None:
                try:
                    recommendations = get_recommendations(sid, df, G, seg, override_mot)
                    rec_types = {
                        "🚧 Bridge": [], "🧠 HOTS": [], "📚 Practice": [],
                        "📝 Quiz": [],  "🎥 Media": [],  "🔗 Analogy": [],
                        "📊 Formula": [], "🔧 Subtopics": [], "🔄 Apply": [],
                        "👍 Easy Win": [], "💭 Quote": []
                    }
                    for r in recommendations:
                        for prefix in rec_types:
                            if r.startswith(prefix):
                                rec_types[prefix].append(r)
                                break
                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown("### Study Plan")
                        for item in rec_types["🚧 Bridge"] + rec_types["📚 Practice"] + rec_types["📊 Formula"]:
                            st.info(item)
                    with cols[1]:
                        st.markdown("### Challenges")
                        for item in rec_types["🧠 HOTS"] + rec_types["📝 Quiz"] + rec_types["🔧 Subtopics"]:
                            st.success(item)
                    with cols[2]:
                        st.markdown("### Engagement")
                        for item in rec_types["💭 Quote"]:
                            st.markdown(f'<div class="highlight">{item}</div>', unsafe_allow_html=True)
                        for item in rec_types["🎥 Media"] + rec_types["🔗 Analogy"] + rec_types["🔄 Apply"] + rec_types["👍 Easy Win"]:
                            st.warning(item)
                except Exception as e:
                    st.error(f"Recommendation error: {str(e)}")

        # Strategic Peer Comparison
        with st.expander("🔍 Strategic Peer Comparison", expanded=True):
            if not df.empty and sid is not None:
                try:
                    current_data = df[df.StudentID == sid]
                    current_start = current_data.ExamDate.min()
                    comparable_peers = df[
                        (df.ExamDate > current_start - pd.Timedelta(days=60)) &
                        (df.ExamDate < current_start + pd.Timedelta(days=60)) &
                        (df.StudentID != sid)
                    ].StudentID.unique()
                    if len(comparable_peers) > 0:
                        peer_perf = df[df.StudentID.isin(comparable_peers)].groupby('StudentID').Correct.mean()
                        current_perf = current_data.Correct.mean()
                        peer_diff = abs(peer_perf - current_perf)
                        if not peer_diff.empty:
                            best_peer = peer_diff.idxmax()
                            perf_gap = peer_diff.max()
                            if perf_gap >= 0.15:
                                st.markdown(f"#### 🎯 Comparison with Student {best_peer}")
                                st.caption(f"Similar start date, {perf_gap:.0%} performance difference")
                                insights = progression_summary(df, sid, best_peer)
                                col1, col2 = st.columns([2, 3])
                                with col1:
                                    st.markdown("##### 📈 Key Behavioral Differences")
                                    if len(insights) > 1:
                                        for insight in insights[1:4]:
                                            st.markdown(f"<div class='highlight'>{insight}</div>", unsafe_allow_html=True)
                                    else:
                                        st.info("No significant behavioral differences found")
                                with col2:
                                    st.markdown("##### 🚀 Improvement Plan")
                                    if len(insights) >= 4:
                                        top_metric = insights[1].split(':')[0]
                                        s1_val = int(insights[1].split(' ')[-3])
                                        s2_val = int(insights[1].split(' ')[-6])
                                        fig = go.Figure()
                                        fig.add_trace(go.Bar(
                                            x=['You', f'Student {best_peer}'],
                                            y=[s1_val, s2_val],
                                            text=[s1_val, s2_val],
                                            textposition='auto'
                                        ))
                                        fig.update_layout(title=f"{top_metric} Comparison", height=300)
                                        st.plotly_chart(fig, use_container_width=True)
                                        st.markdown(f"<div class='highlight'>{insights[-1]}</div>", unsafe_allow_html=True)
                                    else:
                                        st.warning("Insufficient data for detailed comparison")
                            else:
                                st.info("No peers found with >=15% performance gap")
                        else:
                            st.warning("Could not calculate peer differences")
                    else:
                        st.warning("⚠️ No comparable peers found with similar start dates")
                except Exception as e:
                    st.error(f"Comparison failed: {str(e)}")
            else:
                st.warning("Select a student to enable peer comparison")

        # Quiz Section
        st.markdown('<p class="medium-font">Interactive Quiz Section</p>', unsafe_allow_html=True)
        try:
            comp = df[(df.StudentID == sid) & (df.Completed)].Topic.unique().tolist()
            quiz_topics = [t for t in comp if t in FORMULA_QUIZ_BANK]
            if quiz_topics:
                c1, c2 = st.columns([1, 2])
                with c1:
                    selected_topic = st.selectbox("Select Quiz Topic", quiz_topics)
                    if st.button("Start Quiz", type="primary"):
                        st.session_state.show_quiz = True
                        st.session_state.quiz_topic = selected_topic
                        st.session_state.quiz_answers = {}
                with c2:
                    if st.session_state.get('show_quiz'):
                        topic = st.session_state.quiz_topic
                        st.markdown(f"### {topic} Quiz")
                        with st.form(key=f"quiz_form_{topic}"):
                            for i, (sub, ql) in enumerate(FORMULA_QUIZ_BANK[topic].items()):
                                st.subheader(f"Section: {sub}")
                                for j, q in enumerate(ql):
                                    q_key = f"q_{topic}_{i}_{j}"
                                    st.markdown(f"**Q{j+1}:** {q['question']}")
                                    if q['type'] == 'formula':
                                        st.text_input("Your answer:", key=q_key)
                                    else:
                                        st.radio("Select:", ["Option A", "Option B", "Option C"], key=q_key)
                                    st.markdown("---")
                            if st.form_submit_button("Submit Quiz"):
                                try:
                                    # Get full question objects from bank
                                    topic_questions = FORMULA_QUIZ_BANK[topic]
                                    all_questions = [q for sublist in topic_questions.values() for q in sublist]

                                    # Initialize tracking structures
                                    st.session_state.setdefault('quiz_responses', {}).setdefault(sid, {})
                                    st.session_state.setdefault('question_responses', {}).setdefault(sid, {})

                                    # Process each question
                                    for i, question in enumerate(all_questions):
                                        qid = question['id']
                                        student_answer = st.session_state[f"q_{topic}_{i}"]

                                        # Validate answer
                                        is_correct, feedback = validate_answer(question, student_answer)

                                        # Update progress tracking
                                        st.session_state.question_responses[sid].setdefault(qid, []).append(
                                            int(is_correct))

                                        # Store full response details
                                        st.session_state.quiz_responses[sid].setdefault(topic, []).append({
                                            "qid": qid,
                                            "answer": student_answer,
                                            "is_correct": is_correct,
                                            "timestamp": datetime.now(),
                                            "feedback": feedback
                                        })

                                    # Update knowledge graph
                                    update_knowledge_graph_with_quiz(G, sid, topic)

                                    st.success("Quiz submitted successfully! Recommendations will update accordingly.")

                                except Exception as e:
                                    st.error(f"Quiz error: {str(e)}")
        except Exception as e:
            st.error(f"Quiz section error: {str(e)}")

        # Performance Analytics
        st.markdown('<p class="medium-font">Performance Analytics</p>', unsafe_allow_html=True)
        try:
            if sid is not None:
                student_data = df[df.StudentID == sid]
                if not student_data.empty:
                    tab1, tab2, tab3 = st.tabs(["Topic Performance", "Time Analysis", "Usage Patterns"])
                    with tab1:
                        topic_perf = student_data.groupby('Topic').agg(
                            Accuracy=('Correct', 'mean'),
                            Attempts=('Correct', 'count')
                        ).reset_index().sort_values('Accuracy', ascending=False)

                        if not topic_perf.empty:
                            fig = go.Figure(go.Bar(
                                x=topic_perf['Topic'], y=topic_perf['Accuracy'],
                                text=[f"{acc:.1%}" for acc in topic_perf['Accuracy']],
                                marker=dict(color='green')
                            ))
                            fig.update_layout(
                                title='Topic Performance',
                                yaxis=dict(title='Accuracy', tickformat='.0%', range=[0, 1]),
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        time_data = student_data.groupby('Topic').agg(
                            AvgTime=('TimeTaken', 'mean'),
                            Accuracy=('Correct', 'mean')
                        ).reset_index()

                        if not time_data.empty:
                            fig = go.Figure(go.Scatter(
                                x=time_data['AvgTime'], y=time_data['Accuracy'],
                                mode='markers+text', text=time_data['Topic'],
                                marker=dict(size=15, color=time_data['Accuracy'], colorscale='RdYlGn')
                            ))
                            fig.update_layout(
                                title='Time vs. Accuracy by Topic',
                                xaxis=dict(title='Average Time Taken'),
                                yaxis=dict(title='Accuracy', tickformat='.0%'),
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with tab3:
                        usage_data = student_data.groupby('Topic').agg({
                            'VideosWatched': 'sum', 'QuizzesTaken': 'sum',
                            'PracticeSessions': 'sum', 'MediaClicks': 'sum'
                        }).reset_index()

                        if not usage_data.empty:
                            fig = go.Figure()
                            # Map the correct column names to their display names
                            columns_map = {
                                'VideosWatched': 'Videos',
                                'QuizzesTaken': 'Quizzes',
                                'PracticeSessions': 'Practice',
                                'MediaClicks': 'Media'
                            }
                            colors = {'Videos': '#1f77b4', 'Quizzes': '#ff7f0e',
                                      'Practice': '#2ca02c', 'Media': '#d62728'}

                            for db_col, display_name in columns_map.items():
                                fig.add_trace(go.Bar(
                                    x=usage_data['Topic'],
                                    y=usage_data[db_col],
                                    name=display_name,
                                    marker_color=colors[display_name]
                                ))
                            fig.update_layout(barmode='stack', height=400)
                            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Analytics error: {str(e)}")

        # Question-Level Analytics
        st.markdown('<p class="medium-font">Question-Level Analytics</p>', unsafe_allow_html=True)
        try:
            if sid is not None:
                problem_questions = analyze_item_level_performance(sid)
                if problem_questions:
                    st.warning("🔍 Questions you're struggling with:")
                    for q_id, q_text, topic, success_rate, solution_steps in problem_questions:
                        with st.expander(f"{topic}: {q_text} (Success: {success_rate:.0%})"):
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.write("**Step-by-Step Solution:**")
                                if isinstance(solution_steps, list):
                                    for step in solution_steps:
                                        st.write(f"→ {step}")
                                else:
                                    st.write(solution_steps)

                                st.write("**General Study Tips:**")
                                if topic == "Algebra":
                                    st.write(
                                        "- Break problem into steps\n- Check variable isolation\n- Substitute values")
                                elif topic == "Geometry":
                                    st.write("- Draw diagrams\n- Identify key formulas\n- Look for patterns")
                                elif topic == "Calculus":
                                    st.write("- Review rules\n- Simplify functions\n- Check algebra")
                                else:
                                    st.write("- Review concepts\n- Create visuals\n- Practice basics")

                            with col2:
                                if st.button(f"🧩 Similar Question\nfor {q_id}",
                                             help="Generate practice question on this topic"):
                                    st.session_state.show_practice = True
                                    st.session_state.practice_topic = topic
                                    st.session_state.practice_id = q_id
        except Exception as e:
            st.error(f"Question analysis error: {str(e)}")

    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.stop()
if __name__ == "__main__":
    main()

