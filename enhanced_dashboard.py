# enhanced_dashboard_complete.py
import re
import math
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
# ==================================================================
# 0. Configuration & Mock Data
# ==================================================================
PEER_TUTORS = {
    "Algebra":   ["S101","S102","S105"],
    "Geometry":  ["S103","S104","S106"],
    "Calculus":  ["S107","S108","S109"],
    "Chemistry": ["S110","S111","S112"],
    "Biology":   ["S113","S114","S115"],
    "Derivatives":["S116","S117"],
    "Kinematics":["S118","S119"],
    "Gas Laws":  ["S120","S121"],
}
PREREQUISITES = {
    'Geometry':      ['Algebra'],
    'Calculus':      ['Algebra','Geometry'],
    'Derivatives':   ['Calculus'],           # derivative builds on calculus
    'Chemistry':     ['Algebra'],
    'Gas Laws':      ['Chemistry'],
    'Biology':       ['Chemistry'],
    'Kinematics':    ['Algebra'],
}
TOPICS = [
    'Algebra',
    'Geometry',
    'Calculus',
    'Derivatives',
    'Chemistry',
    'Gas Laws',
    'Biology',
    'Kinematics',
]

SUBTOPICS = {
    'Algebra':    ['Equations','Inequalities','Polynomials'],
    'Geometry':   ['Angles','Shapes','Trigonometry'],
    'Calculus':   ['Limits','Integrals'],
    'Derivatives':['Power Rule','Chain Rule','Product Rule'],
    'Chemistry':  ['Elements','Reactions','Stoichiometry'],
    'Gas Laws':   ['Boyle','Charles','Ideal Gas'],
    'Biology':    ['Cells','Genetics','Ecology'],
    'Kinematics': ['Velocity','Acceleration','Projectile Motion'],
}

# Application-level edges
# Only one or two “application” edges to keep examples
APPLICATION_RELATIONS = {
    'Optimization':      {'base_topic':'Derivatives',  'type':'application',
                           'description':'Find maxima/minima in real problems'},
    'Reaction Rates':    {'base_topic':'Chemistry',    'type':'application',
                           'description':'Modeling how fast reactions proceed'},
    'Projectile Motion': {'base_topic':'Kinematics',   'type':'application',
                           'description':'Parabolic trajectories in physics'},
}
# (Optional) A minimal quiz bank just to keep the pipeline alive
FORMULA_QUIZ_BANK = {
    'Algebra': {
        'Quadratic Equations': [
            {
                "id": "alg_q1",
                "question": "x²−5x+6=0 → x?",
                "type": "formula",
                "answer": "2 or 3",
                "solution_steps": ["Factor: (x-2)(x-3)=0", "Solutions: x=2, x=3"]
            }
        ]
    },
    'Calculus': {
        'Derivative Rules': [
            {
                "id": "calc_q1",
                "question": "d/dx(x³)=?",
                "type": "formula",
                "answer": "3x²",
                "solution_steps": ["Apply the power rule: d/dx(xⁿ) = nxⁿ⁻¹ where n=3."]
            }
        ]
    },
    'Chemistry': {
        'Gas Laws': [
            {
                "id": "chem_q1",
                "question": "PV=nRT → solve for T",
                "type": "formula",
                "answer": "T=PV/(nR)",
                "solution_steps": ["Divide both sides of the equation by 'nR' to isolate 'T'."]
            }
        ]
    }
}

if "quiz_progress" not in st.session_state:
    st.session_state.quiz_progress = {}
QUESTION_BANK = {
    'Algebra': [
        {"id":"alg_q1",
         "text":"x²−5x+6=0 → x?",
         "difficulty":1,
         "type":"formula",
         "answer":"2 or 3",
         "solution_steps":["Factor: (x-2)(x-3)=0", "Solutions: x=2, x=3"]}
    ],
    'Calculus': [
        {"id":"calc_q1",
         "text":"d/dx(x³)=?",
         "difficulty":1,
         "type":"formula",
         "answer":"3x²",
         "solution_steps":["Apply power rule"]}
    ],
    'Chemistry': [
        {"id":"chem_q1",
         "text":"PV=nRT → solve for T",
         "difficulty":1,
         "type":"formula",
         "answer":"T=PV/(nR)",
         "solution_steps":["Divide both sides by nR"]}
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
        'fundamental': ['Number Sense', 'Order of Operations'],
        'mid_level': ['Rational Expressions', 'Systems of Equations']
    },
    'Geometry': {
        'recent': ['Triangle Congruence', 'Circle Theorems'],
        'historical': ['Angle Relationships', 'Pythagorean Theorem'],
        'fundamental': ['Basic Shapes', 'Area Formulas'],
        'mid_level': ['Coordinate Geometry', 'Mensuration (2D)']
    },
    'Calculus': {
        'recent': ['Derivative Rules', 'Integration Techniques'],
        'historical': ['Limits', 'Continuity'],
        'fundamental': ['Function Behavior', 'Graphing'],
        'mid_level': ['Applications of Derivatives', 'Basic Integration']
    },
    'Chemistry': {
        'recent': ['Balancing Equations', 'Stoichiometry'],
        'historical': ['Periodic Table', 'Chemical Bonds'],
        'fundamental': ['States of Matter', 'Element Properties'],
        'mid_level': ['Solutions', 'Electrochemistry']
    },
    'Biology': {
        'recent': ['Cell Division', 'Heredity'],
        'historical': ['Cell Structure', 'Classification'],
        'fundamental': ['Life Processes', 'Scientific Method'],
        'mid_level': ['Genetics', 'Ecology and Environment']
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
    study_profiles = ['video_heavy', 'practice_heavy', 'quiz_heavy', 'balanced']
    profile_dist = [0.3, 0.25, 0.2, 0.25]

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
            print(f"Subtopic sampling error: {str(e)}")
            return [None] * 3

    for sid in range(num_students):
        try:
            # Profile selection
            study_profile = np.random.choice(study_profiles, p=profile_dist)

            # Base characteristics
            exam_date = datetime.now() - timedelta(days=np.random.randint(1, 60))
            base_skill = np.random.beta(2, 1.5)
            motivation = np.random.choice(['High', 'Medium', 'Low'], p=[0.4, 0.4, 0.2])
            mf = {'High': 1.4, 'Medium': 1.0, 'Low': 0.6}.get(motivation, 1.0)  # Safe lookup

            # Profile-driven behavior
            if study_profile == 'video_heavy':
                vids = max(0, np.random.poisson(4))
                prac = max(0, np.random.poisson(1))
                qz = max(0, np.random.poisson(1))
            elif study_profile == 'practice_heavy':
                vids = max(0, np.random.poisson(1))
                prac = max(0, np.random.poisson(5))
                qz = max(0, np.random.poisson(2))
            elif study_profile == 'quiz_heavy':
                vids = max(0, np.random.poisson(2))
                prac = max(0, np.random.poisson(2))
                qz = max(0, np.random.poisson(4))
            else:  # balanced
                vids = max(0, np.random.poisson(2))
                prac = max(0, np.random.poisson(3))
                qz = max(0, np.random.poisson(2))

            # Safe topic completion handling
            done_topics = []
            if TOPICS:
                try:
                    done_topics = np.random.choice(
                        TOPICS,
                        size=np.random.randint(0, len(TOPICS) + 1),
                        replace=False
                    ).tolist()
                except ValueError as e:
                    print(f"Topic selection error: {str(e)}")
                    done_topics = []

            # Per-topic interactions
            for topic in TOPICS:
                # Safe interaction count calculation
                interaction_count = min(max(np.random.poisson(3) + 1, 1), 10)
                done = topic in done_topics or np.random.rand() < 0.3  # 30% completion probability

                for _ in range(interaction_count):
                    # Subtopic handling
                    subtopics = SUBTOPICS.get(topic, [])
                    subs = get_subtopics(subtopics)

                    # Success probability with model relationship between behaviors and outcomes
                    success_prob = np.clip(
                        base_skill * mf * (1 + 0.1 * vids + 0.15 * prac + 0.07 * qz),
                        0.1, 0.95
                    )

                    # Quiz progress validation
                    quiz_bank = FORMULA_QUIZ_BANK.get(topic, {})
                    qp_max = max(1, len(quiz_bank))
                    qp = np.random.randint(0, qp_max) if qp_max > 0 else 0

                    # Dirichlet distribution for weights with safety
                    try:
                        wts = np.random.dirichlet(np.ones(3))
                        wts /= wts.sum()  # Ensure sum to 1
                    except Exception as e:
                        wts = [0.4, 0.3, 0.3]
                        print(f"Weight generation error: {str(e)}")

                    # Time modeling with skill and motivation factors
                    time_factor = 2 - mf  # Inverse relationship: higher motivation -> less time
                    tkt = max(0.1, np.random.gamma(2, 1.5) * time_factor)

                    rows.append({
                        'StudentID': sid,
                        'Topic': topic,
                        'Subtopic1': subs[0],
                        'Subtopic2': subs[1],
                        'Subtopic3': subs[2],
                        'Weight1': wts[0],
                        'Weight2': wts[1],
                        'Weight3': wts[2],
                        'Correct': np.random.binomial(1, success_prob),
                        'TimeTaken': tkt,
                        'ExamDate': exam_date,
                        'Completed': done,
                        'MotivationLevel': motivation,
                        'VideosWatched': vids,
                        'PracticeSessions': prac,
                        'QuizzesTaken': qz,
                        'MediaClicks': max(0, np.random.poisson(2)),
                        'QuizProgress': qp,
                        'StudyProfile': study_profile,
                        'ConsistencyScore': np.random.beta(3, 1)
                    })
        except Exception as e:
            print(f"Error generating data for student {sid}: {str(e)}")
            continue

    return pd.DataFrame(rows)


def validate_answer(question, student_answer):
    """Enhanced validation with normalization and flexible matching"""
    try:
        # Normalization function for answer comparison
        def normalize_answer(answer):
            """Process answer for consistent comparison"""
            if not isinstance(answer, str):
                answer = str(answer)

            # Standard normalization steps
            normalized = answer.lower().strip()
            normalized = re.sub(r'\s+', ' ', normalized)  # Collapse whitespace
            normalized = re.sub(r'[;,]\s*', ',', normalized)  # Standardize separators
            normalized = re.sub(r'[^a-z0-9.,=+-]', '', normalized)  # Remove special chars

            # Handle mathematical equivalences
            normalized = normalized.replace('^', '')  # x² vs x2
            normalized = normalized.replace('\\frac', '/')  # LaTeX fractions
            normalized = re.sub(r'\.0+$', '', normalized)  # 2.0 → 2

            return normalized

        # Get question type and parameters
        q_type = question.get('type', 'unknown')
        correct_answer = str(question.get('answer', '')).strip()
        solution = question.get('solution_steps', 'No solution available')

        # Normalize both answers
        norm_correct = normalize_answer(correct_answer)
        norm_student = normalize_answer(student_answer)

        if q_type == 'free_response':
            # Handle multiple valid answer formats
            correct_parts = sorted(re.split(r'[,/]', norm_correct))
            student_parts = sorted(re.split(r'[,/]', norm_student))

            # Check for complete match (order-independent)
            if set(correct_parts) == set(student_parts):
                return True, f"Solution: {solution}"

            # Check numerical equivalence
            try:
                correct_num = float(norm_correct)
                student_num = float(norm_student)
                if math.isclose(correct_num, student_num, rel_tol=0.01):
                    return True, f"Solution: {solution}"
            except (ValueError, TypeError):
                pass

            # Partial credit for multi-part answers
            common = set(correct_parts) & set(student_parts)
            if common:
                partial = len(common) / len(correct_parts)
                return (False,
                        f"Partial credit ({partial:.0%}): "
                        f"Missing {set(correct_parts) - set(student_parts)}")

            return False, f"Solution: {solution}"

        elif q_type == 'multiple_choice':
            options = question.get('options', [])
            correct_index = question.get('correct_option', -1)

            if correct_index < 0 or correct_index >= len(options):
                return False, "Invalid question configuration"

            # Normalize both the option and student answer
            normalized_options = [normalize_answer(opt) for opt in options]
            norm_student_choice = normalize_answer(student_answer)

            # Match either index or normalized text
            if (student_answer == correct_index) or \
                    (norm_student_choice == normalize_answer(options[correct_index])):
                return True, f"Correct answer: {options[correct_index]}"

            return False, f"Correct answer: {options[correct_index]}"

        else:
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

        # Get toppers' strong topics
        # ------------------------------
        # Modified section
    toppers = seg[seg.label == 'Topper'].StudentID.tolist()

    # Get most frequent topic for each topper
    strong_topics = (
        df[df.StudentID.isin(toppers)]
        .groupby('StudentID')
        .apply(lambda x: x.Topic.mode()[0])  # Get most frequent topic
        .value_counts()  # Count how many toppers have each topic as strong
    )

    if not strong_topics.empty:
        return [f"🌟 Top | Master {t} first" for t in strong_topics.index[:3]]
    # ------------------------------

    return ["No top performer patterns found"]

def progression_summary(df, student1, student2, time_tolerance=0.35, perf_gap=0.2):
    """
    Compare two students with:
    - Similar start dates (±15 days)
    - Similar total time spent (±15%)
    - Academic performance gap >20%
    """
    s1_data = df[df.StudentID == student1]
    s2_data = df[df.StudentID == student2]

    s1_start = s1_data.ExamDate.min()
    s2_start = s2_data.ExamDate.min()

    s1_duration = s1_data.TimeTaken.sum()
    s2_duration = s2_data.TimeTaken.sum()

    s1_perf = s1_data.Correct.mean()
    s2_perf = s2_data.Correct.mean()

    insights = []

    if abs((s1_start - s2_start).days) > 40:
        return ["Students started more than 40 days apart - not comparable"]

    duration_ratio = abs(s1_duration - s2_duration) / max(s1_duration, s2_duration)
    if duration_ratio > time_tolerance:
        return [f"Time spent on app differs by {duration_ratio:.0%} - beyond {time_tolerance:.0%} threshold"]

    perf_diff = abs(s1_perf - s2_perf)
    if perf_diff < perf_gap:
        return [f"Performance difference {perf_diff:.0%} < {perf_gap:.0%} threshold"]

    better_student = student1 if s1_perf > s2_perf else student2
    weaker_student = student2 if better_student == student1 else student1

    metrics = ['VideosWatched', 'QuizzesTaken', 'PracticeSessions', 'MediaClicks']

    comparisons = []
    for metric in metrics:
        s1_val = int(s1_data[metric].sum())
        s2_val = int(s2_data[metric].sum())
        diff = s1_val - s2_val

        comparisons.append({
            'metric': metric,
            's1_val': s1_val,
            's2_val': s2_val,
            'diff': abs(diff),
            'direction': 'ahead' if diff > 0 else 'behind'
        })

    comparisons.sort(key=lambda x: x['diff'], reverse=True)

    insights.append(
        f"🏆 Better Performer: Student {better_student} "
        f"({s1_perf if better_student == student1 else s2_perf:.0%} vs "
        f"{s2_perf if better_student == student1 else s1_perf:.0%})"
    )

    for comp in comparisons[:3]:
        try:
            insights.append(
                f"📊 {comp['metric']}: You're {comp['direction']} by {comp['diff']} "
                f"(You: {comp['s1_val']} vs Them: {comp['s2_val']})"
            )
        except Exception as e:
            st.error(f"Invalid comparison values: {str(e)}")
            continue

    if comparisons:
        try:
            top_diff = comparisons[0]
            action = "Increase" if top_diff['direction'] == 'behind' else "Maintain"
            insights.append(
                f"🚀 Recommendation: {action} {top_diff['metric'].replace('Watched', '').replace('Taken', '').lower()} "
                f"activities (+{top_diff['diff']} sessions/week)"
            )
        except Exception as e:
            insights.append("🚀 Recommendation: Focus on balanced study habits")
    else:
        insights.append("🚀 Recommendation: Review core concepts and practice regularly")

    return insights

# ==================================================================
# 3. Knowledge Graph Construction
# ==================================================================
def build_knowledge_graph(prereqs, df, topics, OR_thresh=2.0, SHAP_thresh=0.01,
                          min_count=20, application_relations=None):
    if application_relations is None:
        application_relations = APPLICATION_RELATIONS.copy()

    G = nx.DiGraph()

    # Phase 1: Core node initialization
    # ---------------------------------
    for topic in topics:
        G.add_node(topic, type='topic', validated=True)
    for app_key in application_relations:
        G.add_node(app_key, type='application', validated=True)

    # Phase 2: Relationship construction with validation
    # --------------------------------------------------
    # Application relationships
    for app_key, app_info in application_relations.items():
        base_topic = app_info['base_topic']
        if G.has_node(base_topic) and G.has_node(app_key):
            G.add_edge(base_topic, app_key,
                       relation='application',
                       weight=2.5,
                       description=app_info.get('description', ''))
        else:
            missing = [n for n in [base_topic, app_key] if not G.has_node(n)]
            st.error(f"Missing application nodes: {', '.join(missing)}")

    # Prerequisite relationships
    for target, requirements in prereqs.items():
        if not G.has_node(target):
            st.warning(f"Prereq target '{target}' not in core topics")
            continue

        for req in requirements:
            if G.has_node(req) and G.has_node(target):
                G.add_edge(req, target, relation='prereq', weight=3.0, validated=True)
            else:
                missing = [n for n in [req, target] if not G.has_node(n)]
                st.warning(f"Missing prereq nodes: {', '.join(missing)}")

    # Phase 3: Data-driven relationships
    # ----------------------------------
    if not df.empty:
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
                if topic_a not in G.nodes or topic_b not in G.nodes:
                    continue

                # Skip if topics not in struggle matrix
                if topic_a not in struggle_matrix.columns or topic_b not in struggle_matrix.columns:
                    continue

                valid_mask = struggle_matrix[topic_a].notna() & struggle_matrix[topic_b].notna()
                valid_students = struggle_matrix[valid_mask].index

                if len(valid_students) < min_count:
                    continue

                try:
                    # Feature engineering
                    X = pd.DataFrame({
                        'topic_a': struggle_matrix.loc[valid_students, topic_a].astype(int),
                        'proficiency': proficiency.loc[valid_students],
                        'time': avg_time.loc[valid_students]
                    }).dropna()

                    y = struggle_matrix.loc[valid_students, topic_b].astype(int).loc[X.index]

                    if len(X) < 10 or y.nunique() < 2:
                        continue

                    # Odds ratio calculation
                    or_value = np.nan
                    if len(y) >= 50:  # Regularized logistic regression
                        lr = LogisticRegression(penalty='l2', C=0.1, max_iter=1000, random_state=42)
                        lr.fit(X[['topic_a', 'proficiency', 'time']], y)
                        or_value = np.exp(lr.coef_[0][0])
                    else:  # Stratified analysis
                        try:
                            strata = pd.qcut(X.proficiency, q=3, duplicates='drop')
                            if len(strata.unique()) >= 2:
                                contingency_table = np.array([
                                    [np.sum((X.topic_a == 1) & (y == 1)),
                                     np.sum((X.topic_a == 1) & (y == 0))],
                                    [np.sum((X.topic_a == 0) & (y == 1)),
                                     np.sum((X.topic_a == 0) & (y == 0))]
                                ])
                                or_value = (contingency_table[0, 0] * contingency_table[1, 1]) / \
                                           (contingency_table[0, 1] * contingency_table[1, 0])
                        except Exception as e:
                            st.error(f"Stratified odds ratio calculation failed: {str(e)}")
                            or_value = np.nan

                    # SHAP analysis
                    a_importance = 0
                    if len(y) >= 100:
                        try:
                            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                                random_state=42)
                            xgb.fit(X, y)
                            explainer = shap.TreeExplainer(xgb)
                            shap_values = explainer.shap_values(X)
                            a_importance = np.abs(shap_values[:, 0]).mean()
                        except Exception as e:
                            st.error(f"SHAP analysis failed: {str(e)}")

                    # Add edges if valid
                    if or_value > OR_thresh and not np.isnan(or_value):
                        if G.has_node(topic_a) and G.has_node(topic_b):
                            G.add_edge(topic_a, topic_b,
                                       relation='odds_ratio',
                                       weight=min(or_value, 5.0),
                                       validated=False)

                    if a_importance > SHAP_thresh:
                        if G.has_node(topic_a) and G.has_node(topic_b):
                            G.add_edge(topic_a, topic_b,
                                       relation='shap_importance',
                                       weight=min(a_importance * 10, 4.0),
                                       validated=False)

                except Exception as e:
                    st.error(f"Analysis failed for {topic_a}-{topic_b}: {str(e)}")

        except Exception as e:
            st.error(f"Data analysis failed: {str(e)}")

        # Phase 4: Subtopic integration with validation
        # ----------------------------------------------
        seen_subtopics = set()
        for topic in topics:
            topic_data = df[df.Topic == topic]
            if topic_data.empty:
                continue

            # Process subtopics
            subtopic_weights = Counter()
            for _, row in topic_data[topic_data.Correct == 0].iterrows():
                for i in (1, 2, 3):
                    if (subtopic := row.get(f'Subtopic{i}')) and (weight := row.get(f'Weight{i}')):
                        subtopic_weights[subtopic] += weight

            for subtopic, weight in subtopic_weights.most_common(2):
                # Ensure subtopic node exists
                if not G.has_node(subtopic):
                    G.add_node(subtopic, type='subtopic')
                    seen_subtopics.add(subtopic)

                # Add topic-subtopic relationship
                if G.has_node(topic):
                    G.add_edge(topic, subtopic, relation='subtopic',
                               weight=int(min(weight, 5)))

                # Connect prerequisites to subtopic
                for prereq in list(G.predecessors(topic)):  # Convert to list for safe iteration
                    if prereq != subtopic and G.has_node(prereq):
                        G.add_edge(prereq, subtopic, relation='sub_prereq', weight=1.5)

        # Phase 5: Final graph validation
        # -------------------------------
        # Remove invalid edges post-construction
        invalid_edges = []
        for s, t in list(G.edges()):  # Convert to list for safe removal
            if None in (s, t) or not G.has_node(s) or not G.has_node(t):
                invalid_edges.append((s, t))

        for s, t in invalid_edges:
            G.remove_edge(s, t)
            st.error(f"Removed invalid edge: {s} → {t}")

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
    """More impactful graph updates"""
    try:
        responses = st.session_state.quiz_responses.get(sid, {}).get(topic, [])

        # 1. Safer mastery calculation
        total = len(responses)
        if total == 0:
            return

        mastery = sum(1 for r in responses if r.get('is_correct', False)) / total
        mastery = min(max(mastery, 0), 1)  # Clamp between 0-1

        # 2. Safe node updates
        if G.has_node(topic):
            G.nodes[topic]['mastery'] = mastery
            G.nodes[topic]['last_attempt'] = datetime.now().isoformat()
            st.success(f"Updated mastery for {topic} to {mastery:.0%}")

        # 3. Robust subtopic processing
        seen_subtopics = set()
        subtopic_weights = Counter()

        for response in responses:
            try:
                # 4. Safe question lookup across all topics
                question = None
                for t in QUESTION_BANK:
                    for q in QUESTION_BANK[t]:
                        if q['id'] == response['qid']:
                            question = q
                            break
                    if question:
                        break

                if not question:
                    st.error(f"Missing question: {response['qid']}")
                    continue

                # 5. Validate subtopic fields
                for i in (1, 2, 3):
                    subtopic = question.get(f'subtopic{i}')
                    if subtopic:
                        subtopic_weights[subtopic] += int(response.get('is_correct', 0))

            except Exception as e:
                st.error(f"Error processing response: {str(e)}")
                continue

        # 6. Safe edge creation
        for subtopic, weight in subtopic_weights.most_common(2):
            if not G.has_node(subtopic):
                G.add_node(subtopic, type='subtopic')
                seen_subtopics.add(subtopic)
            G.add_edge(topic, subtopic, relation='subtopic', weight=weight)

    except Exception as e:
        st.error(f"Knowledge graph update failed: {str(e)}")


def get_connected_practice_topics(G, completed_topics, sid):
    """Get topics connected to completed ones with tier 1/2 edges"""
    practice_topics = []

    # Edge tier mapping (1=strong, 2=medium, 3=weak)
    for t in completed_topics:
        # Get neighbors with strong/medium connections
        neighbors = []
        for _, neighbor, data in G.out_edges(t, data=True):
            if data.get('tier', 3) in [1, 2]:  # Only strong/medium tiers
                neighbors.append((neighbor, data.get('weight', 1)))

        # Sort by connection strength
        neighbors.sort(key=lambda x: -x[1])
        practice_topics.extend([n[0] for n in neighbors[:2]])  # Top 2 per topic

    # Get unique topics with completion status
    return [t for t in practice_topics
            if t in completed_topics][:3]  # Max 3 recommendations
def get_connection_strength(G, topic, completed):
    """Calculate average edge strength from completed topics"""
    strengths = []
    for ct in completed:
        if G.has_edge(ct, topic):
            strengths.append(G[ct][topic].get('weight', 1))
    return f"{np.mean(strengths):.1f}/5" if strengths else "N/A"
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
def get_quiz_recommendations(sid, sd):
    rec = []
    completed_topics = sd[sd.Completed].Topic.unique()
    topic_perf = sd.groupby('Topic').Correct.mean()

    for t in completed_topics:
        accuracy = topic_perf.get(t, 1.0)
        if accuracy < 0.35 and t in FORMULA_QUIZ_BANK:
            p = st.session_state.quiz_progress.setdefault(sid, {}).get(t, 0)
            subs = list(FORMULA_QUIZ_BANK[t].keys())

            status = "Needs practice" if accuracy < 0.5 else "Review ready"
            color = "🔴" if accuracy < 0.5 else "🟢"

            if p < len(subs):
                rec.append(
                    f"📝 Quiz | {subs[p]} ({t}) {color} "
                    f"[Your score: {accuracy:.0%}]"
                )
            else:
                rec.append(
                    f"🎯 Mastered | {t} quizzes 🟢 "
                    f"[Final accuracy: {accuracy:.0%}]"
                )
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
    topper_recs = recommend_topper_resources(seg, df)
    if topper_recs:
        rec.extend([f"🌟 Topper_habit | {r}" for r in topper_recs[:3]])
    # 1) Add a motivation quote
    if comp:
        selected_topic = np.random.choice(comp)
        if selected_topic in MOTIVATION_QUOTES:
            rec.append(f"💭Quote | \"{MOTIVATION_QUOTES[selected_topic]}\"")

    # 2) Bridge course recommendations for critical low-accuracy topics
    critical_low_topics = acc[acc < 0.3].index.tolist()
    for t in critical_low_topics:
        if t in BRIDGE_COURSES:
            rec.append(f"🚨 Urgent_course | {t} - {BRIDGE_COURSES[t]} [Accuracy: {acc[t]:.0%}]")

    # 3) HOTS for strong topics (accuracy >=70% and motivation not Low)
    if mot != 'Low':
        strong_topics = acc[acc >= 0.7].index.tolist()
        for t in strong_topics[:2]:  # Limit to 2 strongest
            if t in HOTS_QUESTIONS:
                rec.append(
                    f"🏆 Strong(HOTS) | {t} (Accuracy: {acc[t]:.0%}) - "
                    f"Try: {', '.join(HOTS_QUESTIONS[t][:2])}"
                )
    # 4) Practice recommendations (combined logic)
    if comp:
        practice_topics = list(set(
            acc[(acc >= 0.4) & (acc < 0.7)].index.tolist()[:3] +  # Mid-strength
            get_connected_practice_topics(G, comp, sid)  # Connected
        ))[:3]  # Unique topics, max 3

        for t in practice_topics:
            if t in PRACTICE_QUESTIONS:
                pq = PRACTICE_QUESTIONS[t]
                # Unified priority sequence
                seq = (pq.get('recent', []) +
                       pq.get('mid_level', []) +
                       pq.get('historical', []) +
                       pq.get('fundamental', []))

                # Connection context
                conn_strength = get_connection_strength(G, t, comp)
                context = " (connected)" if conn_strength else " (practice zone)"

                rec.append(
                    f"📚 Practice(less_than_70) | {t}{context} [{acc[t]:.0%} accuracy] - " +
                    f"{', '.join(seq[:3])} " +
                    f"[Strength: {conn_strength}]"
                )

    # 5) Quiz recommendations
    quiz_recs = [f"📝 Quiz | {q}" for q in get_quiz_recommendations(sid, sd)]
    rec.extend(quiz_recs[:2])
    # New: Needs Work recommendations (20-59% accuracy)
    needs_work_topics = acc[(acc >= 0.2) & (acc < 0.6)].index.tolist()
    for t in needs_work_topics:
        rec.append(f"🚨 Needs Work | {t} - Focused Practice Plan [Accuracy: {acc[t]:.0%}]")

    # 6) Media & analogies
    seen_videos = set()
    for t in comp[:2]:  # Limit to 2 recent completions
        if m := MEDIA_LINKS.get(t):
            for video in m.get('videos', [])[:1]:
                if video not in seen_videos:
                    rec.append(f"🎥 Media | {video}")
                    seen_videos.add(video)
            if analogy := m.get('analogies'):
                rec.append(f"🔗 Analogy | {analogy}")
    for t in acc[acc < 0.5].index.tolist():
        if not MEDIA_LINKS.get(t):
            rec.append(f"🎥 Media | General {t} Concepts: https://example.com/{t.replace(' ', '')}-intro")
            rec.append(f"🔗 Analogy | {t} is like... [Ask teacher for analogy]")


    # 8) Easy‐win topics if motivation is Low
    if mot == 'Low' and comp:
        easy_topic = np.random.choice(comp)
        if easy_topic in EASY_TOPICS:
            rec.append(f"👍 Easy | {easy_topic} - {EASY_TOPICS[easy_topic][0]}")

        # b) collaborative filtering suggestions
        collab_recs = [f"👥 Peer | {cr}" for cr in collaborative_filtering_recommendations(sid, df, seg)]
        rec.extend(collab_recs)

        # c) “future” topics (prereqs or odds_ratio)
        future_topics = [
            v for _, v, d in G.out_edges(easy_topic, data=True)
            if d.get('relation') in ('prereq', 'odds_ratio')
        ]
        if future_topics:
            rec.append(f"🔄 Apply | {easy_topic} in: {', '.join(future_topics[:2])}")

        # d) direct applications
        apps = [
            v for _, v, d in G.out_edges(easy_topic, data=True)
            if d.get('relation') in ('app_preparation', 'application')
        ]
        if apps:
            rec.append(f"🔬 Real |  {easy_topic}: {', '.join(apps[:2])}")
    return rec

# ==================================================================
# 7. Question‑Level Analytics
# ==================================================================
def track_question_response(sid, question_id, is_correct):
    """Enhanced tracking with graph updates"""
    if 'question_responses' not in st.session_state:
        st.session_state.question_responses = {}

    # Record response
    student_responses = st.session_state.question_responses.setdefault(sid, {})
    question_history = student_responses.setdefault(question_id, [])
    question_history.append(int(is_correct))

    # Update knowledge graph
    G = st.session_state.knowledge_graph
    for topic in QUESTION_BANK:
        if any(q['id'] == question_id for q in QUESTION_BANK[topic]):
            update_knowledge_graph_with_quiz(G, sid, topic)
            break


def analyze_item_level_performance(sid):
    """Enhanced analysis with graph visualization"""
    if sid not in st.session_state.question_responses:
        return []

    G = st.session_state.knowledge_graph
    responses = st.session_state.question_responses[sid]
    problem_questions = []

    for q_id, attempts in responses.items():
        if len(attempts) >= 1:
            success_rate = sum(attempts) / len(attempts)
            if success_rate < 0.5:
                # Find question details
                for topic, questions in QUESTION_BANK.items():
                    for q in questions:
                        if q['id'] == q_id:
                            # Get graph connections
                            connections = []
                            if G.has_node(topic):
                                edges = list(G.in_edges(topic)) + list(G.out_edges(topic))
                                connections = [f"{s}→{t}" for s, t in edges]

                            problem_questions.append((
                                q_id,
                                q['text'],
                                topic,
                                success_rate,
                                q.get('solution_steps', 'No feedback available'),
                                connections  # Add connections to output
                            ))
                            break
                    if q_id in [q['id'] for q in questions]:
                        break

    return sorted(problem_questions, key=lambda x: x[3])



# ==================================================================
# 8. Streamlit UI
# ==================================================================
# ==================================================================
# 8. Streamlit UI
# ==================================================================
def main():
    if 'knowledge_graph' not in st.session_state:
        st.session_state.knowledge_graph = nx.DiGraph()
    st.set_page_config(layout="wide", page_title="Learning Dashboard", page_icon="🧠")

    # Modern CSS styling
    st.markdown("""
    <style>
    /* Glassmorphism effects */
    [data-testid="stExpander"], [data-testid="stVerticalBlock"] > div > div {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        padding: 1rem !important;
    }

    /* Gradient headers */
    .section-header {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white !important;
        padding: 8px 16px;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    /* Interactive cards */
    .interactive-card {
        transition: transform 0.2s;
        cursor: pointer;
        border: 1px solid #e5e7eb !important;
    }
    .interactive-card:hover {
        transform: translateY(-2px);
    }

    /* Modern progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%);
        height: 10px !important;
        border-radius: 5px;
    }

    /* Code block styling */
    .stCodeBlock {
        background: #0f172a !important;
        border-radius: 8px;
        padding: 1rem;
    }

    /* Metric styling */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px;
        padding: 15px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if 'quiz_progress' not in st.session_state:
        st.session_state.quiz_progress = {}
    if 'question_responses' not in st.session_state:
        st.session_state.question_responses = {}

    st.markdown('<div class="section-header">Enhanced Learning Dashboard 🧠</div>', unsafe_allow_html=True)

    # Initialize critical variables
    sid = None
    df = pd.DataFrame()
    G = st.session_state.knowledge_graph

    try:
        with st.spinner("Loading student data..."):
            df = generate_student_data()
            if df.empty:
                st.error("No student data generated")
                st.stop()

        # Modified graph building section
        if not nx.nodes(st.session_state.knowledge_graph):
            with st.spinner("Building knowledge graph..."):
                st.session_state.knowledge_graph = build_knowledge_graph(PREREQUISITES, df, TOPICS)
                apply_dual_tier_scoring(st.session_state.knowledge_graph)

        # Segment students
        with st.spinner("Analyzing student segments..."):
            seg = segment_students(df)
            if seg.empty:
                st.error("Student segmentation failed")
                st.stop()

        # Sidebar settings
        st.sidebar.markdown('<div class="section-header">Settings</div>', unsafe_allow_html=True)

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

        # Motivation override
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

        # Peer Tutoring Section
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
            st.markdown('<div class="section-header">Knowledge Graph</div>', unsafe_allow_html=True)
            with st.container(height=600, border=True):
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
                        height=550
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Topic colors: 🔴 Needs work | 🟡 Average | 🟢 Strong | 🟠 No data")
                except Exception as e:
                    st.error(f"Graph display error: {str(e)}")

        # Recommendations Panel
        with col2:
            st.markdown('<div class="section-header">Personalized Learning Plan</div>', unsafe_allow_html=True)
            if sid is not None:
                try:
                    recommendations = get_recommendations(sid, df, G, seg, override_mot)

                    # Categorization dictionary with updated structure
                    rec_types = {
                        "🚨 Urgent": [],
                        "🚨 Needs Work": [],
                        "📚 Practice": [],
                        "📝 Quiz": [],
                        "🎥 Media": [],
                        "🔗 Analogy": [],
                        "🔄 Apply": [],
                        "💭 Quote": [],
                        "🏆 Strong": [],
                        "🌟 Topper Habit": []
                    }

                    # Enhanced categorization logic
                    for r in recommendations:
                        if r.startswith("🚨 Urgent_course"):
                            rec_types["🚨 Urgent"].append(r.split("|")[-1])
                        elif r.startswith("🚨 Needs Work"):
                            rec_types["🚨 Needs Work"].append(r.split("|")[-1])
                        elif r.startswith("📚 Practice"):
                            rec_types["📚 Practice"].append(r.split("|")[-1])
                        elif r.startswith("📝 Quiz"):
                            rec_types["📝 Quiz"].append(r.split("|")[-1])
                        elif r.startswith("🎥 Media"):
                            rec_types["🎥 Media"].append(r)
                        elif r.startswith("🔗 Analogy"):
                            rec_types["🔗 Analogy"].append(r)
                        elif r.startswith("🔄 Apply"):
                            rec_types["🔄 Apply"].append(r.split("|")[-1])
                        elif r.startswith("💭 Quote"):
                            rec_types["💭 Quote"].append(r)
                        elif r.startswith("🏆 Strong"):
                            rec_types["🏆 Strong"].append(r.split("|")[-1])
                        elif r.startswith("🌟 Topper_habit"):
                            rec_types["🌟 Topper Habit"].append(r.split("→ ")[-1])

                    # Visual priority matrix
                    with st.container(border=True):
                        cols = st.columns([1, 2])
                        with cols[0]:
                            st.markdown("#### 🗺️ Knowledge Map")
                            for topic, color in zip(node_text, node_colors):
                                st.markdown(f"<span style='color:{color}'>●</span> {topic}",
                                            unsafe_allow_html=True)

                        with cols[1]:
                            st.markdown("#### 🎯 Action Zones")
                            tabs = st.tabs(["Urgent Needs", "Practice Areas", "Strengths"])

                            with tabs[0]:
                                for item in rec_types["🚨 Urgent"] + rec_types["🚨 Needs Work"]:
                                    st.error(item, icon="🚨")

                            with tabs[1]:
                                for item in rec_types["📚 Practice"] + rec_types["📝 Quiz"]:
                                    st.info(item, icon="📘")
                                    st.progress(0.5, text="Mastery Progress")

                            with tabs[2]:
                                for item in rec_types["🏆 Strong"]:
                                    st.success(item, icon="💪")
                                    st.button("Challenge Yourself →", key=f"challenge_{item[:10]}")

                    # Engagement Hub
                    with st.container(border=True):
                        cols = st.columns(3)
                        with cols[0]:
                            st.markdown("#### 🧠 Mindset")
                            for quote in rec_types["💭 Quote"]:
                                st.markdown(f"""```diff
+ {quote.split("|")[-1]}
```""")

                        with cols[1]:
                            st.markdown("#### 🎮 Interactive")
                            for media in rec_types["🎥 Media"]:
                                st.video(media.split("|")[-1])
                            for analogy in rec_types["🔗 Analogy"]:
                                st.info(f"**Real-world Connection**\n{analogy.split('|')[-1]}")

                        with cols[2]:
                            st.markdown("#### 🛠️ Applications")
                            for app in rec_types["🔄 Apply"]:
                                st.success(f"🔧 {app}")
                            st.button("Explore More →", use_container_width=True)

                except Exception as e:
                    st.error(f"Recommendation error: {str(e)}")

        # Strategic Peer Comparison
        with st.expander("🔍 Peer Benchmarking", expanded=True):
            if not df.empty and sid is not None:
                try:
                    current_data = df[df.StudentID == sid]
                    current_start = current_data.ExamDate.min()
                    comparable_peers = df[
                        (df.ExamDate > current_start - pd.Timedelta(days=40)) &
                        (df.ExamDate < current_start + pd.Timedelta(days=40)) &
                        (df.StudentID != sid)
                        ].StudentID.unique()

                    if len(comparable_peers) > 0:
                        peer_perf = df[df.StudentID.isin(comparable_peers)].groupby('StudentID').Correct.mean()
                        current_perf = current_data.Correct.mean()
                        peer_diff = abs(peer_perf - current_perf)

                        if not peer_diff.empty:
                            best_peer = peer_diff.idxmax()
                            perf_gap = peer_diff.max()

                            if perf_gap >= 0.2:
                                # Modern layout
                                with st.container(border=True):
                                    cols = st.columns([1, 3])
                                    with cols[0]:
                                        st.metric("Your Performance Tier", perf_tier,
                                                  delta=f"vs {len(comparable_peers)} peers")
                                        st.plotly_chart(fig, use_container_width=True)

                                    with cols[1]:
                                        st.markdown("#### 📊 Activity Comparison")
                                        tabs = st.tabs(["Study Patterns", "Progress Timeline", "Efficiency Matrix"])

                                        with tabs[0]:
                                            # Radar chart for activity comparison
                                            s1_vid = current_data.VideosWatched.mean()
                                            s1_quiz = current_data.QuizzesTaken.mean()
                                            s1_prac = current_data.PracticeSessions.mean()
                                            s2_vid = df[df.StudentID == best_peer].VideosWatched.mean()
                                            s2_quiz = df[df.StudentID == best_peer].QuizzesTaken.mean()
                                            s2_prac = df[df.StudentID == best_peer].PracticeSessions.mean()

                                            fig = go.Figure()
                                            fig.add_trace(go.Scatterpolar(
                                                r=[s1_vid, s1_quiz, s1_prac],
                                                theta=['Videos', 'Quizzes', 'Practice'],
                                                fill='toself',
                                                name='You'
                                            ))
                                            fig.add_trace(go.Scatterpolar(
                                                r=[s2_vid, s2_quiz, s2_prac],
                                                theta=['Videos', 'Quizzes', 'Practice'],
                                                fill='toself',
                                                name=f'Peer {best_peer}'
                                            ))
                                            st.plotly_chart(fig, use_container_width=True)

                                        with tabs[1]:
                                            # Timeline visualization
                                            timeline_data = pd.DataFrame({
                                                'Date': pd.date_range(start=current_start - pd.Timedelta(days=30),
                                                                      periods=60),
                                                'Your Progress': np.random.rand(60).cumsum(),
                                                'Peer Progress': np.random.rand(60).cumsum() * 1.2
                                            }).set_index('Date')
                                            st.area_chart(timeline_data)

                                        with tabs[2]:
                                            # Heatmap of activity vs performance
                                            efficiency_matrix = """
                                            | Activity   | Efficiency | Impact |
                                            |------------|------------|--------|
                                            | Videos     | 65%        | ★★★☆☆ |
                                            | Quizzes    | 82%        | ★★★★☆ |
                                            | Practice   | 94%        | ★★★★★ |
                                            """
                                            st.write("```\n" + efficiency_matrix + "\n```")

                                # Strategy cards
                                with st.container():
                                    cols = st.columns(3)
                                    with cols[0]:
                                        with st.container(border=True, height=200):
                                            st.markdown("#### 🏆 Top Performer Insight")
                                            st.caption("What successful peers do differently")
                                            top_strat = rec_types["🌟 Topper Habit"][0] if rec_types["🌟 Topper Habit"] else "N/A"
                                            st.write(f"```\n{top_strat}\n```")

                                    with cols[1]:
                                        with st.container(border=True, height=200):
                                            st.markdown("#### ⚡ Quick Wins")
                                            quick_win_score = min(int(perf_gap * 100) / 100, 0.75)
                                            st.progress(quick_win_score, "Immediate impact potential")
                                            st.button("Implement Now →")

                                    with cols[2]:
                                        with st.container(border=True, height=200):
                                            st.markdown("#### 📅 Weekly Plan")
                                            st.write("1. 2h Focus Sessions\n2. Peer Reviews\n3. Skill Drills")
                                            weekly_plan = "Sample plan content"
                                            st.download_button("Export Plan", data=weekly_plan)

                except Exception as e:
                    st.error(f"Comparison failed: {str(e)}")

        # Quiz Section
        st.markdown('<div class="section-header">Interactive Quiz Section</div>', unsafe_allow_html=True)
        try:
            comp = df[(df.StudentID == sid) & (df.Completed)].Topic.unique().tolist()
            quiz_topics = [t for t in comp if t in FORMULA_QUIZ_BANK]

            if quiz_topics:
                c1, c2 = st.columns([1, 2])
                with c1:
                    selected_topic = st.selectbox("Select Quiz Topic", quiz_topics)
                    if st.button("Start Quiz", type="primary"):
                        # Initialize quiz session state
                        st.session_state.show_quiz = True
                        st.session_state.quiz_topic = selected_topic
                        st.session_state.quiz_answers = {}

                        # Pre-initialize all question keys
                        if selected_topic in FORMULA_QUIZ_BANK:
                            topic_data = FORMULA_QUIZ_BANK[selected_topic]
                            for sub, questions in topic_data.items():
                                for q in questions:
                                    q_key = f"q_{selected_topic}_{q['id']}"
                                    if q_key not in st.session_state:
                                        st.session_state[q_key] = ""
                        else:
                            st.error("Invalid quiz topic selected")
                            st.session_state.show_quiz = False

                with c2:
                    if st.session_state.get('show_quiz'):
                        topic = st.session_state.quiz_topic

                        # Validate topic exists in quiz bank
                        if topic not in FORMULA_QUIZ_BANK:
                            st.error("Invalid quiz topic selected")
                            return

                        # Validate questions exist for topic
                        if not FORMULA_QUIZ_BANK[topic]:
                            st.error("No questions available for this topic")
                            return

                        st.markdown(f"### {topic} Quiz")

                        with st.form(key=f"quiz_form_{topic}"):
                            # Initialize form with proper question IDs
                            for sub, questions in FORMULA_QUIZ_BANK[topic].items():
                                st.subheader(f"Section: {sub}")
                                for q in questions:
                                    q_key = f"q_{topic}_{q['id']}"
                                    st.markdown(f"**Q{q['id']}:** {q['question']}")

                                    if q['type'] == 'formula':
                                        st.text_input("Your answer:",
                                                      key=q_key,
                                                      value=st.session_state.get(q_key, ""))
                                    else:
                                        options = q.get('options', ["Option A", "Option B", "Option C"])
                                        st.radio("Select:", options,
                                                 key=q_key,
                                                 index=0)
                                    st.markdown("---")

                            if st.form_submit_button("Submit Quiz"):
                                try:
                                    # Initialize response structures
                                    if 'quiz_responses' not in st.session_state:
                                        st.session_state.quiz_responses = {}
                                    if 'question_responses' not in st.session_state:
                                        st.session_state.question_responses = {}

                                    # Initialize student-specific structures
                                    if sid not in st.session_state.quiz_responses:
                                        st.session_state.quiz_responses[sid] = {}
                                    if sid not in st.session_state.question_responses:
                                        st.session_state.question_responses[sid] = {}

                                    # Generate responses from form data
                                    responses = []
                                    for sub, questions in FORMULA_QUIZ_BANK[topic].items():
                                        for q in questions:
                                            q_key = f"q_{topic}_{q['id']}"
                                            student_answer = st.session_state.get(q_key, "")
                                            responses.append({
                                                "qid": q['id'],
                                                "answer": student_answer
                                            })

                                    # Process all responses
                                    for response in responses:
                                        try:
                                            # Find question safely
                                            question = None
                                            for t in QUESTION_BANK:
                                                for q in QUESTION_BANK[t]:
                                                    if q['id'] == response['qid']:
                                                        question = q
                                                        break
                                                if question:
                                                    break

                                            if not question:
                                                st.error(f"Missing question: {response['qid']}")
                                                continue

                                            # Validate answer
                                            is_correct, feedback = validate_answer(question, response['answer'])

                                            # Track response
                                            track_question_response(sid, question['id'], is_correct)

                                            # Store detailed response
                                            if topic not in st.session_state.quiz_responses[sid]:
                                                st.session_state.quiz_responses[sid][topic] = []

                                            st.session_state.quiz_responses[sid][topic].append({
                                                "qid": question['id'],
                                                "answer": response['answer'],
                                                "is_correct": is_correct,
                                                "timestamp": datetime.now().isoformat(),
                                                "feedback": feedback
                                            })

                                        except Exception as e:
                                            st.error(f"Error processing response: {str(e)}")
                                            continue

                                    # Update knowledge graph
                                    update_knowledge_graph_with_quiz(st.session_state.knowledge_graph, sid,
                                                                     topic)

                                    # Clear temporary input states
                                    for sub, questions in FORMULA_QUIZ_BANK[topic].items():
                                        for q in questions:
                                            q_key = f"q_{topic}_{q['id']}"
                                            if q_key in st.session_state:
                                                del st.session_state[q_key]

                                    st.success("Quiz submitted successfully! Recommendations updated.")
                                    st.session_state.show_quiz = False

                                except Exception as e:
                                    st.error(f"Quiz submission error: {str(e)}")

        except Exception as e:
            st.error(f"Quiz section error: {str(e)}")

        # Performance Analytics
        st.markdown('<div class="section-header">Performance Analytics</div>', unsafe_allow_html=True)
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
                            'VideosWatched': 'sum',
                            'QuizzesTaken': 'sum',
                            'PracticeSessions': 'sum',
                            'MediaClicks': 'sum'
                        }).reset_index()

                        if not usage_data.empty:
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=usage_data['Topic'],
                                y=usage_data['VideosWatched'],
                                name='Videos',
                                marker=dict(color='#2196F3')
                            ))
                            fig.add_trace(go.Bar(
                                x=usage_data['Topic'],
                                y=usage_data['QuizzesTaken'],
                                name='Quizzes',
                                marker=dict(color='#4CAF50')
                            ))
                            fig.add_trace(go.Bar(
                                x=usage_data['Topic'],
                                y=usage_data['PracticeSessions'],
                                name='Practice',
                                marker=dict(color='#FF9800')
                            ))
                            fig.add_trace(go.Bar(
                                x=usage_data['Topic'],
                                y=usage_data['MediaClicks'],
                                name='Media',
                                marker=dict(color='#9C27B0')
                            ))

                            fig.update_layout(
                                barmode='stack',
                                title='Study Activity Distribution',
                                xaxis_title='Topics',
                                yaxis_title='Total Activities',
                                height=400,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Analytics error: {str(e)}")

        # Question-Level Analytics
        st.markdown('<div class="section-header">Question-Level Analytics</div>', unsafe_allow_html=True)
        try:
            if sid is not None:
                problem_questions = analyze_item_level_performance(sid)
                if problem_questions:
                    st.subheader("Knowledge Graph Impact Analysis")

                    for q in problem_questions:
                        with st.expander(f"{q[2]}: {q[1]} (Success Rate: {q[3]:.0%})"):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown("**Solution Steps:**")
                                for step in q[4]:
                                    st.write(f"- {step}")
                            with col2:
                                st.markdown("**Graph Connections:**")
                                if q[5]:
                                    for conn in q[5]:
                                        st.write(f"`{conn}`")
                                else:
                                    st.write("No connections found")

                                st.metric("Current Mastery",
                                          f"{st.session_state.knowledge_graph.nodes[q[2]].get('mastery', 0):.0%}",
                                          help="Calculated from quiz performance")
        except Exception as e:
            st.error(f"Question analysis error: {str(e)}")

    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.stop()


if __name__ == "__main__":
    main()
