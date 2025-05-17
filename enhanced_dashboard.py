
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
        'recent': ['Chemical Kinetics', 'Chemical Equilibrium'],
        'historical': ['Periodic Table', 'Chemical Bonds'],
        'fundamental': ['States of Matter', 'Element Properties'],
        'mid_level': ['Solutions', 'Electrochemistry']
    },
    'Biology': {
        'recent': ['Molecular Biology', 'Biotechnology'],
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
    """
    Validate a student's answer against the expected answer.

    Args:
        question: Question dictionary with 'answer' key
        student_answer: The student's submitted answer

    Returns:
        tuple: (is_correct, feedback)
    """
    try:
        if question['type'] == 'formula':
            # For formula questions, normalize and compare
            expected_answer = str(question.get('answer', '')).strip().lower()
            student_answer = str(student_answer).strip().lower()

            # Basic normalization for formula comparison
            expected_normalized = expected_answer.replace(" ", "").replace("*", "")
            student_normalized = student_answer.replace(" ", "").replace("*", "")

            is_correct = (expected_normalized == student_normalized)

            if is_correct:
                feedback = "Correct! Good job."
            else:
                feedback = f"Incorrect. The correct answer is: {question.get('answer', 'Unknown')}"

        else:  # Multiple choice
            # For multiple choice, direct comparison
            correct_option = question.get('answer', '')
            is_correct = (student_answer == correct_option)

            if is_correct:
                feedback = "Correct! Good job."
            else:
                feedback = f"Incorrect. The correct answer is: {correct_option}"

        return is_correct, feedback

    except Exception as e:
        st.error(f"Error validating answer: {str(e)}")
        return False, f"Error validating answer: {str(e)}"


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
    """More impactful graph updates with debugging"""
    try:
        # DEBUG: Check if quiz responses exist
        if 'quiz_responses' not in st.session_state:
            st.error("No quiz_responses in session state")
            st.session_state.quiz_responses = {}

        # DEBUG: Check if sid exists in quiz responses
        if sid not in st.session_state.quiz_responses:
            st.error(f"No responses for student {sid}")
            st.session_state.quiz_responses[sid] = {}

        # DEBUG: Check if topic exists in quiz responses for sid
        if topic not in st.session_state.quiz_responses.get(sid, {}):
            st.error(f"No responses for topic {topic}")
            st.session_state.quiz_responses[sid][topic] = []

        # Get responses with fallback
        responses = st.session_state.quiz_responses.get(sid, {}).get(topic, [])

        # DEBUG: Print responses for debugging
        st.write(f"DEBUG - Found {len(responses)} responses for {topic}")
        if len(responses) > 0:
            st.write(f"First response: {responses[0]}")

        # Safer mastery calculation with better fallback
        total = len(responses)
        if total == 0:
            st.warning(f"No quiz responses found for {topic}")
            # Set default mastery to 0
            mastery = 0
        else:
            # Calculate mastery from responses
            correct_count = sum(1 for r in responses if r.get('is_correct', False))
            mastery = correct_count / total
            mastery = min(max(mastery, 0), 1)  # Clamp between 0-1
            st.write(f"DEBUG - Correct answers: {correct_count}/{total}")

        # Update node in graph
        if not G.has_node(topic):
            G.add_node(topic, type='topic', mastery=mastery, last_attempt=datetime.now().isoformat())
        else:
            # Update existing node attributes
            G.nodes[topic]['mastery'] = mastery
            G.nodes[topic]['last_attempt'] = datetime.now().isoformat()

        # Only show the success message once
        st.success(f"Updated mastery for {topic} to {int(mastery * 100)}%")

        # Only continue with subtopic processing if we have responses
        if total > 0:
            # Process subtopics
            seen_subtopics = set()
            subtopic_weights = Counter()

            # Find all questions that match the topic
            all_questions = []
            if topic in FORMULA_QUIZ_BANK:
                for sub, questions in FORMULA_QUIZ_BANK[topic].items():
                    all_questions.extend(questions)

            for response in responses:
                try:
                    # First look in FORMULA_QUIZ_BANK flattened questions
                    question = next((q for q in all_questions if q['id'] == response['qid']), None)

                    # Fallback to QUESTION_BANK if not found
                    if not question and topic in QUESTION_BANK:
                        question = next((q for q in QUESTION_BANK[topic] if q['id'] == response['qid']), None)

                    if not question:
                        st.warning(f"Missing question: {response['qid']}")
                        continue

                    # Process subtopics from either question format
                    for i in (1, 2, 3):
                        subtopic = question.get(f'subtopic{i}')
                        if subtopic:
                            subtopic_weights[subtopic] += int(response.get('is_correct', 0))

                except Exception as e:
                    st.warning(f"Error processing response: {str(e)}")
                    continue

            # Safe edge creation
            for subtopic, weight in subtopic_weights.most_common(2):
                if not G.has_node(subtopic):
                    G.add_node(subtopic, type='subtopic')
                    seen_subtopics.add(subtopic)
                G.add_edge(topic, subtopic, relation='subtopic', weight=weight)

        # Update quiz progress in a separate session state
        if 'quiz_progress' not in st.session_state:
            st.session_state.quiz_progress = {}
        if sid not in st.session_state.quiz_progress:
            st.session_state.quiz_progress[sid] = {}

        # If mastery is high enough, increment the subtopic progress
        if mastery >= 0.7:  # 70% mastery
            current_progress = st.session_state.quiz_progress.get(sid, {}).get(topic, 0)
            st.session_state.quiz_progress[sid][topic] = current_progress + 1

    except Exception as e:
        st.error(f"Knowledge graph update failed: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


# Modified version of the quiz submission handler
def process_quiz_submission(sid, topic):
    """Process quiz submission and store responses"""
    try:
        # Initialize response structures if not present
        if 'quiz_responses' not in st.session_state:
            st.session_state.quiz_responses = {}
        if 'question_responses' not in st.session_state:
            st.session_state.question_responses = {}

        # Initialize student-specific structures
        if sid not in st.session_state.quiz_responses:
            st.session_state.quiz_responses[sid] = {}
        if sid not in st.session_state.question_responses:
            st.session_state.question_responses[sid] = {}

        # Validate topic again (defense in depth)
        if topic not in FORMULA_QUIZ_BANK:
            st.error("Invalid quiz topic selected")
            return False

        # Clear any previous responses for this topic to avoid duplicates
        st.session_state.quiz_responses[sid][topic] = []

        # Process all questions with the new error handling
        for sub, questions in FORMULA_QUIZ_BANK[topic].items():
            for q in questions:
                q_key = f"q_{topic}_{q['id']}"
                student_answer = st.session_state.get(q_key, "")

                # Validate answer directly using the question from FORMULA_QUIZ_BANK
                is_correct, feedback = validate_answer(q, student_answer)

                # Track response
                track_question_response(sid, q['id'], is_correct)

                # Store detailed response
                st.session_state.quiz_responses[sid][topic].append({
                    "qid": q['id'],
                    "answer": student_answer,
                    "is_correct": is_correct,
                    "timestamp": datetime.now().isoformat(),
                    "feedback": feedback
                })

        # Log the responses for debugging
        response_count = len(st.session_state.quiz_responses[sid][topic])
        correct_count = sum(1 for r in st.session_state.quiz_responses[sid][topic] if r.get('is_correct', False))
        st.write(f"DEBUG - Processed {response_count} responses with {correct_count} correct")

        return True

    except Exception as e:
        st.error(f"Error processing quiz: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


def validate_answer(question, student_answer):
    """
    Validate a student's answer against the expected answer.

    Args:
        question: Question dictionary with 'answer' key
        student_answer: The student's submitted answer

    Returns:
        tuple: (is_correct, feedback)
    """
    try:
        if question['type'] == 'formula':
            # For formula questions, normalize and compare
            expected_answer = str(question.get('answer', '')).strip().lower()
            student_answer = str(student_answer).strip().lower()

            # Basic normalization for formula comparison
            expected_normalized = expected_answer.replace(" ", "").replace("*", "")
            student_normalized = student_answer.replace(" ", "").replace("*", "")

            is_correct = (expected_normalized == student_normalized)

            if is_correct:
                feedback = "Correct! Good job."
            else:
                feedback = f"Incorrect. The correct answer is: {question.get('answer', 'Unknown')}"

        else:  # Multiple choice
            # For multiple choice, direct comparison
            correct_option = question.get('answer', '')
            is_correct = (student_answer == correct_option)

            if is_correct:
                feedback = "Correct! Good job."
            else:
                feedback = f"Incorrect. The correct answer is: {correct_option}"

        return is_correct, feedback

    except Exception as e:
        st.error(f"Error validating answer: {str(e)}")
        return False, f"Error validating answer: {str(e)}"


def track_question_response(sid, question_id, is_correct):
    """
    Track student responses to questions for analytics.

    Args:
        sid: Student ID
        question_id: Question identifier
        is_correct: Whether the answer was correct
    """
    try:
        if 'question_responses' not in st.session_state:
            st.session_state.question_responses = {}

        if sid not in st.session_state.question_responses:
            st.session_state.question_responses[sid] = {}

        if question_id not in st.session_state.question_responses[sid]:
            st.session_state.question_responses[sid][question_id] = {
                'attempts': 0,
                'correct': 0
            }

        # Update question response tracking
        st.session_state.question_responses[sid][question_id]['attempts'] += 1
        if is_correct:
            st.session_state.question_responses[sid][question_id]['correct'] += 1

    except Exception as e:
        st.error(f"Error tracking question response: {str(e)}")
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
    """
    Get quiz recommendations based on student completion and quiz progress.

    Args:
        sid: Student ID
        completed: List of completed topics

    Returns:
        list: Quiz recommendations
    """
    rec = []

    # Ensure quiz_progress is initialized
    if 'quiz_progress' not in st.session_state:
        st.session_state.quiz_progress = {}
    if sid not in st.session_state.quiz_progress:
        st.session_state.quiz_progress[sid] = {}

    # Only recommend topics the student has completed
    for t in completed:
        if t in FORMULA_QUIZ_BANK:
            # Get current progress or default to 0
            p = st.session_state.quiz_progress.setdefault(sid, {}).get(t, 0)
            subs = list(FORMULA_QUIZ_BANK[t].keys())

            # Check if there are more subtopics available
            if p < len(subs):
                rec.append(f"📝 Quiz Alert: {subs[p]} ({t})")
            else:
                # All subtopics completed
                # Get mastery from graph
                if 'knowledge_graph' in st.session_state:
                    G = st.session_state.knowledge_graph
                    if G.has_node(t):
                        mastery = G.nodes[t].get('mastery', 0) * 100  # Convert to percentage
                        if mastery < 70:
                            # Recommend review if mastery is low
                            rec.append(f"📝 Review Alert: {t} (Mastery: {int(mastery)}%)")

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
        rec.insert(0, "🌟 Top Performer Strategies:")
        rec.extend([f"   → {r}" for r in topper_recs[:3]])
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

    # 4) Practice recommendations using knowledge graph relationships instead of just completed topics
    practice_candidates = []

    # First, add completed topics (direct practice)
    for t in comp[:3]:
        if t in PRACTICE_QUESTIONS:
            practice_candidates.append((t, 3.0, "completed"))  # Base priority for completed topics

    # Second, add topics with strong graph relationships
    for t in comp:
        # Get outgoing edges from this topic
        if G.has_node(t):
            for _, connected_topic, edge_data in G.out_edges(t, data=True):
                # Skip if already in practice_candidates or not in PRACTICE_QUESTIONS
                if connected_topic not in [c[0] for c in practice_candidates] and connected_topic in PRACTICE_QUESTIONS:
                    # Priority based on edge weight/relationship
                    priority = 1.0
                    if 'weight' in edge_data:
                        priority = min(edge_data['weight'], 5.0) / 2  # Scale to 0.5-2.5
                    relation = edge_data.get('relation', 'other')

                    # Boost priority based on relationship type
                    if relation in ['prereq', 'application']:
                        priority += 1.0
                    elif relation in ['subtopic']:
                        priority += 0.5

                    practice_candidates.append((connected_topic, priority, relation))

    # Sort by priority and recommend top 3
    practice_candidates.sort(key=lambda x: x[1], reverse=True)
    for t, priority, relation in practice_candidates[:3]:
        if t in PRACTICE_QUESTIONS:
            pq = PRACTICE_QUESTIONS[t]
            seq = pq.get('recent', []) + pq.get('historical', []) + pq.get('fundamental', [])
            relation_note = ""
            if relation != "completed":  # Not a directly completed topic
                relation_note = f" ({relation} to {t})"
            rec.append(f"📚 Practice {t}{relation_note}: {', '.join(seq[:3])}")
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

        # Enhanced data-driven subtopic recommendations
        if t in G.nodes():
            # Initialize containers for weighted subtopics
            sub_weights = {}

            # 1. Get direct subtopics from graph with weights
            direct_subs = [(v, d.get('weight', 1.0))
                           for _, v, d in G.out_edges(t, data=True)
                           if d.get('relation') == 'subtopic']

            for sub, weight in direct_subs:
                sub_weights[sub] = weight

            # 2. Incorporate question-level performance data
            if sid in st.session_state.question_responses:
                for qid, attempts in st.session_state.question_responses[sid].items():
                    # Find question details and associated subtopics
                    for topic, questions in QUESTION_BANK.items():
                        for q in questions:
                            if q['id'] == qid:
                                # Check if question is related to current topic
                                if topic == t:
                                    # Calculate performance on this question
                                    success_rate = sum(attempts) / len(attempts) if attempts else 0

                                    # If student struggles with this question
                                    if success_rate < 0.5:
                                        # Extract subtopics from question if available
                                        for i in range(1, 4):
                                            subtopic = q.get(f'subtopic{i}')
                                            if subtopic and subtopic in sub_weights:
                                                # Boost weight for struggled subtopics
                                                sub_weights[subtopic] += (1.0 - success_rate) * 2

            # 3. Apply centrality measures from the knowledge graph
            # Calculate betweenness centrality (how important nodes are as bridges)
            try:
                centrality = nx.betweenness_centrality(G, k=min(10, len(G.nodes)))
                for sub in sub_weights:
                    if sub in centrality:
                        # Boost weight based on centrality (importance in graph)
                        sub_weights[sub] += centrality[sub] * 5
            except:
                pass  # Skip if centrality calculation fails

            # 4. Sort and recommend subtopics based on final weights
            weighted_subs = sorted(sub_weights.items(), key=lambda x: x[1], reverse=True)

            if weighted_subs:
                # Format subtopic recommendations with weights
                formatted_subs = []
                for sub, weight in weighted_subs[:3]:
                    # Higher weights get more emphasis
                    emphasis = "🔥" if weight > 3 else "📌" if weight > 2 else ""
                    formatted_subs.append(f"{sub} {emphasis}")

                rec.append(f"🔧 Priority subtopics in {t}: {', '.join(formatted_subs)}")

                # 5. Add extra insight for highest weighted subtopic
                if weighted_subs and weighted_subs[0][1] > 2:
                    top_sub = weighted_subs[0][0]
                    # Find connected concepts to this subtopic
                    related = [v for _, v, d in G.out_edges(top_sub, data=True)]
                    if related:
                        rec.append(f"🔍 Master '{top_sub}' to unlock: {', '.join(related[:2])}")

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
def main():
    if 'knowledge_graph' not in st.session_state:
        st.session_state.knowledge_graph = nx.DiGraph()
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
                                if perf_gap >= 0.2:  # Changed to match function's default threshold of 0.2
                                    st.markdown(f"#### 🎯 Comparison with Student {best_peer}")
                                    st.caption(f"Similar start date, {perf_gap:.0%} performance difference")
                                    # Call progression_summary with the correct parameters
                                    insights = progression_summary(df, sid, best_peer)
                                    col1, col2 = st.columns([2, 3])
                                    with col1:
                                        st.markdown("##### 📈 Key Behavioral Differences")
                                        if len(insights) > 1:
                                            for insight in insights[1:4]:
                                                st.markdown(f"<div class='highlight'>{insight}</div>",
                                                            unsafe_allow_html=True)
                                        else:
                                            st.info("No significant behavioral differences found")
                                    with col2:
                                        if len(insights) >= 1:
                                            # Add topper resource tips
                                            topper_tips = recommend_topper_resources(seg, df)
                                            if topper_tips:
                                                st.markdown("**💡 Top Performer's Habits:**")
                                                for tip in topper_tips[:2]:
                                                    st.info(tip)
                                        st.markdown("##### 🚀 Improvement Plan")
                                        if len(insights) >= 1:
                                            try:
                                                # Display insights based on their prefixes
                                                recommendation = next(
                                                    (insight for insight in insights if "🚀" in insight), None)
                                                if recommendation:
                                                    st.markdown("**Priority Action:**")
                                                    st.success(f"{recommendation}")

                                                # Display other insights
                                                for insight in insights:
                                                    if "🚀" not in insight and "🏆" not in insight:
                                                        st.markdown(f"- {insight}")

                                                # Extract metrics and values for visualization
                                                metrics = []
                                                values = []
                                                for insight in insights:
                                                    if "📊" in insight:
                                                        parts = insight.split(":")
                                                        if len(parts) > 1:
                                                            metric = parts[0].replace("📊 ", "")
                                                            # Extract the numerical value
                                                            value_text = parts[1].strip()
                                                            if "by" in value_text:
                                                                try:
                                                                    value = int(value_text.split("by ")[1].split()[0])
                                                                    metrics.append(metric)
                                                                    values.append(value)
                                                                except (ValueError, IndexError):
                                                                    continue

                                                if metrics and values:
                                                    fig = go.Figure()
                                                    fig.add_trace(go.Bar(
                                                        x=metrics,
                                                        y=values,
                                                        text=values,
                                                        textposition='auto'
                                                    ))
                                                    fig.update_layout(
                                                        title="Key Activity Differences",
                                                        height=250
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)

                                            except Exception as e:
                                                st.error(f"Display error: {str(e)}")
                                                # Fallback to raw insights display
                                                st.markdown("**Key Findings:**")
                                                for insight in insights:
                                                    st.write(f"- {insight}")
                                            else:
                                                st.markdown("**General improvement tips:**")
                                                st.write("1. Balance different study activities")
                                                st.write("2. Review foundational concepts weekly")
                                                st.write("3. Track your time distribution")
                                else:
                                    st.info("No peers found with >=20% performance gap")  # Changed threshold message
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
                                        # Initialize response structures if not present
                                        if 'quiz_responses' not in st.session_state:
                                            st.session_state.quiz_responses = {}
                                        if 'question_responses' not in st.session_state:
                                            st.session_state.question_responses = {}

                                        # Initialize student-specific structures
                                        if sid not in st.session_state.quiz_responses:
                                            st.session_state.quiz_responses[sid] = {}
                                        if sid not in st.session_state.question_responses:
                                            st.session_state.question_responses[sid] = {}

                                        # Validate topic again (defense in depth)
                                        if topic not in FORMULA_QUIZ_BANK:
                                            st.error("Invalid quiz topic selected")
                                            return

                                        # Validate questions exist for topic
                                        if not FORMULA_QUIZ_BANK[topic]:
                                            st.error("No questions available for this topic")
                                            return

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

                                        # Process all responses with the new error handling
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

                                                # Safely get student answer
                                                student_answer = response['answer']

                                                # Validate answer
                                                is_correct, feedback = validate_answer(question, student_answer)

                                                # Track response
                                                track_question_response(sid, question['id'], is_correct)

                                                # Store detailed response
                                                if topic not in st.session_state.quiz_responses[sid]:
                                                    st.session_state.quiz_responses[sid][topic] = []

                                                st.session_state.quiz_responses[sid][topic].append({
                                                    "qid": question['id'],
                                                    "answer": student_answer,
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
                                        import traceback
                                        st.error(traceback.format_exc())

            except Exception as e:
                st.error(f"Quiz section error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
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
