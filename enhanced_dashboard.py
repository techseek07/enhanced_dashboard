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
    'Biology': ['Chemistry']
}
TOPICS = ['Algebra', 'Geometry', 'Calculus', 'Chemistry', 'Biology']
SUBTOPICS = {
    'Algebra': ['Equations', 'Inequalities', 'Polynomials'],
    'Geometry': ['Angles', 'Shapes', 'Trigonometry'],
    'Calculus': ['Limits', 'Derivatives', 'Integrals'],
    'Chemistry': ['Elements', 'Reactions', 'Compounds'],
    'Biology': ['Cells', 'Genetics', 'Ecology']
}

# Application-level edges
APPLICATION_RELATIONS = {
    'Biomolecules': {'base_topic': 'Chemical Bonding', 'type': 'application'},
    'Optimization': {'base_topic': 'Derivatives', 'type': 'application'}
}

# Quiz bank & progress tracking
FORMULA_QUIZ_BANK = {
    'Calculus': {
        'Derivative Rules': [
            {"question": "d/dx(sin x) = ?", "type": "formula"},
            {"question": "Product rule formula?", "type": "definition"}
        ],
        'Integration Formulas': [
            {"question": "∫e^x dx = ?", "type": "formula"},
            {"question": "Integration by parts formula?", "type": "definition"}
        ]
    }
}
if "quiz_progress" not in st.session_state:
    st.session_state.quiz_progress = {}
QUESTION_BANK = {
    'Algebra': [
        {"id": "alg_1", "text": "Solve for x: 2x + 5 = 13", "difficulty": 1},
        {"id": "alg_2", "text": "Factor: x² - 9", "difficulty": 2},
        {"id": "alg_3", "text": "Solve the system: 2x + y = 7, x - y = 1", "difficulty": 3}
    ],
    'Geometry': [
        {"id": "geo_1", "text": "Find the area of a circle with radius 5", "difficulty": 1},
        {"id": "geo_2", "text": "Prove triangles ABC and DEF are similar", "difficulty": 3},
        {"id": "geo_3", "text": "Calculate the volume of a cone with height 6 and radius 2", "difficulty": 2}
    ],
    'Calculus': [
        {"id": "calc_1", "text": "Find the derivative of f(x) = x³ + 2x²", "difficulty": 2},
        {"id": "calc_2", "text": "Evaluate ∫(2x + 1)dx from 0 to 3", "difficulty": 2},
        {"id": "calc_3", "text": "Find the inflection points of f(x) = x³ - 6x²", "difficulty": 3}
    ],
    'Chemistry': [
        {"id": "chem_1", "text": "Balance: H₂ + O₂ → H₂O", "difficulty": 1},
        {"id": "chem_2", "text": "Calculate pH of 0.01M HCl solution", "difficulty": 2},
        {"id": "chem_3", "text": "Draw the Lewis structure for CO₂", "difficulty": 2}
    ],
    'Biology': [
        {"id": "bio_1", "text": "List the phases of mitosis in order", "difficulty": 1},
        {"id": "bio_2", "text": "Explain how DNA replication works", "difficulty": 2},
        {"id": "bio_3", "text": "Calculate Hardy-Weinberg equilibrium for a population", "difficulty": 3}
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
    # Add 3 hardwired students first
    hardwired_students = [
        # Student 0 (Topper)
        {'skill': 0.95, 'mot': 'High', 'correct_rate': 0.9, 'time_factor': 0.7},
        # Student 1 (Average)
        {'skill': 0.65, 'mot': 'Medium', 'correct_rate': 0.6, 'time_factor': 1.0},
        # Student 2 (Poor)
        {'skill': 0.3, 'mot': 'Low', 'correct_rate': 0.4, 'time_factor': 1.5}
    ]

    for sid in range(3):  # First 3 students are special
        hw = hardwired_students[sid]
        exam_date = datetime.now() - timedelta(days=30)  # Fixed start date

        # Force completion of core topics
        done = ['Algebra', 'Geometry'] if sid == 0 else ['Algebra']

        # Generate interactions for all topics
        for topic in TOPICS:
            for _ in range(5):  # Fixed interaction count
                subtopics = SUBTOPICS.get(topic, [])
                subs = (
                    np.random.choice(subtopics, 3, replace=False)
                    if subtopics
                    else [None] * 3
                )

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

    # Validate and ensure minimum student count
    num_students = max(1, num_students)  # Ensure at least 1 student

    for sid in range(num_students):
        # Safe date generation
        exam_date = datetime.now() + timedelta(days=np.random.randint(7, 60))

        # Skill generation with beta distribution safeguards
        skill = np.clip(np.random.beta(2, 1.5), 0.01, 0.99)  # Prevent 0 or 1

        # Motivation level with probability validation
        mot = np.random.choice(['High', 'Medium', 'Low'],
                               p=[0.5, 0.3, 0.2])

        # Completed topics with size validation
        done = np.random.choice(
            TOPICS,
            size=np.random.randint(0, len(TOPICS) + 1),
            replace=False
        ) if TOPICS else []

        for topic in TOPICS:
            # Safe Poisson distribution for interaction count
            interaction_count = np.random.poisson(3) + 1  # Ensures ≥1 interaction
            interaction_count = max(1, min(interaction_count, 10))  # Cap at 10

            for _ in range(interaction_count):
                # Subtopic handling with fallbacks
                subtopics = SUBTOPICS.get(topic, [])
                subs = (
                    np.random.choice(subtopics, 3, replace=len(subtopics) < 3)
                    if subtopics
                    else [None] * 3
                )

                # Dirichlet distribution with normalization
                wts = np.random.dirichlet(np.ones(3))
                wts /= wts.sum()  # Ensure proper normalization

                # Motivation factor with bounds
                mf = {'High': 1.0, 'Medium': 0.9, 'Low': 0.7}.get(mot, 0.7)

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

                # Quiz progress with empty bank handling
                quiz_bank = FORMULA_QUIZ_BANK.get(topic, {})
                qp_max = max(1, len(quiz_bank))  # Ensure ≥1 to prevent randint error
                qp = np.random.randint(0, qp_max)

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

    # Handle single-point edge case
    if len(perf) == 1:
        perf['cluster'] = 0
        perf['label'] = 'Average'
        return perf

    try:
        perf['cluster'] = kmeans.fit_predict(perf[['acc', 'time']])
    except Exception as e:
        perf['cluster'] = 0
        # MANUALLY OVERRIDE DEMO STUDENTS AFTER CLUSTERING
    if not perf.empty:
        # Assign clusters based on performance
        perf.loc[perf.StudentID == 0, 'cluster'] = 0  # Topper cluster
        perf.loc[perf.StudentID == 1, 'cluster'] = 1  # Average cluster
        perf.loc[perf.StudentID == 2, 'cluster'] = 2  # Poor cluster

        # DIRECT LABEL MAPPING FOR DEMO STUDENTS
    label_map = {
        0: 'Topper',
        1: 'Average',
        2: 'Poor'
    }
    perf['label'] = perf['cluster'].map(label_map)

    # Validation of cluster statistics
    cluster_stats = perf.groupby('cluster')['acc'].agg(['mean', 'count'])
    valid_clusters = cluster_stats[cluster_stats['mean'] > 0]  # Filter zero-acc clusters

    if valid_clusters.empty:
        perf['label'] = 'Needs Help'
        return perf

    # Stable sorting with secondary key
    ranked_clusters = valid_clusters.sort_values(
        ['mean', 'count'],
        ascending=[False, False]
    ).reset_index()

    # Dynamic labeling
    label_map = {}
    for i, row in ranked_clusters.iterrows():
        label = ['Topper', 'Average', 'Poor'][i] if i < 3 else 'Unknown'
        label_map[row['cluster']] = label

    perf['label'] = perf['cluster'].map(label_map).fillna('Unknown')
    return perf

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
    if abs((s1_start - s2_start).days) > 15:
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

    # Initialize application nodes
    for app_key, app_info in application_relations.items():
        G.add_node(app_key, type='application')
        G.add_node(app_info['base_topic'], type='base_topic')

    # Add predefined relationships
    for target, requirements in prereqs.items():
        for req in requirements:
            G.add_edge(req, target, relation='prereq', weight=3.0)

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
            rec.append(f"🎥 {t}: {', '.join(m['videos'][:1])}")
    # then, analogies for all “hard” topics
    for t in hard_topics:
        m = MEDIA_LINKS.get(t, {})
        if m.get('analogies'):
            rec.append(f"🔗 Analogy for {t}: {m['analogies']}")

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

    # Get student's responses
    responses = st.session_state.question_responses[sid]

    # Find questions with high failure rates (>50% failure)
    problem_questions = []
    for q_id, attempts in responses.items():
        if len(attempts) >= 2:  # Only consider questions with multiple attempts
            success_rate = sum(attempts) / len(attempts)
            if success_rate < 0.5:  # More than 50% failure
                # Find the question text
                q_text = None
                for topic, questions in QUESTION_BANK.items():
                    for q in questions:
                        if q['id'] == q_id:
                            q_text = q['text']
                            problem_questions.append((q_id, q_text, topic, success_rate))
                            break
                    if q_text:
                        break

    return problem_questions


# ==================================================================
# 8. Streamlit UI
# ==================================================================
# ==================================================================
# 8. Streamlit UI
# ==================================================================
def main():
    st.set_page_config(layout="wide", page_title="Learning Dashboard", page_icon="🧠")

    # Add CSS for better styling
    st.markdown("""
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
            " Motivation Level",
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

                    # Create traces
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
                                go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=col, width=2), name=rel))

                    # Node trace
                    node_x = [pos[n][0] for n in G.nodes if n in pos]
                    node_y = [pos[n][1] for n in G.nodes if n in pos]
                    node_text = [n for n in G.nodes if n in pos]

                    # Node colors
                    acc_by_topic = df[df.StudentID == sid].groupby('Topic').Correct.mean()
                    node_colors = []
                    for node in [n for n in G.nodes if n in pos]:
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
                        "📝 Quiz": [], "🎥 Media": [], "🔗 Analogy": [],
                        "📊 Formula": [], "🔧 Subtopics": [], "🔄 Apply": [],
                        "👍 Easy Win": [], "💭 Quote": []
                    }

                    for rec in recommendations:
                        for prefix in rec_types:
                            if rec.startswith(prefix):
                                rec_types[prefix].append(rec)
                                break

                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown("### Study Plan")
                        for rec in rec_types["🚧 Bridge"] + rec_types["📚 Practice"] + rec_types["📊 Formula"]:
                            st.info(rec)
                    with cols[1]:
                        st.markdown("### Challenges")
                        for rec in rec_types["🧠 HOTS"] + rec_types["📝 Quiz"] + rec_types["🔧 Subtopics"]:
                            st.success(rec)
                    with cols[2]:
                        st.markdown("### Engagement")
                        for rec in rec_types["💭 Quote"]:
                            st.markdown(
                                f'<div style="background-color:#f0f7fb; padding:10px; border-radius:5px;">{rec}</div>',
                                unsafe_allow_html=True)
                        for rec in rec_types["🎥 Media"] + rec_types["🔗 Analogy"] + rec_types["🔄 Apply"] + rec_types[
                            "👍 Easy Win"]:
                            st.warning(rec)
                except Exception as e:
                    st.error(f"Recommendation error: {str(e)}")


        # Strategic Peer Comparison
        with st.expander("🔍 Strategic Peer Comparison", expanded=True):
            if not df.empty and sid is not None:
                try:
                    # Get current student's start date
                    current_data = df[df.StudentID == sid]
                    current_start = current_data.ExamDate.min()

                    # Find comparable peers (started within 30 days)
                    comparable_peers = df[
                        (df.ExamDate > current_start - pd.Timedelta(days=30)) &
                        (df.ExamDate < current_start + pd.Timedelta(days=30)) &
                        (df.StudentID != sid)
                        ].StudentID.unique()

                    if len(comparable_peers) > 0:
                        # Calculate performance differences
                        peer_perf = df[df.StudentID.isin(comparable_peers)].groupby('StudentID').Correct.mean()
                        current_perf = current_data.Correct.mean()
                        peer_diff = abs(peer_perf - current_perf)

                        if not peer_diff.empty:
                            best_peer = peer_diff.idxmax()
                            perf_gap = peer_diff.max()

                            if perf_gap >= 0.2:  # Minimum 20% performance gap
                                st.markdown(f"#### 🎯 Comparison with Student {best_peer}")
                                st.caption(f"Similar start date, {perf_gap:.0%} performance difference")

                                # Get comparison insights
                                insights = progression_summary(df, sid, best_peer)

                                # Layout
                                col1, col2 = st.columns([2, 3])

                                with col1:
                                    st.markdown("##### 📈 Key Behavioral Differences")
                                    if len(insights) > 1:
                                        for insight in insights[1:4]:
                                            st.markdown(f"""
                                            <div class="metric-box" style="margin-bottom:10px;">
                                                {insight.replace(':', '<br>').replace('(', '<br>')}
                                            </div>
                                            """, unsafe_allow_html=True)
                                    else:
                                        st.info("No significant behavioral differences found")

                                with col2:
                                    st.markdown("##### 🚀 Improvement Plan")
                                    if len(insights) >= 4:
                                        # Visual comparison for top metric
                                        top_metric = insights[1].split(':')[0]
                                        s1_val = int(insights[1].split(' ')[-3])
                                        s2_val = int(insights[1].split(' ')[-6])

                                        fig = go.Figure()
                                        fig.add_trace(go.Bar(
                                            x=['You', f'Student {best_peer}'],
                                            y=[s1_val, s2_val],
                                            marker=dict(color=['#FFA07A', '#20B2AA']),
                                            text=[s1_val, s2_val],
                                            textposition='auto'
                                        ))
                                        fig.update_layout(
                                            title=f"{top_metric} Comparison",
                                            height=300,
                                            margin=dict(t=40, b=20)
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                                        # Action items
                                        st.markdown(f"""
                                        <div style="background-color:#E8F4FC; padding:15px; border-radius:10px;">
                                            <h4 style="color:#2C5F8A;">Recommended Actions</h4>
                                            <ul style="color:#2C5F8A;">
                                                <li>{insights[-1].replace('Recommendation:', '')}</li>
                                                <li>Review {top_metric.lower()} resources weekly</li>
                                                <li>Schedule 2 extra practice sessions</li>
                                            </ul>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.warning("Insufficient data for detailed comparison")
                            else:
                                st.info("No peers found with >20% performance gap")
                        else:
                            st.warning("Could not calculate peer differences")
                    else:
                        st.warning("⚠️ No comparable peers found with similar start dates")
                except Exception as e:
                    st.error(f"Comparison failed: {str(e)}")
            else:
                st.warning("Select a student to enable peer comparison")

        # Quiz Section
        try:
            st.markdown('<p class="medium-font">Interactive Quiz Section</p>', unsafe_allow_html=True)
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
                                    st.markdown(f"**Q{j + 1}:** {q['question']}")
                                    if q['type'] == 'formula':
                                        st.text_input("Your answer:", key=q_key)
                                    else:
                                        st.radio("Select:", ["Option A", "Option B", "Option C"], key=q_key)
                                    st.markdown("---")
                            if st.form_submit_button("Submit Quiz"):
                                # bump the progress counter
                                st.session_state.quiz_progress.setdefault(sid, {})[topic] = (
                                        st.session_state.quiz_progress.get(sid, {}).get(topic, 0) + 1
                                )

                                # ── Persist to session & global history ──
                                st.session_state.setdefault('quiz_responses', {}) \
                                    .setdefault(sid, {})[topic] = st.session_state.quiz_answers
                            st.session_state.question_responses.update(st.session_state['quiz_responses'][sid])

                            st.success("Quiz submitted successfully!")
            else:
                st.info("Complete some topics to unlock quizzes!")
        except Exception as e:
            st.error(f"Quiz error: {str(e)}")

        # Performance Analytics
        try:
            st.markdown('<p class="medium-font">Performance Analytics</p>', unsafe_allow_html=True)
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
                            # Fix: Map the correct column names to their display names
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
        try:
            st.markdown('<p class="medium-font">Question-Level Analytics</p>', unsafe_allow_html=True)
            if sid is not None:
                problem_questions = analyze_item_level_performance(sid)
                if problem_questions:
                    st.warning("🔍 Questions you're struggling with:")
                    for q_id, q_text, topic, success_rate in problem_questions:
                        with st.expander(f"{topic}: {q_text} (Success: {success_rate:.0%})"):
                            st.write("**Suggested micro-lesson:**")
                            if topic == "Algebra":
                                st.write(
                                    "- Break the problem into steps\n- Check variable isolation\n- Substitute values")
                            elif topic == "Geometry":
                                st.write("- Draw diagrams\n- Identify key formulas\n- Look for patterns")
                            elif topic == "Calculus":
                                st.write("- Review rules\n- Simplify functions\n- Check algebra")
                            else:
                                st.write("- Review concepts\n- Create visuals\n- Practice basics")

                            if st.button(f"Generate similar question for {q_id}"):
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
