# enhanced_dashboard_improved.py
from statsmodels.stats.contingency_tables import StratifiedTable
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import shap
from itertools import combinations
from collections import Counter
from datetime import datetime, timedelta

# ==================================================================
# 0. Configuration & Mock Data
# ==================================================================
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
QUIZ_PROGRESS = {}
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
QUESTION_RESPONSES = {}

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
FORMULA_QUIZ_TOPICS = {
    'Algebra': ['Quadratic Formula', 'Factoring Patterns', 'Exponent Rules'],
    'Geometry': ['Area/Volume Formulas', 'Trigonometric Identities', 'Circle Theorems'],
    'Calculus': ['Derivative Rules', 'Integration Formulas', 'Series Tests'],
    'Chemistry': ['Gas Laws', 'Equilibrium Constants', 'pH Calculations'],
    'Biology': ['Hardy-Weinberg', 'Population Growth', 'Genetic Inheritance']
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
# 1. Data Generation (with usage & quiz‑progress)
# ==================================================================
@st.cache_data
def generate_student_data(num_students=500):
    np.random.seed(42)
    rows = []
    for sid in range(num_students):
        exam_date = datetime.now() + timedelta(days=np.random.randint(7, 60))
        skill = np.random.beta(2, 1.5)
        mot = np.random.choice(['High', 'Medium', 'Low'], p=[0.5, 0.3, 0.2])
        done = np.random.choice(TOPICS, size=np.random.randint(0, len(TOPICS) + 1), replace=False)
        for topic in TOPICS:
            for _ in range(np.random.poisson(3) + 1):
                subtopics = SUBTOPICS.get(topic, [])
                if subtopics:
                    subs = np.random.choice(subtopics, 3, replace=len(subtopics) < 3)
                else:
                    subs = [None] * 3

                wts = np.random.dirichlet(np.ones(3))
                mf = {'High': 1.0, 'Medium': 0.9, 'Low': 0.7}[mot]
                corr = np.random.binomial(1, skill * mf)
                tkt = np.random.gamma(2, 1.5) * (2 - mf)

                # usage metrics
                vids = np.random.poisson(1)
                qz = np.random.binomial(1, 0.3)
                prac = np.random.poisson(2)
                media = np.random.poisson(1)
                qp = np.random.randint(0, len(FORMULA_QUIZ_BANK.get(topic, {})))

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
    perf = df.groupby('StudentID').agg(acc=('Correct', 'mean'), time=('TimeTaken', 'mean')).reset_index()

    # Handle edge case of empty dataframe
    if perf.empty:
        return pd.DataFrame(columns=['StudentID', 'acc', 'time', 'cluster', 'label'])

    # Ensure we have at least 3 students for KMeans
    if len(perf) < 3:
        perf['cluster'] = 0
        perf['label'] = 'Average'
        return perf

    perf['cluster'] = KMeans(n_clusters=min(3, len(perf)), random_state=0).fit_predict(perf[['acc', 'time']])

    # Determine order based on accuracy
    cluster_stats = perf.groupby('cluster')['acc'].mean()
    if len(cluster_stats) == 3:
        order = cluster_stats.sort_values(ascending=False).index
        perf['label'] = perf['cluster'].map({order[0]: 'Topper', order[1]: 'Average', order[2]: 'Poor'})
    else:
        # Handle case with fewer clusters
        if len(cluster_stats) == 2:
            order = cluster_stats.sort_values(ascending=False).index
            perf['label'] = perf['cluster'].map({order[0]: 'Topper', order[1]: 'Poor'})
        else:
            perf['label'] = 'Average'

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


def progression_summary(df, high_id, low_id):
    if df.empty or high_id == low_id:
        return ["Select different students to compare progression."]

    # Ensure both IDs exist in the data
    if high_id not in df.StudentID.unique() or low_id not in df.StudentID.unique():
        return ["One or both selected students not found in data."]

    cutoff = df.ExamDate.min() + timedelta(days=180)
    sub = df[df.ExamDate <= cutoff]

    # Get high performer data
    high_data = sub[sub.StudentID == high_id]
    if high_data.empty:
        return [f"Student {high_id} has no data within the timeframe."]

    # Get low performer data
    low_data = sub[sub.StudentID == low_id]
    if low_data.empty:
        return [f"Student {low_id} has no data within the timeframe."]

    a = high_data.sum(numeric_only=True)
    b = low_data.sum(numeric_only=True)

    # Calculate differences and sort
    diff = (a - b).sort_values(ascending=False)

    # Filter for relevant metrics only
    relevant_metrics = ['VideosWatched', 'QuizzesTaken', 'PracticeSessions', 'MediaClicks']
    diff = diff[diff.index.isin(relevant_metrics)]

    insights = [f"In 6m, topper did {int(diff[i])} more {i}" for i in diff.index if diff[i] > 0]

    # Add fallback message if no significant differences
    if not insights:
        insights = ["No significant difference in learning activities between the students."]

    return insights


# ==================================================================
# 3. Knowledge Graph Construction
# ==================================================================
def build_knowledge_graph(prereqs, df, topics, OR_thresh=2.0, SHAP_thresh=0.01,
                          min_count=20, application_relations=APPLICATION_RELATIONS):
    """
    Constructs a knowledge graph combining prerequisite relationships, application connections,
    and statistically validated topic relationships from student performance data.

    Parameters:
    - prereqs (dict): Prerequisite relationships {topic: [prerequisites]}
    - df (DataFrame): Student interaction data
    - topics (list): All topics to include
    - OR_thresh (float): Odds ratio threshold for edge creation
    - SHAP_thresh (float): SHAP importance threshold
    - min_count (int): Minimum student count for analysis
    - application_relations (dict): Application topic definitions

    Returns:
    - DiGraph: Constructed knowledge graph
    """
    G = nx.DiGraph()

    # Initialize all node types
    for topic in topics:
        G.add_node(topic, type='topic')
    for app_info in application_relations.values():
        G.add_node(app_info['base_topic'], type='base_topic')
        G.add_node(app_info['app_node'], type='application')

    # Add predefined relationships
    for target, requirements in prereqs.items():
        for req in requirements:
            G.add_edge(req, target, relation='prereq', weight=3.0)

    for app_key, app_info in application_relations.items():
        G.add_edge(app_info['base_topic'], app_info['app_node'],
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
                    strata = pd.qcut(X.proficiency, q=3, duplicates='drop')
                    if len(strata.unique()) >= 2:
                        table = StratifiedTable.from_data(X.topic_a, y, strata)
                        or_value = table.oddsratio_pooled

                if or_value > OR_thresh and not np.isnan(or_value):
                    G.add_edge(topic_a, topic_b, relation='odds_ratio',
                               weight=min(or_value, 5.0))

                # SHAP feature importance
                if len(y) >= 100:
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
                st.error(f"Relationship analysis failed for {topic_a}-{topic_b}: {str(e)}")
                continue

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
                subtopic = row[f'Subtopic{i}']
                weight = row[f'Weight{i}']
                if pd.notna(subtopic) and pd.notna(weight):
                    subtopic_weights[subtopic] += weight

        for subtopic, _ in subtopic_weights.most_common(2):
            if subtopic not in G.nodes:
                G.add_node(subtopic, type='subtopic')

            G.add_edge(topic, subtopic, relation='subtopic',
                       weight=min(subtopic_weights[subtopic], 5.0))

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


def apply_dual_tier_scoring(G):
    """
    Applies tiered scoring to edges based on relationship type and node connectivity.
    Maintains original tier logic as it effectively captures edge importance.
    """
    for source, target, data in G.edges(data=True):
        # Base score from relationship type
        base_scores = {
            'prereq': 3,
            'application': 2,
            'odds_ratio': 2,
            'shap_importance': 1.5,
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
# 4. Quiz Recommendation
# ==================================================================
def get_quiz_recommendations(sid, completed):
    rec = []

    # Only recommend topics the student has completed
    for t in completed:
        if t in FORMULA_QUIZ_BANK:
            # Get current progress or default to 0
            p = QUIZ_PROGRESS.setdefault(sid, {}).get(t, 0)
            subs = list(FORMULA_QUIZ_BANK[t].keys())

            # Recommend next subtopic if available
            if p < len(subs):
                rec.append(f"📝 Quiz Alert: {subs[p]} ({t})")

    return rec


# ==================================================================
# 5. Comprehensive Recommendations
# ==================================================================
def get_recommendations(sid, df, G,seg, mot='High'):
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

    # Add motivation quote based on student's topics
    if comp:
        selected_topic = np.random.choice(comp)
        if selected_topic in MOTIVATION_QUOTES:
            rec.append(f"💭 \"{MOTIVATION_QUOTES[selected_topic]}\"")

    # Bridge course recommendations for struggling topics
    low_topics = acc[acc < 0.3].index.tolist()
    for t in low_topics:
        if t in BRIDGE_COURSES:
            rec.append(f"🚧 Bridge: {t} - {BRIDGE_COURSES[t]}")
        else:
            rec.append(f"🚧 Bridge: {t}")

    # HOTS (Higher Order Thinking Skills) for high-performing topics
    if mot != 'Low':
        high_topics = acc[acc > 0.7].index.tolist()
        for t in high_topics[:2]:  # Limit to 2 topics to avoid overwhelming
            if t in HOTS_QUESTIONS:
                rec.append(f"🧠 HOTS {t}: {', '.join(HOTS_QUESTIONS[t][:2])}")

    # Practice recommendations
    for t in comp[:3]:  # Limit to 3 topics
        if t in PRACTICE_QUESTIONS:
            pq = PRACTICE_QUESTIONS[t]
            seq = pq.get('recent', []) + pq.get('historical', []) + pq.get('fundamental', [])
            rec.append(f"📚 Practice {t}: {', '.join(seq[:3])}")

    # Quiz recommendations
    quiz_recs = get_quiz_recommendations(sid, comp)
    rec.extend(quiz_recs[:2])  # Limit to 2 quiz recommendations

    # Media/Analogy recommendations - provide analogies for all difficult topics
    hard_topics = acc[acc < 0.5].index.tolist()

    # First add videos for completed topics
    for t in comp[:2]:  # Limit to 2 topics
        m = MEDIA_LINKS.get(t, {})
        if m.get('videos'):
            rec.append(f"🎥 {t}: {', '.join(m['videos'][:1])}")

    # Always provide analogies for hard topics regardless of motivation
    for t in hard_topics:
        m = MEDIA_LINKS.get(t, {})
        if m.get('analogies'):
            rec.append(f"🔗 Analogy for {t}: {m['analogies']}")
    # Formula revision materials for low-performing topics
    for t in low_topics[:2]:
        if t in FORMULA_REVISION:
            rec.append(f"📊 Formula Help {t}: {', '.join(FORMULA_REVISION[t][:2])}")

    # Easy topics for motivation
    if mot == 'Low' and comp:
        easy_topic = np.random.choice(comp)
        if easy_topic in EASY_TOPICS:
            rec.append(f"👍 Easy Win: {easy_topic} - {EASY_TOPICS[easy_topic][0]}")

        # Subtopic & application recommendations from knowledge graph
    for t in comp:  # Check all completed topics for applications
        # Get subtopics that need focus
        if t in low_topics:
            subs = [v for u, v, d in G.out_edges(t, data=True)
                    if d.get('relation') == 'topic_sub_top']
            if subs:
                rec.append(f"🔧 Subtopics to review in {t}: {', '.join(subs[:3])}")
        # Add collaborative filtering recommendations
        collab_recs = collaborative_filtering_recommendations(sid, df, seg)
        rec.extend(collab_recs)

        # For all completed topics, find applications and future topics
        # Find topics that this one is prerequisite for
        future_topics = [v for u, v, d in G.out_edges(t, data=True)
                         if d.get('relation') == 'prereq' or d.get('relation') == 'odds']

        # Find direct applications
        apps = [v for u, v, d in G.out_edges(t, data=True)
                if d.get('relation') == 'app_connection' or d.get('relation') == 'application']

        if future_topics:
            rec.append(f"🔄 Apply {t} in: {', '.join(future_topics[:2])}")

        if apps:
            rec.append(f"🔬 Real applications of {t}: {', '.join(apps[:2])}")


def analyze_item_level_performance(sid):
    if sid not in QUESTION_RESPONSES:
        return []

    # Get student's responses
    responses = QUESTION_RESPONSES[sid]

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
# 6. Streamlit UI (with quizzes)
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

    st.markdown('<p class="big-font">Enhanced Learning Dashboard 🧠</p>', unsafe_allow_html=True)

    try:
        # Generate student data
        with st.spinner("Loading student data..."):
            df = generate_student_data()

        # Build knowledge graph
        with st.spinner("Building knowledge graph..."):
            G = apply_dual_tier_scoring(build_knowledge_graph(PREREQUISITES, df, TOPICS))

        # Segment students
        with st.spinner("Analyzing student segments..."):
            seg = segment_students(df)

        # Sidebar settings
        st.sidebar.markdown('<p class="medium-font">Settings</p>', unsafe_allow_html=True)

        # Student selection with search
        all_students = sorted(df.StudentID.unique())
        sid = st.sidebar.selectbox("Select Student", all_students)

        # Display student tier
        student_info = seg[seg.StudentID == sid]
        if not student_info.empty:
            tier = student_info['label'].iloc[0]
            tier_colors = {'Topper': 'green', 'Average': 'blue', 'Poor': 'red'}
            tier_color = tier_colors.get(tier, 'gray')
            st.sidebar.markdown(f'<div style="background-color:{tier_color}20; padding:10px; border-radius:5px;">'
                                f'<strong>Performance Tier:</strong> {tier}</div>',
                                unsafe_allow_html=True)

        # Show recommendations from top performers for Average/Poor students
        if tier in ('Average', 'Poor'):
            st.sidebar.markdown('<p class="medium-font">Insights from Top Performers</p>',
                                unsafe_allow_html=True)

            topper_tips = recommend_topper_resources(seg, df)
            for tip in topper_tips:
                st.sidebar.info(tip)

        # 6-month progression comparison
        st.sidebar.markdown('<p class="medium-font">6-Month Progression</p>',
                            unsafe_allow_html=True)

        other_students = [s for s in all_students if s != sid]
        if other_students:
            benchmark_student = st.sidebar.selectbox("Compare with", other_students)

            if st.sidebar.button("Show Comparison"):
                with st.sidebar.expander("Progression Tips", expanded=True):
                    for tip in progression_summary(df, benchmark_student, sid):
                        st.write(f"• {tip}")

        # Motivation level selector
        default_mot = 'Medium'
        if not student_info.empty and student_info['label'].iloc[0] in ['High', 'Medium', 'Low']:
            default_mot = student_info['label'].iloc[0]

        st.sidebar.markdown('<p class="medium-font">How are you feeling today?</p>', unsafe_allow_html=True)
        mot_emoji = {'High': '😃 Energetic', 'Medium': '😐 Neutral', 'Low': '😔 Unmotivated'}
        mot = st.sidebar.radio(
            "Current Motivation",
            ['High', 'Medium', 'Low'],
            format_func=lambda x: mot_emoji[x],
            index=['High', 'Medium', 'Low'].index(default_mot)
        )
        st.sidebar.markdown(f"<div style='padding:10px; background-color:#f0f2f6; border-radius:5px;'>"
                            f"Selected: <b>{mot_emoji[mot]}</b><br>"
                            f"Recommendations will adapt to your current mood.</div>",
                            unsafe_allow_html=True)
        # Main content area
        col1, col2 = st.columns([1, 2])

        # Knowledge graph visualization
        with col1:
            st.markdown('<p class="medium-font">Knowledge Graph</p>', unsafe_allow_html=True)

            with st.container(height=400, border=True):
                # Create graph layout
                try:
                    pos = nx.spring_layout(G, seed=42)

                    # Create edge traces by relation type
                    edge_colors = {
                        'prereq': '#FF0000',  # Red
                        'odds': '#00FF00',  # Green
                        'xgb_shap': '#0000FF',  # Blue
                        'application': '#800080',  # Purple
                        'topic_sub_top': '#FFA500',  # Orange
                        'sub_prereq': '#FF69B4',  # Pink
                        'app_connection': '#008080'  # Teal
                    }

                    traces = []

                    # Create a legend
                    st.caption("Legend:")
                    legend_cols = st.columns(3)
                    for i, (rel, col) in enumerate(edge_colors.items()):
                        legend_cols[i % 3].markdown(f'<span style="color:{col}">■</span> {rel}',
                                                    unsafe_allow_html=True)

                    # Create traces for each edge type
                    for rel, col in edge_colors.items():
                        es = [(u, v) for u, v, d in G.edges(data=True) if d.get('relation') == rel]
                        if not es:
                            continue

                        xs, ys = [], []
                        for u, v in es:
                            if u in pos and v in pos:  # Make sure nodes exist in layout
                                xs += [pos[u][0], pos[v][0], None]
                                ys += [pos[u][1], pos[v][1], None]

                        if xs and ys:  # Only add traces if there are edges
                            traces.append(go.Scatter(
                                x=xs, y=ys,
                                mode='lines',
                                line=dict(color=col, width=2),
                                name=rel
                            ))

                    # Create node trace
                    node_x = [pos[n][0] for n in G.nodes() if n in pos]
                    node_y = [pos[n][1] for n in G.nodes() if n in pos]
                    node_text = [n for n in G.nodes() if n in pos]

                    # Color nodes based on student performance
                    acc_by_topic = df[df.StudentID == sid].groupby('Topic').Correct.mean()

                    # Default color for nodes not in student data
                    node_colors = []
                    for node in [n for n in G.nodes() if n in pos]:
                        if node in acc_by_topic:
                            # Color based on performance (red to green)
                            perf = acc_by_topic[node]
                            if np.isnan(perf):
                                node_colors.append('#FFA500')  # Orange for no data
                            elif perf < 0.3:
                                node_colors.append('#FF0000')  # Red for low performance
                            elif perf < 0.7:
                                node_colors.append('#FFFF00')  # Yellow for medium
                            else:
                                node_colors.append('#00FF00')  # Green for high
                        else:
                            node_colors.append('#FFA500')  # Orange for no data

                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition="top center",
                        marker=dict(
                            size=15,
                            color=node_colors,
                            line=dict(width=1, color='#000000')
                        ),
                        name='Topics'
                    )

                    # Create and display figure
                    fig = go.Figure(
                        data=traces + [node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=350
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Topic colors explanation
                    st.caption("Topic colors: 🔴 Needs work | 🟡 Average | 🟢 Strong | 🟠 No data")

                except Exception as e:
                    st.error(f"Error displaying graph: {str(e)}")
                    st.markdown("Try selecting a different student or refreshing the page.")

        # Recommendations panel
        with col2:
            st.markdown('<p class="medium-font">Personalized Learning Recommendations</p>',
                        unsafe_allow_html=True)

            # Get recommendations
            recommendations = get_recommendations(sid, df, G,seg, mot)

            # Group recommendations by type
            rec_types = {
                "🚧 Bridge": [],
                "🧠 HOTS": [],
                "📚 Practice": [],
                "📝 Quiz": [],
                "🎥 Media": [],
                "🔗 Analogy": [],
                "📊 Formula": [],
                "🔧 Subtopics": [],
                "🔄 Apply": [],
                "👍 Easy Win": [],
                "💭 Quote": []
            }

            for rec in recommendations:
                for prefix in rec_types:
                    if rec.startswith(prefix):
                        rec_types[prefix].append(rec)
                        break

            # Display recommendations in three columns
            col1, col2, col3 = st.columns(3)

            # Study plan column
            with col1:
                st.markdown("### Study Plan")
                for rec in rec_types["🚧 Bridge"] + rec_types["📚 Practice"] + rec_types["📊 Formula"]:
                    st.info(rec)

            # Challenge column
            with col2:
                st.markdown("### Challenges")
                for rec in rec_types["🧠 HOTS"] + rec_types["📝 Quiz"] + rec_types["🔧 Subtopics"]:
                    st.success(rec)

            # Engagement column
            with col3:
                st.markdown("### Engagement")
                # Show motivation quote first if available
                for rec in rec_types["💭 Quote"]:
                    st.markdown(f'<div style="background-color:#f0f7fb; padding:10px; border-radius:5px;">{rec}</div>',
                                unsafe_allow_html=True)

                for rec in rec_types["🎥 Media"] + rec_types["🔗 Analogy"] + rec_types["🔄 Apply"] + rec_types[
                    "👍 Easy Win"]:
                    st.warning(rec)

        # Quiz section
        st.markdown('<p class="medium-font">Interactive Quiz Section</p>', unsafe_allow_html=True)

        # Get completed topics with quizzes
        comp = df[(df.StudentID == sid) & (df.Completed)].Topic.unique().tolist()
        quiz_topics = [t for t in comp if t in FORMULA_QUIZ_BANK]

        if quiz_topics:
            col1, col2 = st.columns([1, 2])

            with col1:
                selected_topic = st.selectbox("Select Quiz Topic", quiz_topics)

                if st.button("Start Quiz", type="primary"):
                    st.session_state.show_quiz = True
                    st.session_state.quiz_topic = selected_topic
                    st.session_state.quiz_answers = {}

            with col2:
                if hasattr(st.session_state, 'show_quiz') and st.session_state.show_quiz:
                    topic = st.session_state.quiz_topic

                    st.markdown(f"### {topic} Quiz")

                    with st.form(key=f"quiz_form_{topic}"):
                        all_correct = True

                        for i, (sub, ql) in enumerate(FORMULA_QUIZ_BANK[topic].items()):
                            st.subheader(f"Section: {sub}")

                            for j, q in enumerate(ql):
                                q_key = f"q_{topic}_{i}_{j}"
                                st.markdown(f"**Q{j + 1}:** {q['question']}")

                                if q['type'] == 'formula':
                                    answer = st.text_input("Your answer:", key=q_key)
                                    st.session_state.quiz_answers[q_key] = answer
                                else:
                                    options = ["Option A", "Option B", "Option C"]
                                    answer = st.radio("Select:", options, key=q_key)
                                    st.session_state.quiz_answers[q_key] = answer

                                st.markdown("---")

                        submit = st.form_submit_button("Submit Quiz")

                        if submit:
                            # Update quiz progress
                            prev = QUIZ_PROGRESS.setdefault(sid, {}).get(topic, 0)
                            QUIZ_PROGRESS[sid][topic] = prev + 1

                            # Show success message
                            st.success("Quiz submitted successfully!")

                            # Show feedback option
                            st.markdown("### How was this quiz?")
                            st.slider("Difficulty", 1, 5, 3)
                            st.text_area("Feedback (optional)")

                            if st.button("Send Feedback"):
                                st.success("Thank you for your feedback!")
                                st.session_state.show_quiz = False
        else:
            st.info("Complete some topics to unlock quizzes!")

        # Performance Analytics Section
        st.markdown('<p class="medium-font">Performance Analytics</p>', unsafe_allow_html=True)
        # Question-Level Analytics Section
        st.markdown('<p class="medium-font">Question-Level Analytics</p>', unsafe_allow_html=True)

        # Display problematic questions
        problem_questions = analyze_item_level_performance(sid)

        if problem_questions:
            st.warning("🔍 Questions you're struggling with:")

            for q_id, q_text, topic, success_rate in problem_questions:
                with st.expander(f"{topic}: {q_text} (Success: {success_rate:.0%})"):
                    st.write("**Suggested micro-lesson:**")
                    st.write(f"This question from {topic} is causing difficulty. Here's a targeted approach:")

                    # Create micro-lesson recommendation based on topic
                    if topic == "Algebra":
                        st.write("- Break the problem into steps")
                        st.write("- Check if you can isolate the variable")
                        st.write("- Try substituting simple values to check your work")
                    elif topic == "Geometry":
                        st.write("- Draw the figure and label all known values")
                        st.write("- Identify the key formula needed")
                        st.write("- Look for similar triangles or other patterns")
                    elif topic == "Calculus":
                        st.write("- Review the relevant differentiation/integration rules")
                        st.write("- Break complex functions into simpler parts")
                        st.write("- Double-check your algebraic manipulations")
                    else:
                        st.write("- Review the fundamental concepts in this area")
                        st.write("- Try creating a visual representation")
                        st.write("- Practice with simpler examples first")

                    # Add practice button
                    if st.button(f"Generate similar practice question for {q_id}"):
                        st.session_state.show_practice = True
                        st.session_state.practice_topic = topic
                        st.session_state.practice_id = q_id
        else:
            st.info("No specific question difficulties identified yet. Keep practicing!")

        # Show practice question if requested
        if hasattr(st.session_state, 'show_practice') and st.session_state.show_practice:
            topic = st.session_state.practice_topic
            q_id = st.session_state.practice_id

            st.markdown("### Practice Question")
            st.markdown(f"Topic: **{topic}**")

            # Generate a similar question (in a real app, this would use an LLM API)
            st.markdown("**Question:** " + QUESTION_BANK[topic][0]['text'])

            user_answer = st.text_input("Your answer:")

            if st.button("Check Answer"):
                # In a real app, this would evaluate the answer
                correct = np.random.choice([True, False], p=[0.7, 0.3])

                if correct:
                    st.success("Correct! Well done.")
                else:
                    st.error("Not quite right. Try again!")

                # Record the response
                if sid not in QUESTION_RESPONSES:
                    QUESTION_RESPONSES[sid] = {}
                if q_id not in QUESTION_RESPONSES[sid]:
                    QUESTION_RESPONSES[sid][q_id] = []

                QUESTION_RESPONSES[sid][q_id].append(int(correct))

                st.session_state.show_practice = False
        # Filter data for current student
        student_data = df[df.StudentID == sid]

        if not student_data.empty:
            # Create tabs for different analytics
            tab1, tab2, tab3 = st.tabs(["Topic Performance", "Time Analysis", "Usage Patterns"])

            with tab1:
                # Topic performance chart
                topic_perf = student_data.groupby('Topic').agg(
                    Accuracy=('Correct', 'mean'),
                    Attempts=('Correct', 'count')
                ).reset_index()

                if not topic_perf.empty:
                    # Sort by performance
                    topic_perf = topic_perf.sort_values('Accuracy', ascending=False)

                    # Create bar chart with Plotly
                    fig = go.Figure()

                    # Add bar for accuracy
                    fig.add_trace(go.Bar(
                        x=topic_perf['Topic'],
                        y=topic_perf['Accuracy'],
                        name='Accuracy',
                        marker_color='green',
                        text=[f"{acc:.1%}" for acc in topic_perf['Accuracy']],
                        textposition='auto'
                    ))

                    # Update layout
                    fig.update_layout(
                        title='Topic Performance',
                        yaxis=dict(title='Accuracy', tickformat='.0%', range=[0, 1]),
                        xaxis=dict(title='Topic'),
                        height=400,
                        hovermode='closest'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No performance data available.")

            with tab2:
                # Time analysis
                time_data = student_data.groupby('Topic').agg(
                    AvgTime=('TimeTaken', 'mean'),
                    Accuracy=('Correct', 'mean')
                ).reset_index()

                if not time_data.empty:
                    # Create scatter plot
                    fig = go.Figure()

                    # Add scatter plot
                    fig.add_trace(go.Scatter(
                        x=time_data['AvgTime'],
                        y=time_data['Accuracy'],
                        mode='markers+text',
                        marker=dict(
                            size=15,
                            color=time_data['Accuracy'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title='Accuracy')
                        ),
                        text=time_data['Topic'],
                        textposition="top center"
                    ))

                    # Update layout
                    fig.update_layout(
                        title='Time vs. Accuracy by Topic',
                        xaxis=dict(title='Average Time Taken'),
                        yaxis=dict(title='Accuracy', tickformat='.0%', range=[0, 1]),
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.caption("Topics in the upper-left quadrant are your strengths (high accuracy, low time).")
                    st.caption("Topics in the lower-right quadrant need more focus (low accuracy, high time).")
                else:
                    st.info("No time analysis data available.")

            with tab3:
                # Usage patterns
                usage_data = student_data.groupby('Topic').agg(
                    Videos=('VideosWatched', 'sum'),
                    Quizzes=('QuizzesTaken', 'sum'),
                    Practice=('PracticeSessions', 'sum'),
                    Media=('MediaClicks', 'sum')
                ).reset_index()

                if not usage_data.empty and not usage_data[['Videos', 'Quizzes', 'Practice', 'Media']].sum().sum() == 0:
                    # Melt the dataframe for stacked bar chart
                    usage_melted = pd.melt(
                        usage_data,
                        id_vars=['Topic'],
                        value_vars=['Videos', 'Quizzes', 'Practice', 'Media'],
                        var_name='Activity',
                        value_name='Count'
                    )

                    # Create stacked bar chart
                    fig = go.Figure()

                    # Color mapping
                    colors = {
                        'Videos': '#1f77b4',
                        'Quizzes': '#ff7f0e',
                        'Practice': '#2ca02c',
                        'Media': '#d62728'
                    }

                    # Add bars for each activity type
                    for activity, color in colors.items():
                        activity_data = usage_melted[usage_melted['Activity'] == activity]
                        fig.add_trace(go.Bar(
                            x=activity_data['Topic'],
                            y=activity_data['Count'],
                            name=activity,
                            marker_color=color
                        ))

                    # Update layout for stacked bars
                    fig.update_layout(
                        title='Resource Usage by Topic',
                        xaxis=dict(title='Topic'),
                        yaxis=dict(title='Usage Count'),
                        barmode='stack',
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Add correlation analysis
                    corr_data = student_data.groupby('Topic').agg(
                        Videos=('VideosWatched', 'sum'),
                        Quizzes=('QuizzesTaken', 'sum'),
                        Practice=('PracticeSessions', 'sum'),
                        Media=('MediaClicks', 'sum'),
                        Accuracy=('Correct', 'mean')
                    )

                    # Calculate correlations with accuracy
                    if not corr_data.empty:
                        corrs = {}
                        for col in ['Videos', 'Quizzes', 'Practice', 'Media']:
                            corrs[col] = np.corrcoef(corr_data[col], corr_data['Accuracy'])[0, 1]

                        # Display most effective resources
                        st.subheader("Most Effective Resources")
                        best_activity = max(corrs, key=corrs.get)
                        best_corr = corrs[best_activity]

                        if not np.isnan(best_corr):
                            st.info(
                                f"💡 **{best_activity}** show the strongest positive correlation with your performance ({best_corr:.2f}).")

                        # Suggest resource focus
                        low_usage = usage_data.melt(
                            id_vars=['Topic'],
                            value_vars=['Videos', 'Quizzes', 'Practice', 'Media']
                        )
                        low_usage = low_usage[low_usage['value'] == 0]

                        if not low_usage.empty:
                            st.warning(
                                f"You haven't used {low_usage['Activity'].nunique()} resource types. Try diversifying your study approach!")
                else:
                    st.info("No usage data available yet. Use more learning resources to see patterns.")
        else:
            st.info("No data available for the selected student.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.warning("Please refresh the page and try again.")
    # Question-Level Analytics Section
    st.markdown('<p class="medium-font">Question-Level Analytics</p>', unsafe_allow_html=True)

    # Display problematic questions
    problem_questions = analyze_item_level_performance(sid)

    if problem_questions:
        st.warning("🔍 Questions you're struggling with:")

        for q_id, q_text, topic, success_rate in problem_questions:
            with st.expander(f"{topic}: {q_text} (Success: {success_rate:.0%})"):
                st.write("**Suggested micro-lesson:**")
                st.write(f"This question from {topic} is causing difficulty. Here's a targeted approach:")

                # Create micro-lesson recommendation based on topic
                if topic == "Algebra":
                    st.write("- Break the problem into steps")
                    st.write("- Check if you can isolate the variable")
                    st.write("- Try substituting simple values to check your work")
                elif topic == "Geometry":
                    st.write("- Draw the figure and label all known values")
                    st.write("- Identify the key formula needed")
                    st.write("- Look for similar triangles or other patterns")
                elif topic == "Calculus":
                    st.write("- Review the relevant differentiation/integration rules")
                    st.write("- Break complex functions into simpler parts")
                    st.write("- Double-check your algebraic manipulations")
                else:
                    st.write("- Review the fundamental concepts in this area")
                    st.write("- Try creating a visual representation")
                    st.write("- Practice with simpler examples first")

                # Add practice button
                if st.button(f"Generate similar practice question for {q_id}"):
                    st.session_state.show_practice = True
                    st.session_state.practice_topic = topic
                    st.session_state.practice_id = q_id
    else:
        st.info("No specific question difficulties identified yet. Keep practicing!")

    # Show practice question if requested
    if hasattr(st.session_state, 'show_practice') and st.session_state.show_practice:
        topic = st.session_state.practice_topic
        q_id = st.session_state.practice_id

        st.markdown("### Practice Question")
        st.markdown(f"Topic: **{topic}**")

        # Generate a similar question (in a real app, this would use an LLM API)
        st.markdown("**Question:** " + QUESTION_BANK[topic][0]['text'])

        user_answer = st.text_input("Your answer:")

        if st.button("Check Answer"):
            # In a real app, this would evaluate the answer
            correct = np.random.choice([True, False], p=[0.7, 0.3])

            if correct:
                st.success("Correct! Well done.")
            else:
                st.error("Not quite right. Try again!")

            # Record the response
            if sid not in QUESTION_RESPONSES:
                QUESTION_RESPONSES[sid] = {}
            if q_id not in QUESTION_RESPONSES[sid]:
                QUESTION_RESPONSES[sid][q_id] = []

            QUESTION_RESPONSES[sid][q_id].append(int(correct))

            st.session_state.show_practice = False

if __name__ == "__main__":
    main()
