Enhanced Learning Analytics Dashboard 🧠

A powerful Streamlit-based dashboard that provides personalized learning recommendations and analytics using knowledge graphs, collaborative filtering, and data-driven insights.
Features
Knowledge Graph Visualization: Interactive topic relationship map showing prerequisites, applications, and statistical connections
Personalized Recommendations: AI-powered recommendations across study plans, challenges, and engagement resources
Performance Analytics: Detailed topic-level and question-level performance insights
Interactive Quizzes: Formative assessments with immediate feedback
Peer Insights: Comparative analytics and peer tutoring matches
Adaptive Content: Motivation-sensitive recommendations that adapt to student mood
Note: Initial graph building takes approximately 5-6 minutes to load
Demo
Add screenshots of main dashboard components here
Installation
bash
# Clone the repository
git clone https://github.com/yourusername/enhanced-learning-dashboard.git
cd enhanced-learning-dashboard

# Install required packages
pip install -r requirements.txt and runtime.txt

# Run the application
streamlit run enhanced_dashboard_complete.py
Requirements
Python 3.7+
Streamlit
pandas
numpy
networkx
plotly
scikit-learn
xgboost
shap
data generation done using:-
The function generates data for 75(optimized here)  students by default with four distinct learning profiles distributed according to specific probabilities:
30% video-heavy learners
25% practice-heavy learners
20% quiz-heavy learners
25% balanced learners
Student Characteristic Generation
For each student, the function creates:
Base characteristics:
base_skill: Random beta distribution (slightly skewed toward higher values)
motivation: Categorical (High/Medium/Low) with probabilities [0.4, 0.4, 0.2]
exam_date: Random date within the past 60 days
Profile-based behaviors:
For video-heavy students: High videos, low practice, low quizzes
For practice-heavy: Low videos, high practice, medium quizzes
For quiz-heavy: Medium videos, medium practice, high quizzes
For balanced: Medium videos, medium practice, medium quizzes
For each topic interaction, it generates:
Random subtopics from the topic's subtopic list
Dirichlet-distributed weights for the subtopics
Success probability that factors in:
Base skill
Motivation factor
Learning activities (videos, practice, quizzes)
Time taken (modeled with gamma distribution)
Other metrics like quiz progress and media clicks
Built-in Safety Mechanisms
The code includes several safeguards:
Fallback values for failed random distributions
Exception handling for individual student generation
Safe subtopic sampling with replacement if needed
The resulting DataFrame contains student-topic interactions with 19 features including performance metrics, learning behaviors, and demographic information - creating a realistic dataset for the learning analytics dashboard.


How It Works
1. Knowledge Graph Engine
The system builds a comprehensive knowledge graph using  that represents:
Topic Prerequisites: Essential foundational knowledge
Subtopic Relationships: Fine-grained concept mapping
Real-world Applications: Practical relevance connections
Statistical Relationships: Data-driven relationships (odds ratio, SHAP importance)
The graph is used to prioritize learning paths, identify knowledge gaps, and contextualize student progress.
2. Recommendation System
Recommendations are organized into three categories:
Study Plan: Bridge courses, practice questions, and formula help
Challenges: Higher-order thinking questions, quizzes, and key subtopics
Engagement: Media resources, analogies, applications, and motivational content
All recommendations are personalized based on:
Knowledge graph relationships
Student performance data
Motivation level
Learning history
3. Student Performance Analytics
Multiple visualization layers provide insights into:
Topic-level performance with accuracy metrics
Time-to-completion analysis
Usage pattern tracking
Question-level difficulties with solution paths
Knowledge connections through graph analytics
4. Quiz System
The interactive quiz system:
Updates the knowledge graph based on quiz performance
Tracks student mastery of concepts
Identifies specific question-level difficulties
Recommends targeted follow-up resources
Implementation Details
Data Models
Student interaction data tracking multiple metrics
Knowledge graph with tiered edge weights
Quiz response tracking with temporal analysis
Question difficulty assessment
Algorithm Components
Student Segmentation: K-means clustering for performance tiers
Knowledge Graph Construction: Multi-phase graph building with statistical validation
Recommendations: Multi-factor prioritization algorithm
Collaborative Filtering: Peer-based recommendation engine
Question Analytics: Performance-based difficulty assessment
UI Components
Interactive graph visualization
Multi-column recommendation layout
Tabbed analytics sections
Expandable detailed insights
Mood-tracking interface
Usage
Select a Student: Choose a student profile from the sidebar
View Recommendations: Access personalized study plans in the recommendations section
Explore Analytics: Navigate through performance data using the interactive charts
Take Quizzes: Test knowledge with topic-specific formula quizzes
Track Progress: Monitor improvement through the knowledge graph visualization
Customization
The system uses configuration dictionaries that can be easily modified:
TOPICS and SUBTOPICS: Define your curriculum structure
PREREQUISITES: Set topic dependencies
APPLICATION_RELATIONS: Connect topics to real-world applications
QUESTION_BANK: Add your own assessment items
MEDIA_LINKS, BRIDGE_COURSES, etc.: Customize learning resources


DETAILED IMPLEMENTATION STEPS:-
Expert-Defined Relationships: Manual mapping of prerequisites and applications
Statistical Relationship Detection:
Odds Ratio calculation via LogisticRegression (for datasets ≥50 samples)
Stratified contingency table analysis (smaller datasets)
SHAP (SHapley Additive exPlanations) feature importance using XGBoost
Dual-Tier Edge Scoring: Weighted relationship importance based on edge type and connectivity
2. Student Segmentation Algorithm
K-means Clustering (from sklearn) on:
Accuracy metrics
Time-taken metrics
Adaptive Clustering: Handles edge cases with single students or small datasets
3. Recommendation Algorithms
Graph-Enhanced Bridge Course: Prioritizes foundation topics with many prerequisite edges
Practice Recommendation: Weighted multi-factor ranking combining:
Recency (recently completed topics)
Graph relationship weights
Edge relationship types
Enhanced Subtopic Selection: Combines centrality measures with question performance
Easy Win Selection: Topic strength scoring based on mastery and graph connectivity
4. Collaborative Learning Algorithms
Peer Matching: Complementary strength/weakness detection
Similar Student Filtering: Performance-based similarity grouping
Progression Comparison: Time-normalized performance gap analysis
5. Quiz and Assessment Algorithms
Formula Answer Validation: String normalization with exponent handling
Graph Updating from Quiz Results: Knowledge mastery calculation with subtopic relationship strengthening(updates knowledge graph based on quiz result).
Question Difficulty Analysis: Success rate threshold detection with question categorization
6. Network Analysis Methods
Betweenness Centrality: Used to identify important bridging concepts
Spring Layout: For knowledge graph visualization
Edge Weight Normalization: Capping relationship strengths for visual clarity


Future Enhancements:-
Integration with external LMS data sources
 implementing Causal Bayesian Networks Integration: and PC algorithm could predict better causation effect for building the knowldege graph.
Integrating APIs for Natural language processing for content recommendation
Expanded peer learning features
Mobile-responsive design
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
