Enhanced Learning Analytics Dashboard 🧠

A powerful Streamlit-based dashboard that provides personalized learning recommendations and analytics using knowledge graphs, collaborative filtering, and data-driven insights.
Features
Knowledge Graph Visualization: Interactive topic relationship map showing prerequisites, applications, and statistical connections
Personalized Recommendations: AI-powered recommendations across study plans, challenges, and engagement resources
Performance Analytics: Detailed topic-level and question-level performance insights
Interactive Quizzes: Formative assessments with immediate feedback
Peer Insights: Comparative analytics and peer tutoring matches
Adaptive Content: Motivation-sensitive recommendations that adapt to student mood
Demo
![Dashboareenshots of main dashboard components here*
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
How It Works
1. Knowledge Graph Engine
The system builds a comprehensive knowledge graph that represents:
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
Future Enhancements
Integration with external LMS data sources
 integrating api's for Natural language processing for content recommendation
Expanded peer learning features
Mobile-responsive design
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
