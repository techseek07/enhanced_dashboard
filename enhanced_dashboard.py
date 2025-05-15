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

    # Initialize critical variables
    sid = None
    df = pd.DataFrame()
    G = nx.DiGraph()
    seg = pd.DataFrame()

    try:
        # Generate student data
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

        # --- Sidebar Section ---
        st.sidebar.markdown('<p class="medium-font">Settings</p>', unsafe_allow_html=True)

        # Student selection with validation
        all_students = sorted(df.StudentID.unique())
        if not all_students:
            st.sidebar.error("No students found in data")
            st.stop()
            
        sid = st.sidebar.selectbox("Select Student", all_students)

        # Student tier display
        student_info = seg[seg.StudentID == sid]
        tier = "Unknown"
        if not student_info.empty:
            tier = student_info['label'].iloc[0]
            tier_colors = {'Topper': 'green', 'Average': 'blue', 'Poor': 'red'}
            tier_color = tier_colors.get(tier, 'gray')
            st.sidebar.markdown(
                f'<div style="background-color:{tier_color}20; padding:10px; border-radius:5px;">'
                f'<strong>Performance Tier:</strong> {tier}</div>',
                unsafe_allow_html=True
            )

        # --- Main Content Sections ---
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
                            traces.append(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=col, width=2), name=rel))

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
                    recommendations = get_recommendations(sid, df, G, seg, tier)
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
                            st.markdown(f'<div style="background-color:#f0f7fb; padding:10px; border-radius:5px;">{rec}</div>', unsafe_allow_html=True)
                        for rec in rec_types["🎥 Media"] + rec_types["🔗 Analogy"] + rec_types["🔄 Apply"] + rec_types["👍 Easy Win"]:
                            st.warning(rec)
                except Exception as e:
                    st.error(f"Recommendation error: {str(e)}")

        # Quiz Section
        try:
            st.markdown('<p class="medium-font">Interactive Quiz Section</p>', unsafe_allow_html=True)
            if sid is not None:
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
                                            answer = st.text_input("Your answer:", key=q_key)
                                        else:
                                            answer = st.radio("Select:", ["Option A", "Option B", "Option C"], key=q_key)
                                        st.markdown("---")
                                
                                if st.form_submit_button("Submit Quiz"):
                                    QUIZ_PROGRESS.setdefault(sid, {})[topic] = QUIZ_PROGRESS.get(sid, {}).get(topic, 0) + 1
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
                                marker_color='green'
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
                            colors = {'Videos': '#1f77b4', 'Quizzes': '#ff7f0e', 
                                    'Practice': '#2ca02c', 'Media': '#d62728'}
                            for col, color in colors.items():
                                fig.add_trace(go.Bar(
                                    x=usage_data['Topic'], y=usage_data[col],
                                    name=col, marker_color=color
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
                                st.write("- Break the problem into steps\n- Check variable isolation\n- Substitute values")
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
                else:
                    st.info("No specific question difficulties identified yet. Keep practicing!")
        except Exception as e:
            st.error(f"Question analysis error: {str(e)}")

    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
