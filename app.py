import streamlit as st
import os

# Load .env locally without external libraries
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k] = v

import database
import agents
import json
import importlib
importlib.reload(agents)
importlib.reload(database)

# Force page config
st.set_page_config(page_title="Graphify Swarm", layout="wide", initial_sidebar_state="expanded")

# Custom Premium Styling (Sleek Dark Cyberpunk Theme)
st.markdown("""
<style>
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
    }
    div[data-testid="stSidebar"] {
        background-color: #12161b;
        border-right: 1px solid #1f2937;
    }
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subheader {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #181d24;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .agent-title {
        color: #38bdf8;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .agent-focus {
        font-size: 0.85rem;
        color: #818cf8;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    .agent-comment {
        background-color: #1c2330;
        border-left: 3px solid #818cf8;
        padding: 0.75rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
database.init_db()

# Session State defaults
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")
if "specialists" not in st.session_state:
    st.session_state.specialists = []
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False
if "synthesis" not in st.session_state:
    st.session_state.synthesis = ""
if "debate_history" not in st.session_state:
    st.session_state.debate_history = []
if "chart_info" not in st.session_state:
    st.session_state.chart_info = None

# Sidebar Configuration
with st.sidebar:
    st.markdown("<h2 style='color: #38bdf8; margin-top:0;'>⚙️ Settings</h2>", unsafe_allow_html=True)
    
    # Check if loaded from secrets/environment first to avoid displaying/exposing it
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    if env_api_key:
        st.success("🔒 API Key loaded from secrets")
        st.session_state.api_key = env_api_key
    else:
        api_key_input = st.text_input("Google AI Studio API Key", type="password", value=st.session_state.api_key)
        if api_key_input:
            st.session_state.api_key = api_key_input
        
    if st.button("🗑️ Reset Database & Swarm", use_container_width=True):
        database.clear_db()
        st.session_state.specialists = []
        st.session_state.ingestion_done = False
        st.session_state.synthesis = ""
        st.session_state.debate_history = []
        st.session_state.chart_info = None
        st.success("App state reset successfully!")
        st.rerun()
        
    st.markdown("---")
    st.markdown("<h3 style='color: #e2e8f0;'>📂 Ingest Document</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])
    
    # Check if local test_document.pdf exists
    test_pdf_path = os.path.join(os.path.dirname(__file__), "test_document.pdf")
    has_test_pdf = os.path.exists(test_pdf_path)
    
    load_test = False
    if has_test_pdf and st.session_state.api_key:
        if st.button("🧪 Ingest test_document.pdf (5 pages)", use_container_width=True):
            load_test = True
            
    if (uploaded_file or load_test) and st.session_state.api_key:
        if load_test or st.button("🚀 Process & Build Graph", use_container_width=True):
            # Reset old states instantly so old data doesn't persist during loading
            st.session_state.specialists = []
            st.session_state.ingestion_done = False
            st.session_state.synthesis = ""
            st.session_state.debate_history = []
            st.session_state.chart_info = None
            
            # Save uploaded file locally or copy test doc
            temp_path = os.path.join(os.path.dirname(__file__), "temp_uploaded.pdf")
            if load_test:
                import shutil
                shutil.copy(test_pdf_path, temp_path)
            else:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with st.spinner("Processing PDF and chunking..."):
                # Clear previous database state
                database.clear_db()
                ingestor = agents.IngestorAgent(api_key=st.session_state.api_key)
                chunks, _ = ingestor.process_pdf(temp_path)
                
            with st.spinner("Formulating ontology schema..."):
                architect = agents.ArchitectAgent(api_key=st.session_state.api_key)
                ontology = architect.generate_ontology(chunks)
                
            with st.spinner("Extracting graph nodes & edges (gemini-3.1-flash-lite)..."):
                architect.extract_graph(chunks, ontology)
                st.session_state.ingestion_done = True
                
            st.success("Graph constructed successfully in local SQLite!")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            # Trigger Specialist Spawn
            with st.spinner("Spawning 5 Specialist Agents..."):
                director = agents.DirectorAgent(api_key=st.session_state.api_key)
                st.session_state.specialists = director.analyze_and_spawn_swarm()
                
            st.success("5 Specialist Agents generated and active!")
            st.rerun()
            
    if not st.session_state.api_key:
        st.warning("Please provide your Gemini API key in the input box above.")

# Main Page Layout
st.markdown("<div class='main-header'>🕸️ Graphify Swarm</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Autonomous Multi-Agent Predictive Intelligence Engine</div>", unsafe_allow_html=True)

# Fetch current nodes and edges
nodes = database.get_all_nodes()
edges = database.get_all_edges()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h3 style='color:#38bdf8;'>📊 Knowledge Graph Explorer</h3>", unsafe_allow_html=True)
    if not nodes:
        st.info("Upload a document to build the knowledge graph.")
    else:
        st.write(f"**Canonical Nodes:** {len(nodes)} | **Weighted Edges:** {len(edges)}")
        
        # Interactive tables via Tabs
        tab1, tab2 = st.tabs(["📌 Nodes Table", "🔗 Edges Table"])
        with tab1:
            node_df = [{"Name": n["name"], "Type": n["type"], "Description": n["description"]} for n in nodes]
            st.dataframe(node_df, use_container_width=True, height=250)
        with tab2:
            edge_df = [{"Source": e["source"], "Relationship": e["type"], "Target": e["target"], "Confidence": e.get("confidence", 1.0), "Context/Evidence": e.get("quote", "")} for e in edges]
            st.dataframe(edge_df, use_container_width=True, height=250)
        
        # Interactive 2D Force Graph Render (High-clarity with text labels and mouse zoom)
        graph_data = {
            "nodes": [{"id": n["id"], "name": n["name"], "type": n["type"], "description": n["description"]} for n in nodes],
            "links": [{"source": e["source"], "target": e["target"], "label": e["type"], "weight": e["weight"]} for e in edges]
        }
        graph_data_json = json.dumps(graph_data)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <script src="https://unpkg.com/force-graph"></script>
          <style>
            body {{ margin: 0; background-color: #0d0f12; overflow: hidden; }}
            #2d-graph {{ width: 100%; height: 400px; }}
            .graph-tooltip {{
                background: rgba(11, 12, 16, 0.95) !important;
                border: 1px solid #38bdf8 !important;
                border-radius: 8px !important;
                padding: 8px 12px !important;
                color: #fff !important;
                font-family: sans-serif !important;
                font-size: 12px;
                max-width: 250px;
            }}
          </style>
        </head>
        <body>
          <div id="controls" style="position: absolute; top: 10px; left: 10px; z-index: 10;">
             <button onclick="Graph.zoomToFit(400)" style="background: rgba(11,12,16,0.9); border: 1px solid #38bdf8; color: #fff; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; font-family: sans-serif;">🔍 Recenter</button>
          </div>
          <div id="2d-graph"></div>
          <script>
            const gData = {graph_data_json};
            
            const typeColors = {{
                'person': '#ff4b91',
                'company': '#00d2ff',
                'organization': '#00d2ff',
                'location': '#00f5d4',
                'event': '#9b5de5',
                'technology': '#fee440',
                'concept': '#ff9f1c'
            }};
            
            function getColor(type) {{
                if (!type) return '#a8dadc';
                const t = type.toLowerCase();
                for (const [k, v] of Object.entries(typeColors)) {{
                    if (t.includes(k)) return v;
                }}
                return '#a8dadc';
            }}

            // Pre-calculate node degrees for centrality sizing
            const degrees = {{}};
            gData.links.forEach(l => {{
                const srcId = typeof l.source === 'object' ? l.source.id : l.source;
                const tgtId = typeof l.target === 'object' ? l.target.id : l.target;
                degrees[srcId] = (degrees[srcId] || 0) + 1;
                degrees[tgtId] = (degrees[tgtId] || 0) + 1;
            }});

            const Graph = ForceGraph()(document.getElementById('2d-graph'))
                .graphData(gData)
                .width(document.getElementById('2d-graph').clientWidth)
                .height(400)
                .showNavInfo(false)
                .nodeCanvasObject((node, ctx, globalScale) => {{
                    const label = node.name;
                    const degree = degrees[node.id] || 0;
                    const radius = Math.min(12, 4.5 + degree * 1.2);
                    const fontSize = 11 / Math.max(0.5, globalScale * 0.4);
                    
                    // Draw node glow/shadow
                    ctx.shadowColor = getColor(node.type);
                    ctx.shadowBlur = 12;
                    
                    // Draw colored dot
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
                    ctx.fillStyle = getColor(node.type);
                    ctx.fill();
                    
                    ctx.shadowBlur = 0; // Disable shadow for strokes and text
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 0.8;
                    ctx.stroke();

                    // Draw text label
                    ctx.font = `bold ${{fontSize}}px sans-serif`;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.fillStyle = '#e2e8f0';
                    ctx.fillText(label, node.x, node.y + radius + 3);
                }})
                .linkWidth(link => Math.log(link.weight || 1) + 1.2)
                .linkColor(link => 'rgba(255, 255, 255, 0.25)')
                .linkDirectionalArrowLength(4)
                .linkDirectionalArrowRelPos(0.95)
                .linkDirectionalParticles(2)
                .linkDirectionalParticleWidth(1.8)
                .linkDirectionalParticleSpeed(0.006)
                .nodeLabel(node => `
                    <div class="graph-tooltip">
                        <div style="font-weight:bold;color:#38bdf8;">${{node.name}}</div>
                        <div style="font-size:10px;color:#818cf8;text-transform:uppercase;margin-bottom:4px;">${{node.type || 'Unknown'}}</div>
                        <div>${{node.description || ''}}</div>
                    </div>
                `)
                .linkLabel(link => `
                    <div class="graph-tooltip">
                        <div style="font-weight:bold;color:#818cf8;">${{link.source.name || link.source}} &rarr; ${{link.label}} &rarr; ${{link.target.name || link.target}}</div>
                    </div>
                `);
          </script>
        </body>
        </html>
        """
        st.components.v1.html(html_content, height=410)

with col2:
    st.markdown("<h3 style='color:#38bdf8;'>👥 Spawned Swarm (5 Specialists)</h3>", unsafe_allow_html=True)
    if not st.session_state.specialists:
        st.info("The swarm will spawn automatically after graph ingestion.")
    else:
        # Render the 5 spawned agents as expanders
        for idx, agent in enumerate(st.session_state.specialists):
            with st.expander(f"🤖 Agent {idx+1}: {agent['name']} ({agent['focus']})"):
                st.markdown(f"**System Prompt:**\n`{agent['system_prompt']}`")
        
        # Explicitly show Visualizer and Shared Tools in the UI
        with st.expander("📊 Utility Agent: Data Visualizer & Coder"):
            st.markdown("""
            **Focus:** Scanning debate logs for numerical forecast trends, auto-generating and sandboxing matplotlib code.
            
            **System Prompt:**
            `You are a Data Visualization Expert. Analyze the scenario and debate history. Output Python matplotlib code with a custom dark-mode theme to render predictive charts.`
            """)
            
        with st.expander("🔍 Shared Swarm Tools"):
            st.markdown("""
            All specialists can call external tools to ground their predictions:
            1. **`web_search(query)`**: Search Google/web for real-time compliance updates and news.
            2. **`reddit_search(query)`**: Search Reddit for public sentiment and merchant discussions.
            """)

# Bottom section for Simulation/Predictions
st.markdown("<h3 style='color:#818cf8;'>🔮 Predictive Simulation Arena</h3>", unsafe_allow_html=True)

if not st.session_state.specialists:
    st.warning("Please ingest a document first to unlock the simulation arena.")
else:
    scenario = st.text_input("Type a 'What-If' Trigger Scenario:", placeholder="e.g. What if there is a severe shortage of components?")
    
    if scenario and st.button("🔥 Run Simulation"):
        with st.spinner("Orchestrating 5-agent debate loop (2 rounds)..."):
            director = agents.DirectorAgent(api_key=st.session_state.api_key)
            synthesis, debate_history, chart_info = director.run_simulation(scenario, st.session_state.specialists)
            st.session_state.synthesis = synthesis
            st.session_state.debate_history = debate_history
            st.session_state.chart_info = chart_info
            
    # Persistently display results if they exist in state
    if st.session_state.synthesis:
        sc_col1, sc_col2 = st.columns([1, 1])
        
        with sc_col1:
            st.markdown("<h4 style='color:#38bdf8;'>💬 Live Debate Log</h4>", unsafe_allow_html=True)
            # Render logs grouped by round
            for round_idx in range(1, 3):
                st.markdown(f"**Round {round_idx}**")
                round_comments = [c for c in st.session_state.debate_history if c["round"] == round_idx]
                for comment in round_comments:
                    st.markdown(f"<div class='agent-comment'><b>{comment['agent']}</b>:<br>{comment['text']}</div>", unsafe_allow_html=True)
                    
        with sc_col2:
            st.markdown("<h4 style='color:#818cf8;'>📈 Forecast Synthesis Report</h4>", unsafe_allow_html=True)
            st.markdown(st.session_state.synthesis)
            
            # Display generated chart if available
            c_info = st.session_state.chart_info
            if c_info and c_info.get("needs_chart") and c_info.get("success"):
                chart_path = os.path.join(os.path.dirname(__file__), "scratch", "temp_chart.png")
                if os.path.exists(chart_path):
                    st.markdown("---")
                    st.markdown(f"<h5 style='color:#00f5d4;'>📊 Visual Forecast Analysis</h5>", unsafe_allow_html=True)
                    st.image(chart_path, caption=c_info.get("chart_description", ""), use_container_width=True)
                    with st.expander("💻 Show Visualisation Code"):
                        st.code(c_info.get("python_code", ""), language="python")
