import streamlit as st
import pandas as pd
import os
import json
import tempfile
from datetime import datetime
from clustering import load_logs_from_file, vectorize_logs, cluster_logs, group_logs_by_cluster
from agent_helper import summarize_clusters
from collections import Counter
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_log_files(directory: str, max_files: int = 100) -> list:
    """Find all log files in a directory."""
    log_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.log'):
                log_files.append(os.path.join(root, file))
            if len(log_files) >= max_files:
                break
    return log_files

def run_analysis(log_dir, verbose=False, disable_ai=False):
    """Run clustering and summarization on log files."""
    try:
        # Set AI enhancement environment variable
        if disable_ai:
            os.environ["DISABLE_AI_ENHANCEMENT"] = "true"
        else:
            os.environ["DISABLE_AI_ENHANCEMENT"] = "false"

        # Find all log files
        log_files = find_log_files(log_dir)
        if not log_files:
            st.error(f"No .log files found in {log_dir}")
            return None

        if verbose:
            st.write("Log files found:")
            for file in log_files[:10]:
                st.write(f"- {file}")
            if len(log_files) > 10:
                st.write(f"... and {len(log_files) - 10} more")

        # Load and process logs
        all_logs = []
        log_file_paths = []
        for log_file in log_files:
            logs = load_logs_from_file(log_file)
            all_logs.extend([{"file": log_file, "line_number": i + 1, "content": log} for i, log in enumerate(logs)])
            log_file_paths.extend([log_file] * len(logs))

        if not all_logs:
            st.error("No logs found in the provided directory")
            return None

        # Vectorize and cluster
        log_contents = [log["content"] for log in all_logs]
        embeddings = vectorize_logs(log_contents)
        labels = cluster_logs(embeddings)
        clustered_logs = group_logs_by_cluster(log_contents, labels)

        # Prepare cluster data for summarization
        cluster_data = [
            {
                "cluster_id": int(cid),
                "log_count": len(logs),
                "logs": logs
            }
            for cid, logs in sorted(clustered_logs.items())
        ]

        # Summarize clusters
        results = summarize_clusters(cluster_data)

        # Add metadata and metrics
        cluster_sizes = Counter(labels)
        n_clusters = len([cid for cid in cluster_sizes if cid != -1])
        n_noise = cluster_sizes.get(-1, 0)

        # Attempt to extract time distribution
        timestamps = []
        for log in log_contents:
            match = re.search(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', log)
            if match:
                timestamps.append(match.group(1))

        time_pattern = None
        if timestamps:
            try:
                timestamps = sorted(timestamps)
                time_pattern = {
                    "first_occurrence": timestamps[0],
                    "last_occurrence": timestamps[-1],
                    "total_occurrences": len(timestamps)
                }
            except Exception as e:
                logger.error(f"Error analyzing timestamps: {e}")

        results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "log_directory": log_dir,
            "total_files_searched": len(log_files),
            "total_logs": len(all_logs),
            "n_clusters": n_clusters,
            "n_noise": n_noise
        }
        results["analysis"] = {
            "total_entries": len(all_logs),
            "cluster_sizes": dict(cluster_sizes),
            "time_pattern": time_pattern
        }
        results["matches"] = all_logs  # Include all logs for reference

        if verbose:
            st.write(f"Total logs: {len(all_logs)}")
            st.write(f"Number of clusters: {n_clusters}")
            st.write(f"Noise points: {n_noise}")

        return results

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        logger.error(f"Analysis failed: {e}")
        return None

def display_results(results):
    """Display analysis results in Streamlit layout with three fixed, scrollable sections."""
    metadata = results.get("metadata", {})
    analysis = results.get("analysis", {})
    clusters = results.get("clusters", [])

    ai_status = results.get("ai_enhancement_used", False)
    if ai_status:
        st.success("✨ AI Enhancement: ENABLED (via Ollama + LLaMA3.2)")
    else:
        st.warning("⚠️ AI Enhancement: DISABLED")

    # Create two vertical columns
    col1, col2 = st.columns(2)

    with col1:
        # Top-left: Summary section (fixed height, scrollable)
        with st.container(height=300):
            st.header("Summary")
            col1a, col1b = st.columns(2)
            col1a.metric("Total Files", metadata.get("total_files_searched", 0))
            col1a.metric("Total Logs", metadata.get("total_logs", 0))
            col1b.metric("Clusters Found", metadata.get("n_clusters", 0))
            col1b.metric("Noise Points", metadata.get("n_noise", 0))

        # Bottom-left: Insights section (fixed height, scrollable)
        with st.container(height=600):
            st.header("Insights")
            insights = results.get("alerts", "No insights identified")
            if isinstance(insights, str):
                st.info(insights)
            else:
                for i, insight in enumerate(insights, 1):
                    with st.expander(f"Insight {i}: Log Highlight"):
                        st.write(f"**Log**:")
                        st.code(insight.get("log", "No log provided"))
                        st.write(f"**Explanation**: {insight.get('explanation', 'No explanation provided')}")
                        st.write(f"**Suggestions**: {insight.get('suggestions', 'No suggestions provided')}")
                        if ai_status:
                            st.markdown("**<span style='color:green'>✨ AI Enhanced</span>**", unsafe_allow_html=True)

    with col2:
        # Right: Clusters section (fixed height, scrollable)
        with st.container(height=600):
            st.header("Clusters")
            if clusters:
                for i, cluster in enumerate(clusters, 1):
                    with st.expander(f"Cluster {i}: {cluster.get('cluster', 'Cluster ' + str(cluster['cluster_id']))} ({cluster['log_count']} logs)"):
                        st.write(f"**Cluster Name**: {cluster.get('cluster', 'Cluster ' + str(cluster['cluster_id']))}")
                        st.write(f"**Summary**: {cluster['summary']}")
                        st.write(f"**Problem**: {cluster['problem']}")
                        st.write(f"**Solution**: {cluster['solution']}")
                        st.write("**Sample Logs**:")
                        for log in cluster["original_logs"][:3]:
                            st.code(log)
                        if cluster.get("ai_summarized"):
                            st.markdown("**<span style='color:green'>✨ AI Enhanced</span>**", unsafe_allow_html=True)
            else:
                st.info("No clusters available")

            st.subheader("Time Distribution")
            tp = analysis.get("time_pattern")
            if tp:
                st.write(f"First: {tp.get('first_occurrence')}")
                st.write(f"Last: {tp.get('last_occurrence')}")
                st.write(f"Total: {tp.get('total_occurrences', 'N/A')}")
            else:
                st.info("No time data available")

def main():
    st.set_page_config(
        page_title="Log Clustering System",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Log Clustering System")
    st.write("A powerful tool for clustering and analyzing log files using AI (Ollama + LLaMA3.2)")

    # Sidebar
    st.sidebar.header("Configuration")
    log_source = st.sidebar.text_input("Log Source Directory", value="./data/logs/weblogs/")
    verbose = st.sidebar.checkbox("Verbose Output")
    disable_ai = st.sidebar.checkbox("Disable AI Enhancement")

    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Clustering and analyzing logs..."):
            results = run_analysis(log_source, verbose, disable_ai)
            if results:
                display_results(results)

    with st.sidebar.expander("ℹ️ Help"):
        st.write("""
        1. Enter log directory containing .log files
        2. Click "Run Analysis"
        3. View clustered logs and AI-enhanced summaries/solutions
        """)

if __name__ == "__main__":
    main()