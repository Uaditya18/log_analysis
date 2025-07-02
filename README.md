Log Clustering and Analysis System
Overview
The Log Clustering and Analysis System is a Python-based application designed to process, cluster, and analyze log files from various sources, such as Apache web server and Linux system logs. By leveraging machine learning, natural language processing (NLP), and a user-friendly Streamlit web interface, the system groups similar log entries into clusters, identifies patterns or issues, and generates AI-enhanced summaries and actionable recommendations. This tool is ideal for system administrators and DevOps teams to monitor system health, troubleshoot errors, and detect anomalies efficiently.
Features

Multi-Format Log Support: Processes Apache ([Day MMM DD HH:MM:SS YYYY]) and Linux (MMM DD HH:MM:SS) log formats, with auto-detection and preprocessing to normalize content.
Log Clustering: Uses DBSCAN with SentenceTransformer embeddings (all-MiniLM-L6-v2) to group similar logs based on semantic content, optimized for memory efficiency with sparse matrices.
AI-Enhanced Insights: Integrates LangChain and Ollama (LLaMA3.2) to generate detailed cluster summaries, problem descriptions, solutions, and file-wide alerts.
Interactive UI: Built with Streamlit, featuring a responsive interface with summary metrics, insights, and cluster details, including format-specific displays.
Memory Optimization: Handles large datasets through batch processing, sparse matrices, and log capping to prevent memory errors (e.g., 128 MiB allocation issues).
Robust Error Handling: Manages encoding issues, LLM failures, and invalid inputs with fallbacks and detailed logging.

Installation
Prerequisites

Python 3.8+
pip package manager
Ollama server with LLaMA3.2 model (for AI enhancement)
Sufficient memory (4 GB RAM recommended; original system had 126 MB LOWMEM)

Dependencies
Install required Python packages using:
pip install streamlit pandas sentence-transformers scikit-learn langchain-ollama scipy numpy python-dateutil

Ollama Setup

Install Ollama: Follow instructions at Ollama.
Pull the LLaMA3.2 model:ollama pull llama3.2


Ensure the Ollama server is running:ollama serve



Project Setup

Clone the repository:git clone https://github.com/your-repo/log-clustering-system.git
cd log-clustering-system


Organize log files in a directory (e.g., ./data/logs/weblogs/).
Run the application:streamlit run ui_app.py



Usage

Launch the Application:
Run streamlit run ui_app.py to start the Streamlit web interface.


Configure Analysis:
In the sidebar, specify the log directory (e.g., ./data/logs/weblogs/).
Select the log format (auto, apache, linux).
Check Verbose Output for detailed logs or Disable AI Enhancement to skip LLM processing.


Run Analysis:
Click the "Run Analysis" button to process logs.
View results in three sections: Summary (metrics), Insights (AI-generated alerts), and Clusters (grouped logs with summaries).


Interpret Results:
Summary: Displays total files, logs, clusters, noise points, and detected log format.
Insights: Highlights critical logs with explanations and suggestions.
Clusters: Shows cluster names, summaries, problems, solutions, and sample logs, with AI enhancement indicators.



Example Log Files

Apache Log:[Thu Jun 09 06:07:05 2005] [error] env.createBean2(): Factory error creating channel.jni:jni


Linux Log:Jun  9 06:06:20 combo syslogd 1.4.1: restart



Example Output
The system generates a JSON output (clustered_logs.json) and displays results in the UI, such as:
{
  "alerts": [
    {
      "log": "[Thu Jun 09 06:07:05 2005] [error] env.createBean2(): Factory error creating channel.jni:jni",
      "explanation": "Failure in Apache mod_jk module due to JNI channel initialization error.",
      "suggestions": "Verify JAVA_HOME and JDK installation. Check workers2.properties configuration."
    },
    {
      "log": "Jun  9 06:06:20 combo kernel: Linux version 2.6.5-1.358 ...",
      "explanation": "Indicates successful kernel startup during system boot.",
      "suggestions": "No action needed unless boot issues are reported."
    }
  ],
  "clusters": [
    {
      "cluster_id": 0,
      "log_count": 5,
      "cluster": "JNI Configuration Errors",
      "summary": "Logs related to mod_jk JNI initialization failures.",
      "problem": "Failure to create JNI channel or VM.",
      "solution": "Ensure JDK is installed and JAVA_HOME is set.",
      "original_logs": ["[error] env.createBean2(): Factory error creating channel.jni:jni", "..."],
      "ai_summarized": true,
      "log_format": "apache"
    }
  ],
  "metadata": {
    "timestamp": "2025-07-02T23:07:00.123456",
    "log_directory": "./data/logs/weblogs/",
    "total_files_searched": 2,
    "total_logs": 30,
    "n_clusters": 2,
    "n_noise": 0,
    "log_format": "mixed"
  },
  "analysis": {
    "total_entries": 30,
    "cluster_sizes": {"0": 5, "1": 3, "-1": 0},
    "time_pattern": {
      "first_occurrence": "2005-06-09T06:06:20",
      "last_occurrence": "2005-06-09T06:07:20",
      "total_occurrences": 30
    }
  },
  "ai_enhancement_used": true,
  "ollama_model_used": "llama3.2"
}

Technical Details
Architecture
The system is modular, with three main components:

clustering.py:
Log Loading: Reads log files with robust encoding handling and preprocesses logs to normalize content.
Embedding: Uses SentenceTransformer (all-MiniLM-L6-v2) to generate 384-dimensional embeddings in batches.
Clustering: Applies DBSCAN with a sparse cosine distance matrix to group logs, optimized for memory efficiency.
Export: Saves clustered logs to JSON with metrics like cluster sizes and silhouette scores.


agent_helper.py:
AI Summarization: Uses LangChain and Ollama (LLaMA3.2) to generate cluster summaries, problem descriptions, solutions, and file-wide alerts.
Format Detection: Identifies log formats (Apache, Linux) to provide context-specific prompts.
Error Handling: Falls back to generic summaries if LLM fails.


ui_app.py:
Interface: Streamlit-based UI with summary, insights, and cluster sections.
Timestamp Parsing: Uses dateutil.parser to handle diverse timestamp formats.
Analysis Pipeline: Orchestrates log loading, clustering, and summarization, with user-configurable options.



Key Algorithms

SentenceTransformer: Generates semantic embeddings for log messages, enabling clustering based on content similarity.
DBSCAN: Clusters logs using cosine distance, with eps=0.5 and min_samples=3, handling noise and variable cluster sizes.
Cosine Distance: Measures semantic similarity between log embeddings, computed in chunks and stored as a sparse matrix.

Memory Optimization

Batch Processing: Embeds logs in batches of 32 and computes distance matrices in chunks of 500.
Sparse Matrices: Uses scipy.sparse.csr_matrix to reduce memory usage during DBSCAN clustering.
Log Capping: Limits log ingestion to 1000 entries per file to prevent memory errors (e.g., 128 MiB allocation issue for (1804, 74380) arrays).

Challenges and Solutions

Memory Errors: Addressed a 128 MiB allocation error by using sparse matrices and batch processing in clustering.py.
Diverse Log Formats: Implemented regex-based format detection and preprocessing to handle Apache and Linux logs, with flexible timestamp parsing using dateutil.
LLM Reliability: Added fallback summaries for LLM failures and format-specific prompts to improve AI-generated insights.
Scalability: Optimized for large datasets with batch processing and sparse matrices, with potential for distributed computing integration.

Future Improvements

Additional Log Formats: Support for Nginx, JSON, or custom log formats.
Real-Time Analysis: Integration with log streaming tools like Kafka for live monitoring.
Scalability: Use distributed frameworks (e.g., Spark) for massive datasets.
Visualization: Add interactive charts for log trends and anomaly detection.
Security: Implement data sanitization for sensitive information and user authentication in the UI.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, please contact your-email@example.com.
