Log Clustering and Analysis System
Project Overview
Introduction
The Log Clustering and Analysis System is a Python-based tool designed to process, cluster, and analyze log files from diverse sources, such as Apache web server and Linux system logs. By leveraging machine learning, natural language processing (NLP), and a Streamlit web interface, it groups similar log entries into clusters, identifies patterns or issues, and provides AI-enhanced summaries and actionable recommendations. This system is ideal for system administrators and DevOps teams to monitor system health, troubleshoot errors, and detect anomalies efficiently.
Purpose
The primary goal is to automate log analysis, reducing manual effort and providing actionable insights into system behavior. It supports multiple log formats (e.g., Apache, Linux) and handles large datasets with memory-efficient techniques, making it suitable for real-world IT environments.
Features
Log Ingestion and Preprocessing

Supports Apache ([Day MMM DD HH:MM:SS YYYY]) and Linux (MMM DD HH:MM:SS) log formats with auto-detection.
Normalizes log content by removing timestamps and hostnames to focus on semantic content for clustering.
Handles multiple encodings (UTF-8, Latin-1, ISO-8859-1) for robust file reading.

Log Clustering

Uses SentenceTransformer (all-MiniLM-L6-v2) to generate 384-dimensional embeddings for log messages.
Applies DBSCAN clustering with a sparse cosine distance matrix to group similar logs efficiently.
Identifies noise points (outliers) to handle diverse or irregular log entries.

AI-Enhanced Analysis

Integrates LangChain and Ollama (LLaMA3.2) to generate detailed summaries, problem descriptions, and solutions for each cluster.
Produces file-wide alerts to highlight critical issues or trends across clusters.
Includes fallback mechanisms for LLM failures to ensure reliability.

User Interface

Built with Streamlit, offering a responsive UI with three sections: Summary, Insights, and Clusters.
Supports user-configurable options for log directory, format selection (auto, Apache, Linux), and AI toggling.
Displays format-specific details and AI enhancement indicators for clarity.

Memory Optimization

Processes logs in batches (e.g., 32 for embeddings, 500 for distance matrices) to reduce memory usage.
Uses sparse matrices (scipy.sparse.csr_matrix) in DBSCAN to handle large datasets.
Caps log ingestion at 1000 entries per file to prevent memory errors (e.g., 128 MiB allocation issue).

Installation
Prerequisites

Python: Version 3.8 or higher
pip: Package manager for Python
Ollama: Server with LLaMA3.2 model for AI enhancement
System Requirements: At least 4 GB RAM recommended (original system had 126 MB LOWMEM, which required optimization)

Install Dependencies
Install required Python packages:
pip install streamlit pandas sentence-transformers scikit-learn langchain-ollama scipy numpy python-dateutil

Set Up Ollama

Install Ollama: Follow instructions at Ollama.
Pull the LLaMA3.2 model:ollama pull llama3.2


Start the Ollama server:ollama serve



Clone and Configure

Clone the repository:git clone https://github.com/your-repo/log-clustering-system.git
cd log-clustering-system


Place log files in a directory (e.g., ./data/logs/weblogs/).
Run the application:streamlit run ui_app.py



Usage
Running the Application

Launch the Streamlit app:streamlit run ui_app.py


Access the web interface at http://localhost:8501.

Configuring Analysis

Log Directory: Specify the directory containing .log files (e.g., ./data/logs/weblogs/).
Log Format: Select auto, apache, or linux from the sidebar dropdown.
Options: Enable Verbose Output for detailed logs or Disable AI Enhancement for lightweight processing.
Run: Click the "Run Analysis" button to process logs.

Viewing Results
The UI displays results in three sections:

Summary: Metrics like total files, logs, clusters, noise points, and detected log format.
Insights: AI-generated alerts with selected logs, explanations, and suggestions.
Clusters: Detailed cluster information, including names, summaries, problems, solutions, and sample logs.

Example Log Files

Apache Log:[Thu Jun 09 06:07:05 2005] [error] env.createBean2(): Factory error creating channel.jni:jni


Linux Log:Jun  9 06:06:20 combo syslogd 1.4.1: restart



Example Output
The system generates a JSON file (clustered_logs.json) and displays results in the UI:
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
    },
    {
      "cluster_id": 1,
      "log_count": 3,
      "cluster": "System Startup Events",
      "summary": "Logs related to Linux system and kernel startup.",
      "problem": "General cluster pattern",
      "solution": "Review logs for further investigation.",
      "original_logs": ["syslogd 1.4.1: restart", "..."],
      "ai_summarized": true,
      "log_format": "linux"
    }
  ],
  "metadata": {
    "timestamp": "2025-07-02T23:09:00.123456",
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
System Architecture
The system is modular, with three core components:

clustering.py:
Log Loading: Reads and preprocesses logs, handling multiple formats and encodings.
Embedding: Generates semantic embeddings using SentenceTransformer.
Clustering: Groups logs using DBSCAN with sparse matrices.
Export: Saves results to JSON with clustering metrics.


agent_helper.py:
AI Summarization: Uses LangChain and Ollama for cluster summaries and alerts.
Format Detection: Identifies log formats to tailor LLM prompts.
Error Handling: Provides fallback summaries for LLM failures.


ui_app.py:
UI: Renders results in a Streamlit interface with configurable options.
Timestamp Parsing: Extracts timestamps using dateutil.parser.
Pipeline: Orchestrates the end-to-end analysis process.



Key Algorithms

SentenceTransformer: Generates 384-dimensional embeddings for log messages, capturing semantic similarity.
DBSCAN: Clusters logs using cosine distance, with eps=0.5 and min_samples=3, optimized for noise handling.
Cosine Distance: Computed in chunks and stored as a sparse matrix to measure log similarity.

Memory Optimization

Batch Processing: Embeds logs in batches of 32 and computes distances in chunks of 500.
Sparse Matrices: Reduces memory usage in DBSCAN with scipy.sparse.csr_matrix.
Log Capping: Limits ingestion to 1000 logs per file to avoid errors (e.g., 128 MiB allocation for (1804, 74380) arrays).

Challenges and Solutions
Memory Constraints

Challenge: A 128 MiB allocation error occurred due to large dense arrays in DBSCAN.
Solution: Implemented sparse matrices and batch processing, with a log cap of 1000 entries.

Diverse Log Formats

Challenge: Apache and Linux logs have different timestamp and message structures.
Solution: Added regex-based format detection and preprocessing, with flexible timestamp parsing using dateutil.

LLM Reliability

Challenge: LLM failures could produce invalid or empty summaries.
Solution: Implemented fallback summaries and format-specific prompts to improve AI output.

Future Improvements
Enhanced Log Support

Add support for additional formats like Nginx or JSON logs.
Implement custom format parsers via configuration files.

Real-Time Analysis

Integrate with log streaming tools (e.g., Kafka) for live monitoring.
Use incremental clustering algorithms like HDBSCAN.

Scalability

Deploy on distributed frameworks (e.g., Spark) for massive datasets.
Cache embeddings in a vector database (e.g., FAISS).

Visualization

Add interactive charts for log trends and anomaly detection.
Implement time-series visualizations for timestamp patterns.

Contributing
How to Contribute

Fork the repository.
Create a feature branch:git checkout -b feature/your-feature


Commit changes:git commit -m 'Add your feature'


Push to the branch:git push origin feature/your-feature


Open a pull request.

Guidelines

Follow PEP 8 for Python code.
Include tests for new features.
Update documentation for changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, contact work.aditya1889@example.com.
Acknowledgments

Built with support from libraries like Streamlit, SentenceTransformers, and LangChain.
Inspired by the need to automate log analysis for system administration.
