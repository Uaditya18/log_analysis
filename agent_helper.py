import os
import json
from langchain_ollama import OllamaLLM
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_MODEL = "llama3.2"

# --- Check AI Enhancement Status ---
def is_ai_enhancement_enabled() -> bool:
    """
    Check if AI enhancement is enabled based on environment variable.

    Returns:
        bool: True if AI enhancement is enabled, False otherwise.
    """
    return os.getenv("DISABLE_AI_ENHANCEMENT", "false").lower() != "true"

# --- LangChain + OllamaLLM Model Wrapper ---
class LangchainOllamaLLMModel:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the LangChain OllamaLLM model.

        Args:
            model_name: Name of the Ollama model to use (defaults to 'llama3.2').
        """
        self.model_name = model_name
        try:
            self.llm = OllamaLLM(model=model_name)
            logger.info(f"LangChain OllamaLLM model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OllamaLLM: {e}")
            raise

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Make the class callable to generate text using the Ollama model.

        Args:
            prompt: The input prompt for the model.
            **kwargs: Additional parameters for the model.

        Returns:
            str: The generated response from the model.
        """
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Error invoking OllamaLLM: {e}")
            raise

# --- Cluster Summarization Function ---
def summarize_cluster(model, cluster_id: int, logs: List[str]) -> Dict:
    """
    Summarize a cluster of logs using the Ollama model.

    Args:
        model: The initialized Ollama model instance.
        cluster_id: The ID of the cluster.
        logs: List of log entries in the cluster.

    Returns:
        dict: Dictionary with summary, problem, solution, cluster title, and ai_summarized status.
    """
    log_examples = logs[:3]  # Limit to 3 logs to avoid token limits
    log_examples_text = "\n".join(log_examples) if log_examples else "No logs available"

    prompt = f"""
You are a professional log analyst. Analyze the following log entries from a cluster and provide a detailed analysis of the common patterns, issues, or themes. Based on the log content, assign a descriptive title to the cluster that reflects the type of logs it contains (e.g., 'Authentication Errors', 'Database Connection Issues', 'HTTP Request Failures').

Cluster ID: {cluster_id}
Log Count: {len(logs)}
Sample Logs:
{log_examples_text}

Provide a JSON response with:
- cluster: A descriptive title for the cluster based on the type of logs (e.g., 'Authentication Errors', 'Database Connection Issues'). Ensure the title is concise and specific to the log content.
- summary: A concise description of the common patterns or issues in the cluster.
- problem: A specific problem description (if identifiable, else "General cluster pattern").
- solution: Detailed resolution steps (if applicable, else "Review logs for further investigation.").

Return only the JSON object.
"""

    try:
        if os.getenv("DEBUG") == "1":
            logger.debug(f"Prompt for cluster {cluster_id}:\n{prompt}")
        
        response = model(prompt)
        parsed = json.loads(response.strip())
        
        if os.getenv("DEBUG") == "1":
            logger.debug(f"Summary for cluster {cluster_id}: {json.dumps(parsed, indent=2)[:100]}...")
        
        if not parsed.get("summary") or len(parsed["summary"].strip()) < 20:
            logger.warning(f"Summary for cluster {cluster_id} is too short or empty. Using fallback.")
            return {
                "cluster": f"Cluster {cluster_id}",
                "summary": f"Cluster {cluster_id} contains {len(logs)} logs with varied content.",
                "problem": "General cluster pattern",
                "solution": "Review logs for further investigation.",
                "ai_summarized": False
            }
        
        return {
            "cluster": parsed.get("cluster", f"Cluster {cluster_id}"),
            "summary": parsed.get("summary", f"Cluster {cluster_id} summary"),
            "problem": parsed.get("problem", "General cluster pattern"),
            "solution": parsed.get("solution", "Review logs for further investigation."),
            "ai_summarized": True
        }
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Error generating summary for cluster {cluster_id}: {e}")
        return {
            "cluster": f"Cluster {cluster_id}",
            "summary": f"Cluster {cluster_id} contains {len(logs)} logs with varied content.",
            "problem": "General cluster pattern",
            "solution": "Review logs for further investigation.",
            "ai_summarized": False
        }

# --- Cluster Summarization Entry Point ---
def summarize_clusters(cluster_data: List[Dict]) -> Dict:
    """
    Summarize clusters from clustering.py's JSON output using Ollama and generate file-wide alerts.

    Args:
        cluster_data: List of dictionaries from clustering.py JSON, each with cluster_id, log_count, logs.

    Returns:
        dict: Summarized results with cluster summaries, file-wide alerts, and metadata.
    """
    if not is_ai_enhancement_enabled():
        logger.info("AI enhancement is disabled, returning basic summaries")
        result = {
            "alerts": "No alerts identified",
            "clusters": [
                {
                    "cluster_id": cluster["cluster_id"],
                    "log_count": cluster["log_count"],
                    "cluster": f"Cluster {cluster['cluster_id']}",
                    "summary": f"Cluster {cluster['cluster_id']} contains {cluster['log_count']} logs with varied content.",
                    "problem": "General cluster pattern",
                    "solution": "Review logs for further investigation.",
                    "original_logs": cluster["logs"],
                    "ai_summarized": False
                }
                for cluster in cluster_data
            ],
            "ai_enhancement_used": False
        }
        return result

    try:
        logger.info("Summarizing clusters using LangChain + OllamaLLM...")
        model = LangchainOllamaLLMModel()
        summarized_clusters = []

        # Summarize individual clusters
        for cluster in cluster_data:
            cluster_id = cluster.get("cluster_id", -1)
            logs = cluster.get("logs", [])
            log_count = cluster.get("log_count", len(logs))

            logger.info(f"Summarizing cluster {cluster_id} with {log_count} logs")
            summary_data = summarize_cluster(model, cluster_id, logs)

            summarized_clusters.append({
                "cluster_id": cluster_id,
                "log_count": log_count,
                "cluster": summary_data["cluster"],
                "summary": summary_data["summary"],
                "problem": summary_data["problem"],
                "solution": summary_data["solution"],
                "original_logs": logs,
                "ai_summarized": summary_data["ai_summarized"]
            })

        # Generate file-wide alerts
        all_logs_text = "\n".join(
            f"Cluster {cluster['cluster_id']} ({cluster.get('cluster', 'Unnamed Cluster')}):\n" +
            "\n".join(cluster["logs"][:3])
            for cluster in cluster_data
        ) or "No logs available"

        alerts_prompt = f"""
You are a professional log analyst. Analyze the following log clusters to identify meaningful insights about the log file, such as critical issues, trends, anomalies, or other significant observations. Select a variable number of logs from the entire log file that provide valuable insights into system behavior or issues. For each selected log, provide an explanation of its significance and specific suggestions or actions to address or leverage the insight. Do not generate insights for individual clusters; instead, select logs from across all clusters to highlight the most relevant observations.

Log Clusters:
{all_logs_text}

Provide a JSON response as a list of objects, each containing:
  - log: The specific log entry selected.
  - explanation: A clear explanation of the significance of the log (e.g., issue, trend, or anomaly).
  - suggestions: Specific suggestions or actions to address or leverage the insight.

Return only the JSON list.
"""

        alerts = "No alerts identified"
        try:
            alerts_response = model(alerts_prompt)
            parsed_alerts = json.loads(alerts_response.strip())
            if isinstance(parsed_alerts, list) and len(parsed_alerts) > 0 and all(
                isinstance(alert, dict) and all(key in alert for key in ["log", "explanation", "suggestions"])
                for alert in parsed_alerts
            ):
                alerts = parsed_alerts
            else:
                logger.warning("Invalid alerts format received from LLM. Using fallback.")
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error generating file-wide alerts: {e}")
           
        result = {
            "alerts": alerts,
            "clusters": summarized_clusters,
            "ai_enhancement_used": True,
            "ollama_model_used": DEFAULT_MODEL
        }
        logger.info(f"Clusters summarized and alerts generated successfully with Ollama model: {DEFAULT_MODEL}")
        return result

    except Exception as e:
        logger.error(f"Error during cluster summarization: {e}")
        result = {
            "alerts": "No alerts identified",
            "clusters": [
                {
                    "cluster_id": cluster["cluster_id"],
                    "log_count": cluster["log_count"],
                    "cluster": f"Cluster {cluster['cluster_id']}",
                    "summary": f"Cluster {cluster['cluster_id']} contains {cluster['log_count']} logs with varied content.",
                    "problem": "General cluster pattern",
                    "solution": "Review logs for further investigation.",
                    "original_logs": cluster["logs"],
                    "ai_summarized": False
                }
                for cluster in cluster_data
            ],
            "ai_enhancement_used": False,
            "ai_error": str(e)
        }
        return result