#!/usr/bin/env python3
"""
Simple log analysis script that uses GenAI to enhance the analysis.
This will search log files for specific terms and analyze the results.
"""

import os
import re
import sys
import argparse
import json
from datetime import datetime
from collections import Counter, defaultdict
import logging
from agent_helper import enhance_solutions, is_ai_enhancement_enabled

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_log_files(directory: str, max_files: int = 100) -> list:
    """Find all log files in a directory.

    Args:
        directory: Path to the directory containing log files.
        max_files: Maximum number of log files to find.

    Returns:
        list: List of paths to log files.
    """
    log_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.log'):
                log_files.append(os.path.join(root, file))
            if len(log_files) >= max_files:
                break
    return log_files

def search_files_for_term(files: list, search_term: str, max_matches: int = 100000) -> list:
    """Search files for a specific term.

    Args:
        files: List of file paths to search.
        search_term: Term to search for in logs.
        max_matches: Maximum number of matches to return.

    Returns:
        list: List of match dictionaries with file, line number, and content.
    """
    matches = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f):
                    if search_term.lower() in line.lower():
                        matches.append({
                            'file': file_path,
                            'line_number': i + 1,
                            'content': line.strip()
                        })
                        if len(matches) >= max_matches:
                            break
            if len(matches) >= max_matches:
                break
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    return matches

def parse_log_entry(line: str) -> dict:
    """Parse a log line into structured data.

    Args:
        line: The log line to parse.

    Returns:
        dict: Structured log entry with timestamp, severity, component, and message.
    """
    log_entry = {'raw': line}
    
    # Extract timestamp
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', line)
    if timestamp_match:
        log_entry["timestamp"] = timestamp_match.group(1)
    
    # Extract severity
    severity_match = re.search(r'\b(ERROR|INFO|WARNING|DEBUG|CRITICAL|WARN|FATAL)\b', line, re.IGNORECASE)
    if severity_match:
        log_entry["severity"] = severity_match.group(1).upper()
    else:
        log_entry["severity"] = "UNKNOWN"
    
    # Extract component and message
    if timestamp_match:
        remainder = line[timestamp_match.end():].strip()
    else:
        remainder = line
        
    if severity_match:
        component_msg = remainder.replace(severity_match.group(0), "", 1).strip()
        
        # Extract component (in brackets or before colon)
        component_match = re.search(r'^\[([^\]]+)\]|^([^:]+):', component_msg)
        if component_match:
            component = component_match.group(1) or component_match.group(2)
            log_entry["component"] = component.strip()
            
            # Message is the rest
            if component_match.group(1):  # [component] format
                log_entry["message"] = component_msg[component_match.end():].strip()
            else:  # component: format
                log_entry["message"] = component_msg[component_match.end():].strip()
        else:
            log_entry["message"] = component_msg
    else:
        log_entry["message"] = remainder
    
    return log_entry

def analyze_log_entries(entries: list) -> dict:
    """Analyze log entries to extract patterns and insights.

    Args:
        entries: List of log entry dictionaries.

    Returns:
        dict: Analysis results with total entries, severity distribution, components, time patterns, and error patterns.
    """
    total_entries = len(entries)
    severities = Counter()
    components = Counter()
    errors_by_component = defaultdict(list)
    timestamps = []
    
    # Extract data
    for entry in entries:
        parsed = parse_log_entry(entry['content'])
        severities[parsed.get('severity', 'UNKNOWN')] += 1
        
        if 'component' in parsed:
            components[parsed['component']] += 1
            if parsed.get('severity') in ['ERROR', 'CRITICAL', 'FATAL']:
                errors_by_component[parsed['component']].append(parsed.get('message', ''))
        
        if 'timestamp' in parsed:
            timestamps.append(parsed['timestamp'])
    
    # Analyze time patterns
    time_pattern = None
    if timestamps:
        try:
            timestamps = sorted(timestamps)
            time_pattern = {
                'first_occurrence': timestamps[0],
                'last_occurrence': timestamps[-1],
                'total_occurrences': len(timestamps)
            }
        except Exception as e:
            logger.error(f"Error analyzing timestamps: {e}")
    
    # Find most common error patterns
    error_patterns = []
    for component, errors in errors_by_component.items():
        for error in errors:
            if error and len(error) > 10:  # Only consider substantial errors
                pattern = re.sub(r'\b[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}\b', '<ID>', error)  # Replace UUIDs
                pattern = re.sub(r'\d+', '<NUM>', pattern)  # Replace numbers
                error_patterns.append((component, pattern))
    
    common_patterns = Counter(error_patterns).most_common(10)
    
    return {
        'total_entries': total_entries,
        'severity_distribution': dict(severities),
        'components': dict(components),
        'time_pattern': time_pattern,
        'error_patterns': [{'component': comp, 'pattern': pat, 'count': count} 
                           for (comp, pat), count in common_patterns]
    }

def suggest_solutions(analysis: dict) -> list:
    """Suggest solutions based on the analysis.

    Args:
        analysis: The analysis results from analyze_log_entries.

    Returns:
        list: List of solution dictionaries with problem and solution keys.
    """
    solutions = []
    
    # Check for connection issues
    connection_errors = any('connection' in pattern.lower() or 'timeout' in pattern.lower() or 'connect' in pattern.lower() 
                          for item in analysis.get('error_patterns', []) 
                          for pattern in [item['pattern'].lower()])
    
    if connection_errors:
        solutions.append({
            'problem': 'Connection issues',
            'solution': 'Check network connectivity between services and verify that all dependent services are running. Look for firewall or DNS issues.'
        })
    
    # Check for permission issues
    permission_errors = any('permission' in pattern.lower() or 'access' in pattern.lower() or 'denied' in pattern.lower() 
                          for item in analysis.get('error_patterns', []) 
                          for pattern in [item['pattern'].lower()])
    
    if permission_errors:
        solutions.append({
            'problem': 'Permission issues',
            'solution': 'Verify file and resource permissions. Check that service accounts have the necessary access rights.'
        })
    
    # Check for resource issues
    resource_errors = any('memory' in pattern.lower() or 'cpu' in pattern.lower() or 'capacity' in pattern.lower() or 'full' in pattern.lower()
                        for item in analysis.get('error_patterns', []) 
                        for pattern in [item['pattern'].lower()])
    
    if resource_errors:
        solutions.append({
            'problem': 'Resource constraints',
            'solution': 'Check system resources (memory, CPU, disk space). Consider scaling up infrastructure or optimizing resource usage.'
        })
    
    # Database issues
    db_errors = any('database' in pattern.lower() or 'db' in pattern.lower() or 'sql' in pattern.lower() or 'query' in pattern.lower()
                   for item in analysis.get('error_patterns', []) 
                   for pattern in [item['pattern'].lower()])
    
    if db_errors:
        solutions.append({
            'problem': 'Database issues',
            'solution': 'Check database connectivity, query performance, and database logs. Verify that database indices are properly set up.'
        })
    
    # General solution if nothing specific found
    if not solutions:
        if analysis.get('total_entries', 0) > 0:
            severity_dist = analysis.get('severity_distribution', {})
            error_count = severity_dist.get('ERROR', 0) + severity_dist.get('CRITICAL', 0) + severity_dist.get('FATAL', 0)
            
            components = analysis.get('components', {})
            most_common_component = max(components.items(), key=lambda x: x[1])[0] if components else 'unknown'
            
            solutions.append({
                'problem': f'Multiple errors in {most_common_component} component',
                'solution': f'Review the {most_common_component} component logs in detail and check recent code changes or configuration updates to this component.'
            })
    
    return solutions

def main():
    parser = argparse.ArgumentParser(description="Log Analysis with AI assistance")
    parser.add_argument("--logs", type=str, default="./data/logs", help="Directory containing log files")
    parser.add_argument("--term", type=str, default="error", help="Search term for logs")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--disable-ai", action="store_true", help="Disable AI enhancement")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    # Set environment variables
    if args.disable_ai:
        os.environ["DISABLE_AI_ENHANCEMENT"] = "true"
        logger.info("AI Enhancement has been DISABLED via command-line flag")
    else:
        logger.info("AI Enhancement status will be determined by environment settings")
    
    if args.debug:
        os.environ["DEBUG"] = "1"
        logger.info("Debug mode is ENABLED")
    
    # Verify log directory exists
    if not os.path.exists(args.logs):
        logger.error(f"Log directory '{args.logs}' does not exist")
        sys.exit(1)
    
    logger.info(f"Searching in {args.logs} for term '{args.term}'...")
    
    # Find log files
    log_files = find_log_files(args.logs)
    logger.info(f"Found {len(log_files)} log files")
    
    if args.verbose:
        logger.info("Log files found:")
        for file in log_files[:10]:
            logger.info(f"  - {file}")
        if len(log_files) > 10:
            logger.info(f"  ... and {len(log_files) - 10} more")
    
    # Search for term in files
    matches = search_files_for_term(log_files, args.term)
    logger.info(f"Found {len(matches)} matches for term '{args.term}'")
    
    if args.verbose:
        logger.info("Sample matches:")
        for match in matches[:5]:
            logger.info(f"  - {match['file']}:{match['line_number']}: {match['content'][:100]}...")
    
    # Analyze log entries
    logger.info("Analyzing log entries...")
    analysis = analyze_log_entries(matches)
    
    # Generate solution suggestions
    logger.info("Generating solutions...")
    solutions = suggest_solutions(analysis)
    
    # Prepare the results
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'search_term': args.term,
            'log_directory': args.logs,
            'total_files_searched': len(log_files),
            'total_matches': len(matches)
        },
        'matches': matches,
        'analysis': analysis,
        'solutions': solutions
    }
    
    # Use AI to enhance the solutions if possible
    ai_status = is_ai_enhancement_enabled()
    if ai_status:
        try:
            logger.info("Enhancing solutions with AI...")
            results = enhance_solutions(results)
        except Exception as e:
            logger.error(f"Error enhancing solutions: {e}")
            results["ai_enhancement_used"] = False
            results["ai_error"] = str(e)
    else:
        results["ai_enhancement_used"] = False
    
    # Output the results
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results written to {args.output}")
        except Exception as e:
            logger.error(f"Error writing results to {args.output}: {e}")
    else:
        # Print summary to console
        logger.info("\n--- Analysis Summary ---")
        logger.info(f"Total entries: {analysis['total_entries']}")
        
        if 'severity_distribution' in analysis:
            logger.info("\nSeverity distribution:")
            for severity, count in analysis['severity_distribution'].items():
                logger.info(f"  - {severity}: {count}")
        
        ai_enhanced_solutions = results.get('solutions', [])
        
        if ai_enhanced_solutions:
            logger.info("\nSuggested solutions:")
            for i, solution in enumerate(ai_enhanced_solutions, 1):
                problem = solution.get('problem', 'Unknown issue')
                solution_text = solution.get('solution', '')
                
                if os.getenv("DEBUG") == "1":
                    logger.debug(f"Solution {i} content length: {len(solution_text)}")
                    logger.debug(f"Solution {i} first 50 chars: {solution_text[:50]}...")
                
                solution_lines = [line for line in solution_text.split('\n') if line.strip()]
                if len(solution_lines) > 4:
                    display_solution = '\n     '.join(solution_lines[:3]) + '\n     ...'
                else:
                    display_solution = '\n     '.join(solution_lines)
                
                logger.info(f"  {i}. {problem}")
                logger.info(f"     {display_solution}")
                if solution.get('ai_enhanced', False):
                    logger.info("     ✨ AI Enhanced")
                
                if i < len(ai_enhanced_solutions):
                    logger.info("")
        
        if results.get("ai_enhancement_used", False):
            model_name = results.get("ollama_model_used", "llama3.2")
            logger.info(f"\n✨ Solutions were enhanced using Ollama model: {model_name}")
        elif results.get("ai_error"):
            logger.info(f"\n⚠️ AI enhancement failed: {results.get('ai_error')}")
        else:
            logger.info("\n⚠️ AI enhancement was not used")

if __name__ == "__main__":
    main()