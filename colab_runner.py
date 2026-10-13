#!/usr/bin/env python3
"""
Colab Runner CLI
Allows running any project script locally or remotely on an active Google Colab GPU session.

Usage:
    python colab_runner.py run <path_to_script> [--session SESSION_NAME]
    python colab_runner.py list-projects
    python colab_runner.py status
"""

import sys
import os
import subprocess
import argparse

DEFAULT_SESSION = "test-gpu"

def run_command(cmd: str, check: bool = True) -> str:
    print(f"\033[1;34m[colab-runner]\033[0m Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, text=True, capture_output=False)
    if check and res.returncode != 0:
        print(f"\033[1;31m[colab-runner] Error (exit code {res.returncode})\033[0m", file=sys.stderr)
        sys.exit(res.returncode)
    return ""

def execute_on_colab(script_path: str, session: str, timeout: int = 300):
    if not os.path.exists(script_path):
        print(f"\033[1;31mFile not found: {script_path}\033[0m", file=sys.stderr)
        sys.exit(1)
        
    print(f"\033[1;32m[colab-runner] Dispatching '{script_path}' to Colab session '{session}'...\033[0m")
    cmd = f"colab exec -s {session} -f {script_path} --timeout {timeout}"
    subprocess.run(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser(description="AI Infra Colab Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a project script on Colab")
    run_parser.add_argument("script", help="Path to Python script")
    run_parser.add_argument("-s", "--session", default=DEFAULT_SESSION, help="Colab session name")
    run_parser.add_argument("--timeout", type=int, default=300, help="Execution timeout in seconds")

    # Status command
    subparsers.add_parser("status", help="Check Colab session status and GPU stats")

    # List projects
    subparsers.add_parser("list-projects", help="List all 15 AI Infra projects")

    args = parser.parse_args()

    if args.command == "run":
        execute_on_colab(args.script, args.session, args.timeout)
    elif args.command == "status":
        run_command("colab status && colab usage")
    elif args.command == "list-projects":
        projects_dir = os.path.join(os.path.dirname(__file__), "projects")
        if os.path.exists(projects_dir):
            for p in sorted(os.listdir(projects_dir)):
                p_path = os.path.join(projects_dir, p)
                if os.path.isdir(p_path):
                    print(f"  * {p}")
        else:
            print("No projects directory found.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
