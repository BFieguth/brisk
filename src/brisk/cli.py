import os
import argparse

def create_project(project_name):
    project_dir = os.path.join(os.getcwd(), project_name)
    os.makedirs(project_dir, exist_ok=True)

    with open(os.path.join(project_dir, 'settings.py'), 'w') as f:
        f.write("# settings.py\n")

    with open(os.path.join(project_dir, 'algorithms.py'), 'w') as f:
        f.write("# algorithms.py\n")

    with open(os.path.join(project_dir, 'metrics.py'), 'w') as f:
        f.write("# metrics.py\n")

    workflows_dir = os.path.join(project_dir, 'workflows')
    os.makedirs(workflows_dir, exist_ok=True)

    with open(os.path.join(workflows_dir, 'workflow.py'), 'w') as f:
        f.write("# workflow.py\n")

    print(f"New ML project created in: {project_dir}")


def main():
    parser = argparse.ArgumentParser(description="ML Toolkit CLI")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    create_parser = subparsers.add_parser("create", help="Create a new project")
    create_parser.add_argument("project_name", help="Name of the project to create")

    args = parser.parse_args()

    if args.command == "create":
        create_project(args.project_name)

if __name__ == "__main__":
    main()
