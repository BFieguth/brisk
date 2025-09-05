"""Environment management for reproducible ML experiments.

This module provides functionality to capture, compare, and manage Python environments
for reproducible machine learning experiments. It handles package versions, Python
version compatibility, and provides tools for environment validation and export.

Classes
-------
VersionMatch : Enum
    Enumeration of version matching states
PackageInfo : dataclass
    Information about a package and its version
EnvironmentDiff : dataclass
    Represents differences between environments
EnvironmentManager : class
    Main class for environment management operations
"""

import subprocess
import sys
import json
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class VersionMatch(Enum):
    """Enum for version matching status."""
    EXACT = "exact"
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    MISSING = "missing"
    EXTRA = "extra"


@dataclass
class EnvironmentDiff:
    """Represents differences between environments."""
    package: str
    original_version: Optional[str]
    current_version: Optional[str]
    status: VersionMatch
    is_critical: bool = False
    
    def __str__(self) -> str:
        critical_marker = "[CRITICAL]" if self.is_critical else ""
        
        if self.status == VersionMatch.MISSING:
            return f"{self.package}=={self.original_version} (not installed) {critical_marker}".strip()
        elif self.status == VersionMatch.EXTRA:
            return f"{self.package}=={self.current_version} (not in original) {critical_marker}".strip()
        elif self.status == VersionMatch.INCOMPATIBLE:
            if self.is_critical:
                return f"{self.package}: {self.original_version} → {self.current_version} (major/minor version change) {critical_marker}"
            else:
                return f"{self.package}: {self.original_version} → {self.current_version} (major version change)"
        elif self.status == VersionMatch.COMPATIBLE:
            return f"{self.package}: {self.original_version} → {self.current_version} (patch version change)"
        else:
            return f"{self.package}=={self.current_version}"


class EnvironmentManager:
    """Manages environment capture, comparison, and export for reproducible runs."""    
    CRITICAL_PACKAGES = {
        'numpy', 'pandas', 'scikit-learn', 'scipy', "joblib"
    }
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
    
    def capture_environment(self) -> Dict:
        """
        Capture current environment as structured data.
        
        Returns
        -------
        Dict
            Environment information in a clean, structured format
        """
        env = {
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "system": {
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
            },
            "packages": self._capture_packages(),
        }        
        return env
    
    def _capture_packages(self) -> Dict[str, Dict]:
        """Capture installed packages in structured format."""
        packages = {}
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            try:
                installed = json.loads(result.stdout)
                
                for pkg in installed:
                    name = pkg['name'].lower()
                    packages[name] = {
                        "version": pkg['version'],
                        "is_critical": name in self.CRITICAL_PACKAGES
                    }
                    
            except json.JSONDecodeError:
                print("Warning: Could not parse pip list output")
            
        return packages

    def compare_environments(self, saved_env: Dict) -> Tuple[List[EnvironmentDiff], bool]:
        """
        Compare current environment with saved environment.
        
        Parameters
        ----------
        saved_env : Dict
            Saved environment configuration
            
        Returns
        -------
        Tuple[List[EnvironmentDiff], bool]
            Tuple of (differences list, whether environments are compatible)
        """
        differences = []
        
        current_packages = self._get_current_packages_dict()
        saved_packages = saved_env.get("packages", {})
        
        saved_python = saved_env.get("python", {}).get("version", "unknown")
        current_python = platform.python_version()
        if saved_python != current_python:
            saved_major_minor = '.'.join(saved_python.split('.')[:2])
            current_major_minor = '.'.join(current_python.split('.')[:2])
            
            if saved_major_minor != current_major_minor:
                differences.append(
                    EnvironmentDiff(
                        package="python",
                        original_version=saved_python,
                        current_version=current_python,
                        status=VersionMatch.INCOMPATIBLE,
                        is_critical=True
                    )
                )
        
        all_packages = set(saved_packages.keys()) | set(current_packages.keys())
        
        for package in all_packages:
            saved_info = saved_packages.get(package)
            current_version = current_packages.get(package)
            
            if saved_info is None:
                differences.append(
                    EnvironmentDiff(
                        package=package,
                        original_version=None,
                        current_version=current_version,
                        status=VersionMatch.EXTRA,
                        is_critical=package in self.CRITICAL_PACKAGES
                    )
                )
            elif current_version is None:
                differences.append(
                    EnvironmentDiff(
                        package=package,
                        original_version=saved_info.get("version"),
                        current_version=None,
                        status=VersionMatch.MISSING,
                        is_critical=package in self.CRITICAL_PACKAGES
                    )
                )
            else:
                saved_version = saved_info.get("version")
                is_critical = package in self.CRITICAL_PACKAGES
                status = self._compare_versions(saved_version, current_version, is_critical)
                
                if status != VersionMatch.EXACT:
                    differences.append(
                        EnvironmentDiff(
                            package=package,
                            original_version=saved_version,
                            current_version=current_version,
                            status=status,
                            is_critical=package in self.CRITICAL_PACKAGES
                        )
                    )

        is_compatible = True
        for diff in differences:
            if diff.package in self.CRITICAL_PACKAGES or diff.package == "python":
                # Critical packages and Python: strict compatibility
                if diff.status in [VersionMatch.MISSING, VersionMatch.INCOMPATIBLE]:
                    is_compatible = False
                    break
            else:
                # Non-critical packages: only missing packages break compatibility
                if diff.status == VersionMatch.MISSING:
                    is_compatible = False
                    break
        return differences, is_compatible
    
    def _compare_versions(self, version1: str, version2: str, is_critical: bool = False) -> VersionMatch:
        """Compare two version strings with optional strict mode for critical packages.
        
        Parameters
        ----------
        version1 : str
            Original version
        version2 : str
            Current version
        is_critical : bool, default=False
            If True, apply stricter compatibility rules (major.minor must match)
            
        Returns
        -------
        VersionMatch
            The compatibility status
        """
        if version1 == version2:
            return VersionMatch.EXACT
        
        v1_parts = version1.split('.')
        v2_parts = version2.split('.')
        
        if v1_parts[0] != v2_parts[0]:
            return VersionMatch.INCOMPATIBLE
        
        if is_critical and len(v1_parts) > 1 and len(v2_parts) > 1:
            if v1_parts[1] != v2_parts[1]:
                return VersionMatch.INCOMPATIBLE
        
        return VersionMatch.COMPATIBLE
    
    def _get_current_packages_dict(self) -> Dict[str, str]:
        """Get currently installed packages as a simple dict."""
        packages = {}
        result = subprocess.run(
            ["pip", "list", "--format=json"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            try:
                for pkg in json.loads(result.stdout):
                    packages[pkg['name'].lower()] = pkg['version']
            except json.JSONDecodeError:
                pass
        
        return packages
    
    def export_requirements(self, saved_env: Dict, output_path: Path,
                           include_all: bool = False,
                           include_python: bool = True) -> Path:
        """
        Export environment as requirements.txt file.
        
        Parameters
        ----------
        saved_env : Dict
            Saved environment configuration
        output_path : Path
            Path to save requirements file
        include_all : bool, default=False
            Include all packages, not just critical ones
        include_python : bool, default=True
            Include Python version as comment
            
        Returns
        -------
        Path
            Path to created requirements file
        """
        packages = saved_env.get("packages", {})
        python_version = saved_env.get("python", {}).get("version", "unknown")
        
        lines = []
        
        lines.append("# Auto-generated requirements file from `brisk run`")
        lines.append(f"# Generated at: {saved_env.get('timestamp', 'unknown')}")
        
        if include_python:
            lines.append(f"# Python version: {python_version}")
        
        lines.append("")
        
        critical_packages = []
        other_packages = []
        
        for name, info in sorted(packages.items()):
            if info.get("is_critical") or include_all:
                version = info.get("version", "")
                if version:
                    if info.get("is_critical"):
                        critical_packages.append(f"{name}=={version}")
                    else:
                        other_packages.append(f"{name}=={version}")
        
        if critical_packages:
            lines.append("# Critical packages")
            lines.extend(critical_packages)
            lines.append("")
        
        if other_packages and include_all:
            lines.append("# Other packages")
            lines.extend(other_packages)
            lines.append("")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('\n'.join(lines))
        
        return output_path
    
    def generate_environment_report(self, saved_env: Dict) -> str:
        """
        Generate a human-readable environment report.
        
        Parameters
        ----------
        saved_env : Dict
            Saved environment configuration
            
        Returns
        -------
        str
            Formatted report string
        """
        differences, is_compatible = self.compare_environments(saved_env)
        
        report = []
        report.append("=" * 60)
        report.append("ENVIRONMENT COMPATIBILITY REPORT")
        report.append("=" * 60)
        report.append("")
        
        saved_python = saved_env.get("python", {}).get("version", "unknown")
        current_python = platform.python_version()
        
        report.append(f"Python Version:")
        report.append(f"  Original: {saved_python}")
        report.append(f"  Current:  {current_python}")
        
        if saved_python != current_python:
            saved_major_minor = '.'.join(saved_python.split('.')[:2])
            current_major_minor = '.'.join(current_python.split('.')[:2])
            
            if saved_major_minor != current_major_minor:
                report.append(f"Major/minor version mismatch!")
        else:
            report.append(f"Versions match")
        
        report.append("")
        
        saved_system = saved_env.get("system", {})
        if saved_system:
            report.append("System Information:")
            report.append(f"  Original platform: {saved_system.get('platform', 'unknown')}")
            report.append(f"  Current platform:  {platform.platform()}")
            report.append("")
        
        if differences:
            # Separate critical from non-critical differences
            critical_missing = [d for d in differences if d.status == VersionMatch.MISSING and d.is_critical]
            critical_incompatible = [d for d in differences if d.status == VersionMatch.INCOMPATIBLE and d.is_critical]
            critical_compatible = [d for d in differences if d.status == VersionMatch.COMPATIBLE and d.is_critical]
            
            non_critical_missing = [d for d in differences if d.status == VersionMatch.MISSING and not d.is_critical]
            non_critical_incompatible = [d for d in differences if d.status == VersionMatch.INCOMPATIBLE and not d.is_critical]
            non_critical_compatible = [d for d in differences if d.status == VersionMatch.COMPATIBLE and not d.is_critical]
            
            extra = [d for d in differences if d.status == VersionMatch.EXTRA]
            
            # Show critical differences first
            if critical_missing or critical_incompatible:
                report.append("CRITICAL PACKAGE DIFFERENCES:")
                report.append("   (These differences may significantly affect results)")
                
                if critical_missing:
                    report.append("\n   Missing Critical Packages:")
                    for diff in critical_missing:
                        report.append(f"     {str(diff)}")
                
                if critical_incompatible:
                    report.append("\n   Incompatible Critical Package Versions:")
                    for diff in critical_incompatible:
                        report.append(f"     {str(diff)}")
                
                report.append("")
            
            if critical_compatible:
                report.append("   Critical Package Patch Differences:")
                report.append("   (Minor differences in critical packages)")
                for diff in critical_compatible:
                    report.append(f"     {str(diff)}")
                report.append("")
            
            # Show non-critical differences
            if non_critical_missing or non_critical_incompatible or non_critical_compatible:
                report.append("  Non-Critical Package Differences:")
                
                if non_critical_missing:
                    report.append("\n   Missing Packages:")
                    for diff in non_critical_missing:
                        report.append(f"     {str(diff)}")
                
                if non_critical_incompatible:
                    report.append("\n   Version Differences:")
                    for diff in non_critical_incompatible:
                        report.append(f"     {str(diff)}")
                
                if non_critical_compatible:
                    report.append("\n   Minor Version Differences:")
                    for diff in non_critical_compatible:
                        report.append(f"     {str(diff)}")
                
                report.append("")
            
            if extra:
                report.append("Additional Packages (not in original):")
                for diff in extra:
                    report.append(f"     {str(diff)}")
                report.append("")
        else:
            report.append("All packages match exactly!")
        
        report.append("")
        report.append("=" * 60)
        
        # Recommendations
        if not is_compatible:
            critical_issues = [d for d in differences 
                             if (d.is_critical or d.package == "python") 
                             and d.status in [VersionMatch.MISSING, VersionMatch.INCOMPATIBLE]]
            
            report.append("\n  RECOMMENDATION:")
            if critical_issues:
                report.append("   Critical package differences detected. Results may vary significantly.")
            report.append("   To recreate the original environment:")
            report.append("")
            report.append("   1. Export requirements:")
            report.append("      brisk export-env <run_id> --output requirements.txt")
            report.append("")
            report.append("   2. Create a new virtual environment:")
            report.append("      python -m venv brisk_env")
            report.append("      source brisk_env/bin/activate  # On Windows: brisk_env\\Scripts\\activate")
            report.append("")
            report.append("   3. Install requirements:")
            report.append("      pip install -r requirements.txt")
            report.append("")
            report.append("   4. Rerun the experiment:")
            report.append("      brisk rerun <run_id>")
        else:
            report.append("\n  Environment is compatible. Results should be reproducible.")
        
        report.append("")
        
        return '\n'.join(report)
