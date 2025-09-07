"""Environment management for reproducible ML experiments.

This module provides functionality to capture, compare, and manage Python
environments for reproducible machine learning experiments. It handles package
versions, Python version compatibility, and provides tools for environment
validation and export.

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
import pathlib
import dataclasses
import enum
from typing import Dict, List, Tuple, Optional

class VersionMatch(enum.Enum):
    """Enumeration of version matching states.
    
    Defines the possible states when comparing package versions between
    environments. Used to categorize compatibility levels.
    
    Attributes
    ----------
    EXACT : str
        Versions match exactly
    COMPATIBLE : str
        Versions are compatible (patch-level differences for critical packages,
        or major version differences for non-critical packages)
    INCOMPATIBLE : str
        Versions are incompatible (major version differences for critical
        packages)
    MISSING : str
        Package was present in original environment but missing in current
    EXTRA : str
        Package is present in current environment but was not in original
    """
    EXACT = "exact"
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    MISSING = "missing"
    EXTRA = "extra"


@dataclasses.dataclass
class EnvironmentDiff:
    """Represents differences between environments.
    
    Contains information about package version differences between a saved
    environment and the current environment.
    
    Attributes
    ----------
    package : str
        Name of the package
    original_version : Optional[str]
        Version in the original/saved environment
    current_version : Optional[str]
        Version in the current environment
    status : VersionMatch
        Compatibility status of the version difference
    is_critical : bool, default=False
        Whether this package is considered critical for reproducibility
    """
    package: str
    original_version: Optional[str]
    current_version: Optional[str]
    status: VersionMatch
    is_critical: bool = False

    def __str__(self) -> str:
        """Return string representation of the environment difference.
        
        Returns
        -------
        str
            Human-readable description of the package difference
        """
        critical_marker = "[CRITICAL]" if self.is_critical else ""

        if self.status == VersionMatch.MISSING:
            return (
                f"{self.package}=={self.original_version} "
                f"(not installed) {critical_marker}".strip()
            )
        if self.status == VersionMatch.EXTRA:
            return (
                f"{self.package}=={self.current_version} "
                f"(not in original) {critical_marker}".strip()
            )
        if self.status == VersionMatch.INCOMPATIBLE:
            if self.is_critical:
                return (
                    f"{self.package}: {self.original_version} → "
                    f"{self.current_version} (major/minor version change) "
                    f"{critical_marker}"
                )
            return (
                f"{self.package}: {self.original_version} → "
                f"{self.current_version} (major version change)"
            )
        if self.status == VersionMatch.COMPATIBLE:
            return (
                f"{self.package}: {self.original_version} → "
                f"{self.current_version} (patch version change)"
            )
        return f"{self.package}=={self.current_version}"


class EnvironmentManager:
    """
    Manages environment capture, comparison, and export for reproducible runs.
    
    Provides comprehensive environment management capabilities including
    capturing current environment state, comparing with saved environments,
    and exporting requirements files for reproducibility.
    
    Attributes
    ----------
    CRITICAL_PACKAGES : set
        Set of package names considered critical for ML reproducibility
    project_root : pathlib.Path
        Path to the project root directory
    """
    CRITICAL_PACKAGES = {
        "numpy", "pandas", "scikit-learn", "scipy", "joblib"
    }

    def __init__(self, project_root: pathlib.Path):
        """Initialize the EnvironmentManager.
        
        Parameters
        ----------
        project_root : pathlib.Path
            Path to the project root directory
        """
        self.project_root = pathlib.Path(project_root)

    def capture_environment(self) -> Dict:
        """Capture current environment as structured data.
        
        Collects comprehensive information about the current Python environment
        including Python version, system information, and installed packages.
        
        Returns
        -------
        Dict
            Environment information in a clean, structured format with keys:
            - python: Python version and implementation details
            - system: Platform, architecture, and processor information
            - packages: Dictionary of installed packages with versions
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
        """Capture installed packages in structured format.
        
        Uses pip to get a list of installed packages and categorizes them
        as critical or non-critical based on the CRITICAL_PACKAGES set.
        
        Returns
        -------
        Dict[str, Dict]
            Dictionary mapping package names to their information including:
            - version: Package version string
            - is_critical: Boolean indicating if package is critical
        """
        packages = {}
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )

        if result.returncode == 0:
            try:
                installed = json.loads(result.stdout)

                for pkg in installed:
                    name = pkg["name"].lower()
                    packages[name] = {
                        "version": pkg["version"],
                        "is_critical": name in self.CRITICAL_PACKAGES
                    }

            except json.JSONDecodeError:
                print("Warning: Could not parse pip list output")

        return packages

    def compare_environments(
        self,
        saved_env: Dict
    ) -> Tuple[List[EnvironmentDiff], bool]:
        """Compare current environment with saved environment.
        
        Performs comprehensive comparison between the current environment
        and a previously saved environment configuration. Identifies package
        differences, version changes, and compatibility issues.
        
        Parameters
        ----------
        saved_env : Dict
            Saved environment configuration containing python, system,
            and packages information
            
        Returns
        -------
        Tuple[List[EnvironmentDiff], bool]
            Tuple containing:
            - List of EnvironmentDiff objects describing all differences
            - Boolean indicating whether environments are compatible
        """
        differences = []
        current_packages = self._get_current_packages_dict()
        saved_packages = saved_env.get("packages", {})
        saved_python = saved_env.get("python", {}).get("version", "unknown")
        current_python = platform.python_version()
        if saved_python != current_python:
            saved_major_minor = ".".join(saved_python.split(".")[:2])
            current_major_minor = ".".join(current_python.split(".")[:2])

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
                status = self._compare_versions(
                    saved_version, current_version, is_critical
                )

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
            if (
                diff.package in self.CRITICAL_PACKAGES or
                diff.package == "python"
            ):
                # Critical packages and Python: strict compatibility
                if diff.status in [
                    VersionMatch.MISSING, VersionMatch.INCOMPATIBLE
                ]:
                    is_compatible = False
                    break
            else:
                # Non-critical packages: missing packages break compatibility
                if diff.status == VersionMatch.MISSING:
                    is_compatible = False
                    break
        return differences, is_compatible

    def _compare_versions(
        self,
        version_original: str,
        version_current: str,
        is_critical: bool = False
    ) -> VersionMatch:
        """Compare two version strings for compatibility.
        
        Implements version comparison logic with different rules for critical
        and non-critical packages. Critical packages require major.minor version
        compatibility, while non-critical packages only require major version
        compatibility.
        
        Parameters
        ----------
        version_original : str
            Original version string from saved environment
        version_current : str
            Current version string from current environment
        is_critical : bool, default=False
            Whether this is a critical package requiring stricter compatibility
            
        Returns
        -------
        VersionMatch
            Compatibility status indicating exact match, compatible, or
            incompatible
        """
        if version_original == version_current:
            return VersionMatch.EXACT

        v1_parts = version_original.split(".")
        v2_parts = version_current.split(".")

        if v1_parts[0] != v2_parts[0]:
            return VersionMatch.INCOMPATIBLE

        if is_critical and len(v1_parts) > 1 and len(v2_parts) > 1:
            if v1_parts[1] != v2_parts[1]:
                return VersionMatch.INCOMPATIBLE

        return VersionMatch.COMPATIBLE

    def _get_current_packages_dict(self) -> Dict[str, str]:
        """Get currently installed packages as a simple dict.
        
        Retrieves a simple mapping of package names to versions for
        the current environment using pip list.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping package names (lowercase) to version strings
        """
        packages = {}
        result = subprocess.run(
            ["pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )

        if result.returncode == 0:
            try:
                for pkg in json.loads(result.stdout):
                    packages[pkg["name"].lower()] = pkg["version"]
            except json.JSONDecodeError:
                pass

        return packages

    def export_requirements(
        self,
        saved_env: Dict,
        output_path: pathlib.Path,
        include_all: bool = False,
        include_python: bool = True
    ) -> pathlib.Path:
        """Export environment as requirements.txt file.
        
        Creates a requirements.txt file from a saved environment configuration
        with options to include all packages or just critical ones.
        
        Parameters
        ----------
        saved_env : Dict
            Saved environment configuration containing packages information
        output_path : pathlib.Path
            Path where the requirements file should be saved
        include_all : bool, default=False
            Whether to include all packages or just critical ones
        include_python : bool, default=True
            Whether to include Python version as a comment in the file
            
        Returns
        -------
        pathlib.Path
            Path to the created requirements file
            
        Notes
        -----
        The generated file includes:
        - Header comments with generation timestamp
        - Python version comment (if include_python=True)
        - Critical packages section (always included)
        - Other packages section (if include_all=True)
        """
        packages = saved_env.get("packages", {})
        python_version = saved_env.get("python", {}).get("version", "unknown")

        lines = []

        lines.append("# Auto-generated requirements file from `brisk run`")
        lines.append(f"# Generated at: {saved_env.get("timestamp", "unknown")}")

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

        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

        return output_path

    def _process_python_version_info(self, saved_env: Dict) -> Dict[str, str]:
        """Process Python version information for the report.
        
        Extracts and compares Python version information between saved
        and current environments for report generation.
        
        Parameters
        ----------
        saved_env : Dict
            Saved environment configuration
            
        Returns
        -------
        Dict[str, str]
            Dictionary containing version comparison information with keys:
            - saved: Original Python version
            - current: Current Python version
            - match: Boolean indicating exact version match
            - major_minor_match: Boolean indicating major.minor version match
        """
        saved_python = saved_env.get("python", {}).get("version", "unknown")
        current_python = platform.python_version()

        version_info = {
            "saved": saved_python,
            "current": current_python,
            "match": saved_python == current_python,
            "major_minor_match": True
        }

        if saved_python != current_python:
            saved_major_minor = ".".join(saved_python.split(".")[:2])
            current_major_minor = ".".join(current_python.split(".")[:2])
            version_info["major_minor_match"] = (
                saved_major_minor == current_major_minor
            )

        return version_info

    def _process_system_info(self, saved_env: Dict) -> Dict[str, str]:
        """Process system information for the report.
        
        Extracts and compares system platform information between saved
        and current environments for report generation.
        
        Parameters
        ----------
        saved_env : Dict
            Saved environment configuration
            
        Returns
        -------
        Dict[str, str]
            Dictionary containing system comparison information with keys:
            - saved_platform: Original platform information
            - current_platform: Current platform information
            - has_system_info: Boolean indicating if system info was saved
        """
        saved_system = saved_env.get("system", {})
        return {
            "saved_platform": saved_system.get("platform", "unknown"),
            "current_platform": platform.platform(),
            "has_system_info": bool(saved_system)
        }

    def _categorize_differences(self, differences: List) -> Dict[str, List]:
        """Categorize package differences by type and criticality.
        
        Groups environment differences into categories based on package
        criticality and difference type for organized report generation.
        
        Parameters
        ----------
        differences : List
            List of EnvironmentDiff objects to categorize
            
        Returns
        -------
        Dict[str, List]
            Dictionary with categorized differences containing keys:
            - critical_missing: Critical packages that are missing
            - critical_incompatible: Critical packages with incompatible
            versions
            - critical_compatible: Critical packages with compatible version
            changes
            - non_critical_missing: Non-critical packages that are missing
            - non_critical_incompatible: Non-critical packages with incompatible
            versions
            - non_critical_compatible: Non-critical packages with compatible
            version changes
            - extra: Packages present in current but not in original environment
        """
        categories = {
            "critical_missing": [],
            "critical_incompatible": [],
            "critical_compatible": [],
            "non_critical_missing": [],
            "non_critical_incompatible": [],
            "non_critical_compatible": [],
            "extra": []
        }

        for diff in differences:
            if diff.status == VersionMatch.EXTRA:
                categories["extra"].append(diff)
            elif diff.is_critical:
                if diff.status == VersionMatch.MISSING:
                    categories["critical_missing"].append(diff)
                elif diff.status == VersionMatch.INCOMPATIBLE:
                    categories["critical_incompatible"].append(diff)
                elif diff.status == VersionMatch.COMPATIBLE:
                    categories["critical_compatible"].append(diff)
            else:
                if diff.status == VersionMatch.MISSING:
                    categories["non_critical_missing"].append(diff)
                elif diff.status == VersionMatch.INCOMPATIBLE:
                    categories["non_critical_incompatible"].append(diff)
                elif diff.status == VersionMatch.COMPATIBLE:
                    categories["non_critical_compatible"].append(diff)

        return categories

    def _format_python_version_section(
        self,
        version_info: Dict[str, str]
    ) -> List[str]:
        """Format the Python version section of the report.
        
        Creates formatted text lines for the Python version comparison
        section of the environment report.
        
        Parameters
        ----------
        version_info : Dict[str, str]
            Python version information from _process_python_version_info
            
        Returns
        -------
        List[str]
            List of formatted text lines for the Python version section
        """
        lines = []
        lines.append("Python Version:")
        lines.append(f"  Original: {version_info["saved"]}")
        lines.append(f"  Current:  {version_info["current"]}")

        if version_info["match"]:
            lines.append("Versions match")
        elif not version_info["major_minor_match"]:
            lines.append("Major/minor version mismatch!")

        lines.append("")
        return lines

    def _format_system_info_section(
        self,
        system_info: Dict[str, str]
    ) -> List[str]:
        """Format the system information section of the report.
        
        Creates formatted text lines for the system platform comparison
        section of the environment report.
        
        Parameters
        ----------
        system_info : Dict[str, str]
            System information from _process_system_info
            
        Returns
        -------
        List[str]
            List of formatted text lines for the system information section
        """
        lines = []
        if system_info["has_system_info"]:
            lines.append("System Information:")
            lines.append(
                f" Original platform: {system_info["saved_platform"]}"
            )
            lines.append(
                f" Current platform:  {system_info["current_platform"]}"
            )
            lines.append("")
        return lines

    def _format_critical_differences_section(
        self,
        categories: Dict[str, List]
    ) -> List[str]:
        """Format the critical differences section of the report.
        
        Creates formatted text lines for critical package differences
        in the environment report.
        
        Parameters
        ----------
        categories : Dict[str, List]
            Categorized differences from _categorize_differences
            
        Returns
        -------
        List[str]
            List of formatted text lines for the critical differences section
        """
        lines = []
        critical_missing = categories["critical_missing"]
        critical_incompatible = categories["critical_incompatible"]
        critical_compatible = categories["critical_compatible"]

        if critical_missing or critical_incompatible:
            lines.append("CRITICAL PACKAGE DIFFERENCES:")
            lines.append(
                "   (These differences may significantly affect results)"
            )

            if critical_missing:
                lines.append("\n   Missing Critical Packages:")
                for diff in critical_missing:
                    lines.append(f"     {str(diff)}")

            if critical_incompatible:
                lines.append("\n   Incompatible Critical Package Versions:")
                for diff in critical_incompatible:
                    lines.append(f"     {str(diff)}")

            lines.append("")

        if critical_compatible:
            lines.append("   Critical Package Patch Differences:")
            lines.append("   (Minor differences in critical packages)")
            for diff in critical_compatible:
                lines.append(f"     {str(diff)}")
            lines.append("")

        return lines

    def _format_non_critical_differences_section(
        self,
        categories: Dict[str, List]
    ) -> List[str]:
        """Format the non-critical differences section of the report.
        
        Creates formatted text lines for non-critical package differences
        in the environment report.
        
        Parameters
        ----------
        categories : Dict[str, List]
            Categorized differences from _categorize_differences
            
        Returns
        -------
        List[str]
            List of formatted text lines for the non-critical differences
            section
        """
        lines = []
        non_critical_missing = categories["non_critical_missing"]
        non_critical_incompatible = categories["non_critical_incompatible"]
        non_critical_compatible = categories["non_critical_compatible"]
        extra = categories["extra"]

        if (
            non_critical_missing or
            non_critical_incompatible or
            non_critical_compatible
        ):
            lines.append("  Non-Critical Package Differences:")

            if non_critical_missing:
                lines.append("\n   Missing Packages:")
                for diff in non_critical_missing:
                    lines.append(f"     {str(diff)}")

            if non_critical_incompatible:
                lines.append("\n   Version Differences:")
                for diff in non_critical_incompatible:
                    lines.append(f"     {str(diff)}")

            if non_critical_compatible:
                lines.append("\n   Minor Version Differences:")
                for diff in non_critical_compatible:
                    lines.append(f"     {str(diff)}")

            lines.append("")

        if extra:
            lines.append("Additional Packages (not in original):")
            for diff in extra:
                lines.append(f"     {str(diff)}")
            lines.append("")

        return lines

    def _format_recommendations_section(
        self,
        is_compatible: bool,
        differences: List
    ) -> List[str]:
        """Format the recommendations section of the report.
        
        Creates formatted text lines with recommendations for handling
        environment compatibility issues.
        
        Parameters
        ----------
        is_compatible : bool
            Whether the environments are compatible
        differences : List
            List of EnvironmentDiff objects
            
        Returns
        -------
        List[str]
            List of formatted text lines for the recommendations section
        """
        lines = []

        if not is_compatible:
            critical_issues = [
                d for d in differences
                if (d.is_critical or d.package == "python")
                and d.status in [
                    VersionMatch.MISSING, VersionMatch.INCOMPATIBLE
                ]
            ]

            lines.append("\n  RECOMMENDATION:")
            if critical_issues:
                lines.append(
                    "   Critical package differences detected. "
                    "Results may vary significantly."
                )
            lines.append("   To recreate the original environment:")
            lines.append("")
            lines.append("   1. Export requirements:")
            lines.append(
                "      brisk export-env <run_id> --output requirements.txt"
            )
            lines.append("")
            lines.append("   2. Create a new virtual environment:")
            lines.append("      python -m venv brisk_env")
            lines.append(
                "      source brisk_env/bin/activate  "
                "# On Windows: brisk_env\\Scripts\\activate"
            )
            lines.append("")
            lines.append("   3. Install requirements:")
            lines.append("      pip install -r requirements.txt")
            lines.append("")
            lines.append("   4. Rerun the experiment:")
            lines.append("      brisk rerun <run_id>")
        else:
            lines.append(
                "\n  Environment is compatible. "
                "Results should be reproducible."
            )

        lines.append("")
        return lines

    def generate_environment_report(self, saved_env: Dict) -> str:
        """Generate a human-readable environment report.
        
        Creates a comprehensive report comparing the current environment
        with a saved environment configuration, including detailed analysis
        of package differences and recommendations.
        
        Parameters
        ----------
        saved_env : Dict
            Saved environment configuration to compare against
            
        Returns
        -------
        str
            Formatted report string containing:
            - Environment compatibility status
            - Python version comparison
            - System information comparison
            - Critical package differences
            - Non-critical package differences
            - Recommendations for environment recreation
        """
        differences, is_compatible = self.compare_environments(saved_env)
        version_info = self._process_python_version_info(saved_env)
        system_info = self._process_system_info(saved_env)
        categories = (
            self._categorize_differences(differences)
            if differences else {}
        )

        report = []
        report.append("=" * 60)
        report.append("ENVIRONMENT COMPATIBILITY REPORT")
        report.append("=" * 60)
        report.append("")

        report.extend(self._format_python_version_section(version_info))
        report.extend(self._format_system_info_section(system_info))

        if differences:
            report.extend(self._format_critical_differences_section(categories))
            report.extend(
                self._format_non_critical_differences_section(categories)
            )
        else:
            report.append("All packages match exactly!")

        report.append("")
        report.append("=" * 60)
        report.extend(
            self._format_recommendations_section(is_compatible, differences)
        )

        return "\n".join(report)
