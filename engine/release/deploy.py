"""
VØX Release - Deploy
--------------------

Deployment helpers for various platforms.

AXIØM Phase 12: Release/Reflect - "How do we ship this and learn from it?"
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from .models import (
    DeploymentTarget,
    DeploymentConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class DeploymentResult:
    """Result of deployment operation."""
    target: DeploymentTarget
    success: bool = True
    message: str = ""
    url: str = ""
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    deployed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target": self.target.value,
            "success": self.success,
            "message": self.message,
            "url": self.url,
            "logs": self.logs,
            "metadata": self.metadata,
            "deployed_at": self.deployed_at.isoformat(),
        }


class DeploymentHelper:
    """
    Helper for deploying VØX to various platforms.

    Features:
        - Docker image building
        - Cloud deployment configs
        - Serverless deployment
        - Kubernetes manifests
    """

    def __init__(
        self,
        config: Optional[DeploymentConfig] = None,
    ):
        """
        Initialize deployment helper.

        Args:
            config: Deployment configuration
        """
        self.config = config or DeploymentConfig()

    def generate_dockerfile(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate Dockerfile.

        Args:
            output_path: Path to write Dockerfile

        Returns:
            Dockerfile content
        """
        content = self.config.generate_dockerfile()

        if output_path:
            Path(output_path).write_text(content)
            logger.info(f"Generated Dockerfile at {output_path}")

        return content

    def generate_docker_compose(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate docker-compose.yml.

        Args:
            output_path: Path to write file

        Returns:
            docker-compose.yml content
        """
        content = self.config.generate_docker_compose()

        if output_path:
            Path(output_path).write_text(content)
            logger.info(f"Generated docker-compose.yml at {output_path}")

        return content

    def generate_kubernetes(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate Kubernetes manifest.

        Args:
            output_path: Path to write file

        Returns:
            Kubernetes manifest content
        """
        content = self.config.generate_kubernetes_manifest()

        if output_path:
            Path(output_path).write_text(content)
            logger.info(f"Generated Kubernetes manifest at {output_path}")

        return content

    def generate_lambda_config(self) -> str:
        """
        Generate AWS Lambda configuration.

        Returns:
            Lambda SAM template content
        """
        template = f'''AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: VØX Lambda Deployment

Globals:
  Function:
    Timeout: 30
    MemorySize: 512
    Runtime: python3.11

Resources:
  VoxFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: {self.config.name}
      CodeUri: .
      Handler: axiom_vox.lambda_handler.handler
      Description: AXIØM VØX Text-to-Speech API
      Environment:
        Variables:
          VOX_ENV: production
      Events:
        Api:
          Type: Api
          Properties:
            Path: /{{proxy+}}
            Method: ANY

  VoxFunctionLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub "/aws/lambda/${{VoxFunction}}"
      RetentionInDays: 14

Outputs:
  VoxApi:
    Description: API Gateway endpoint URL
    Value: !Sub "https://${{ServerlessRestApi}}.execute-api.${{AWS::Region}}.amazonaws.com/Prod/"
'''
        return template

    def generate_cloud_run_config(self) -> str:
        """
        Generate Google Cloud Run configuration.

        Returns:
            Cloud Run service.yaml content
        """
        env_lines = "\n".join(
            f"        - name: {k}\n          value: \"{v}\""
            for k, v in self.config.environment.items()
        )

        default_env = "        - name: VOX_ENV\n          value: production"
        return f'''apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: {self.config.name}
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
      - image: gcr.io/PROJECT_ID/{self.config.image_name}:{self.config.image_tag}
        ports:
        - containerPort: {self.config.port}
        resources:
          limits:
            memory: {self.config.memory_limit}
            cpu: {self.config.cpu_limit}
        env:
{env_lines or default_env}
        livenessProbe:
          httpGet:
            path: {self.config.health_check_path}
        readinessProbe:
          httpGet:
            path: {self.config.readiness_path}
'''

    def build_docker_image(
        self,
        dockerfile_path: str = "Dockerfile",
        context: str = ".",
        push: bool = False,
    ) -> DeploymentResult:
        """
        Build Docker image.

        Args:
            dockerfile_path: Path to Dockerfile
            context: Build context directory
            push: Push image after build

        Returns:
            Deployment result
        """
        result = DeploymentResult(target=DeploymentTarget.DOCKER)
        image_tag = f"{self.config.image_name}:{self.config.image_tag}"

        try:
            # Build image
            build_cmd = [
                "docker", "build",
                "-t", image_tag,
                "-f", dockerfile_path,
                context,
            ]

            build_result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
            )

            result.logs.append(build_result.stdout)

            if build_result.returncode != 0:
                result.success = False
                result.message = f"Build failed: {build_result.stderr}"
                result.logs.append(build_result.stderr)
                return result

            result.message = f"Built image: {image_tag}"
            result.metadata["image"] = image_tag

            # Push if requested
            if push:
                push_result = subprocess.run(
                    ["docker", "push", image_tag],
                    capture_output=True,
                    text=True,
                )

                result.logs.append(push_result.stdout)

                if push_result.returncode != 0:
                    result.success = False
                    result.message = f"Push failed: {push_result.stderr}"
                else:
                    result.message = f"Pushed image: {image_tag}"

        except Exception as e:
            result.success = False
            result.message = f"Docker build failed: {e}"

        return result

    def deploy_to_pypi(
        self,
        dist_dir: str = "dist",
        test_pypi: bool = True,
    ) -> DeploymentResult:
        """
        Deploy package to PyPI.

        Args:
            dist_dir: Directory containing distributions
            test_pypi: Deploy to TestPyPI instead

        Returns:
            Deployment result
        """
        result = DeploymentResult(target=DeploymentTarget.PYPI)

        try:
            cmd = ["python", "-m", "twine", "upload"]

            if test_pypi:
                cmd.extend(["--repository", "testpypi"])

            cmd.append(f"{dist_dir}/*")

            upload_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            result.logs.append(upload_result.stdout)

            if upload_result.returncode != 0:
                result.success = False
                result.message = f"Upload failed: {upload_result.stderr}"
                result.logs.append(upload_result.stderr)
            else:
                repo = "test.pypi.org" if test_pypi else "pypi.org"
                result.message = f"Uploaded to {repo}"
                result.url = f"https://{repo}/project/axiom-vox/"

        except Exception as e:
            result.success = False
            result.message = f"PyPI deploy failed: {e}"

        return result

    def generate_all_configs(
        self,
        output_dir: str = "deploy",
    ) -> Dict[str, str]:
        """
        Generate all deployment configurations.

        Args:
            output_dir: Directory to write configs

        Returns:
            Dict mapping filename to content
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        configs = {}

        # Dockerfile
        dockerfile = self.generate_dockerfile()
        (output_path / "Dockerfile").write_text(dockerfile)
        configs["Dockerfile"] = dockerfile

        # docker-compose.yml
        compose = self.generate_docker_compose()
        (output_path / "docker-compose.yml").write_text(compose)
        configs["docker-compose.yml"] = compose

        # Kubernetes
        k8s = self.generate_kubernetes()
        (output_path / "kubernetes.yaml").write_text(k8s)
        configs["kubernetes.yaml"] = k8s

        # Lambda
        lambda_config = self.generate_lambda_config()
        (output_path / "template.yaml").write_text(lambda_config)
        configs["template.yaml"] = lambda_config

        # Cloud Run
        cloud_run = self.generate_cloud_run_config()
        (output_path / "service.yaml").write_text(cloud_run)
        configs["service.yaml"] = cloud_run

        logger.info(f"Generated {len(configs)} deployment configs in {output_dir}")
        return configs

    def check_prerequisites(
        self,
        target: DeploymentTarget,
    ) -> Dict[str, bool]:
        """
        Check prerequisites for deployment target.

        Args:
            target: Deployment target

        Returns:
            Dict mapping tool to availability
        """
        checks = {}

        if target == DeploymentTarget.DOCKER:
            checks["docker"] = self._check_command("docker")

        elif target == DeploymentTarget.PYPI:
            checks["twine"] = self._check_command("twine")

        elif target == DeploymentTarget.LAMBDA:
            checks["aws"] = self._check_command("aws")
            checks["sam"] = self._check_command("sam")

        elif target == DeploymentTarget.CLOUD_RUN:
            checks["gcloud"] = self._check_command("gcloud")

        elif target == DeploymentTarget.KUBERNETES:
            checks["kubectl"] = self._check_command("kubectl")

        return checks

    def _check_command(self, cmd: str) -> bool:
        """Check if command is available."""
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
            )
            return result.returncode == 0
        except Exception:
            return False


def generate_deployment_configs(
    output_dir: str = "deploy",
    version: str = "",
) -> Dict[str, str]:
    """
    Generate all VØX deployment configurations.

    Args:
        output_dir: Output directory
        version: Version tag

    Returns:
        Generated configs
    """
    config = DeploymentConfig(
        image_tag=version or "latest",
    )
    helper = DeploymentHelper(config)
    return helper.generate_all_configs(output_dir)


def deploy_docker(
    push: bool = False,
) -> DeploymentResult:
    """
    Build and optionally push VØX Docker image.

    Args:
        push: Push image to registry

    Returns:
        Deployment result
    """
    helper = DeploymentHelper()

    # Generate Dockerfile first
    helper.generate_dockerfile("Dockerfile")

    return helper.build_docker_image(push=push)
