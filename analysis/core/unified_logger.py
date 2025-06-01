# analysis/core/unified_logger.py
import logging
import sys
from typing import Optional, Dict, Any, Union
from pathlib import Path
import json
from datetime import datetime


class UnifiedLogger:
    """Unified logging infrastructure for screen, file, and network logging"""

    def __init__(self, experiment_name: str, log_dir: Optional[Path] = None,
                 enable_wandb: bool = False, enable_file: bool = True,
                 enable_screen: bool = True, log_level: str = "INFO"):
        """
        Initialize unified logger

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
            enable_wandb: Enable Weights & Biases logging
            enable_file: Enable file logging
            enable_screen: Enable screen logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.enable_wandb = enable_wandb
        self.enable_file = enable_file
        self.enable_screen = enable_screen

        # Initialize loggers
        self._setup_python_logger(log_level)
        self._setup_wandb() if enable_wandb else None
        self._setup_file_logging() if enable_file else None

        # Metrics storage
        self.metrics_history = []

        self.info(f"🚀 Unified logger initialized for experiment: {experiment_name}")

    def _setup_python_logger(self, log_level: str):
        """Setup Python logging"""
        self.logger = logging.getLogger(f"circuit_analysis_{self.experiment_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear any existing handlers
        self.logger.handlers.clear()

        if self.enable_screen:
            # Console handler with colored output
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        try:
            import wandb

            # Initialize wandb
            wandb.init(
                project="transformer-circuit-analysis",
                name=self.experiment_name,
                config={
                    "experiment_name": self.experiment_name,
                    "start_time": datetime.now().isoformat()
                }
            )

            self.wandb = wandb
            self.info("✅ Weights & Biases logging enabled")

        except ImportError:
            self.warning("⚠️  Weights & Biases not available. Install with: pip install wandb")
            self.enable_wandb = False
            self.wandb = None
        except Exception as e:
            self.warning(f"⚠️  Failed to initialize Weights & Biases: {e}")
            self.enable_wandb = False
            self.wandb = None

    def _setup_file_logging(self):
        """Setup file logging"""
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Main log file
            log_file = self.log_dir / f"{self.experiment_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # Metrics file
            self.metrics_file = self.log_dir / f"{self.experiment_name}_metrics.jsonl"

            self.info(f"📁 File logging enabled: {log_file}")

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None,
                    category: str = "training"):
        """
        Log metrics to all enabled destinations

        Args:
            metrics: Dictionary of metric names and values
            step: Step/epoch number
            category: Category of metrics (training, evaluation, circuit_analysis, etc.)
        """
        timestamp = datetime.now().isoformat()

        # Create metrics entry
        metrics_entry = {
            "timestamp": timestamp,
            "step": step,
            "category": category,
            "metrics": metrics
        }

        # Store in history
        self.metrics_history.append(metrics_entry)

        # Log to wandb
        if self.enable_wandb and self.wandb:
            wandb_metrics = {f"{category}/{k}": v for k, v in metrics.items()}
            if step is not None:
                self.wandb.log(wandb_metrics, step=step)
            else:
                self.wandb.log(wandb_metrics)

        # Log to file
        if self.enable_file and hasattr(self, 'metrics_file'):
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics_entry) + '\n')

        # Log to console (summary)
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in metrics.items()])
        step_str = f"Step {step} | " if step is not None else ""
        self.info(f"📊 {category.title()} Metrics | {step_str}{metrics_str}")

    def log_circuit_discovery(self, circuit_info: Dict[str, Any], epoch: int):
        """Log circuit discovery events"""
        self.log_metrics({
            "circuits_discovered": circuit_info.get("total_discovered", 0),
            "copy_circuits": circuit_info.get("copy_circuits", 0),
            "induction_circuits": circuit_info.get("induction_circuits", 0),
            "component_circuits": circuit_info.get("component_circuits", 0),
        }, step=epoch, category="circuit_discovery")

        # Detailed logging
        circuit_types = circuit_info.get("circuit_types", {})
        if circuit_types:
            type_summary = " | ".join([f"{k}: {v}" for k, v in circuit_types.items()])
            self.info(f"🔍 Circuit Discovery @ Epoch {epoch} | {type_summary}")

    def log_circuit_evolution(self, evolution_info: Dict[str, Any], epoch: int):
        """Log circuit evolution events"""
        self.log_metrics({
            "stable_circuits": evolution_info.get("stable_circuits", 0),
            "emerging_circuits": evolution_info.get("emerging_circuits", 0),
            "declining_circuits": evolution_info.get("declining_circuits", 0),
            "circuit_relationships": evolution_info.get("total_relationships", 0),
        }, step=epoch, category="circuit_evolution")

        # Evolution summary
        changes = []
        if evolution_info.get("new_circuits", 0) > 0:
            changes.append(f"+{evolution_info['new_circuits']} new")
        if evolution_info.get("evolved_circuits", 0) > 0:
            changes.append(f"~{evolution_info['evolved_circuits']} evolved")
        if evolution_info.get("defunct_circuits", 0) > 0:
            changes.append(f"-{evolution_info['defunct_circuits']} defunct")

        if changes:
            self.info(f"🔄 Circuit Evolution @ Epoch {epoch} | {' | '.join(changes)}")

    def log_validation_results(self, validation_info: Dict[str, Any], epoch: int):
        """Log circuit validation results"""
        self.log_metrics({
            "validated_circuits": validation_info.get("circuits_validated", 0),
            "validation_accuracy": validation_info.get("average_accuracy", 0.0),
            "false_positives": validation_info.get("false_positives", 0),
        }, step=epoch, category="circuit_validation")

        # Validation summary
        accuracy = validation_info.get("average_accuracy", 0.0)
        validated = validation_info.get("circuits_validated", 0)
        self.info(f"✅ Circuit Validation @ Epoch {epoch} | {validated} circuits | {accuracy:.2%} accuracy")

    def info(self, message: str):
        """Log info message"""
        if hasattr(self, 'logger'):
            self.logger.info(message)
        else:
            print(f"INFO: {message}")

    def warning(self, message: str):
        """Log warning message"""
        if hasattr(self, 'logger'):
            self.logger.warning(message)
        else:
            print(f"WARNING: {message}")

    def error(self, message: str):
        """Log error message"""
        if hasattr(self, 'logger'):
            self.logger.error(message)
        else:
            print(f"ERROR: {message}")

    def debug(self, message: str):
        """Log debug message"""
        if hasattr(self, 'logger'):
            self.logger.debug(message)
        else:
            print(f"DEBUG: {message}")

    def finalize_experiment(self):
        """Finalize experiment logging"""
        if self.enable_wandb and self.wandb:
            self.wandb.finish()

        # Save final metrics summary
        if self.enable_file and hasattr(self, 'log_dir'):
            summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
            summary = {
                "experiment_name": self.experiment_name,
                "total_metrics_logged": len(self.metrics_history),
                "categories": list(set(entry["category"] for entry in self.metrics_history)),
                "duration": "calculated_from_timestamps"  # Could calculate actual duration
            }

            with open(summary_file, 'w') as f:
                json.dump(summary, indent=2, fp=f)

        self.info(f"🏁 Experiment {self.experiment_name} logging finalized")


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)