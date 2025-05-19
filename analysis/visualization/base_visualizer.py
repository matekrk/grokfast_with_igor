from abc import ABC, abstractmethod


class BaseVisualizer(ABC):
    """
    Abstract base class for all visualizers.
    All concrete visualizer implementations must inherit from this class.
    """

    @abstractmethod
    def can_visualize(self, data):
        """
        Check if this visualizer can handle the given data.

        Args:
            data: The data to check

        Returns:
            bool: True if this visualizer can handle the data, False otherwise
        """
        pass

    @abstractmethod
    def get_supported_fields(self, logger_data):
        """
        Get list of logger fields that this visualizer supports.

        Args:
            logger_data: The logger data dictionary

        Returns:
            list: List of field paths that this visualizer supports
        """
        pass

    @abstractmethod
    def visualize(self, data, config=None):
        """
        Create and return a visualization from the data.

        Args:
            data: The data to visualize
            config: Optional configuration dictionary

        Returns:
            object: The visualization object (e.g., matplotlib figure)
        """
        pass

    @abstractmethod
    def save(self, visualization, path):
        """
        Save the visualization to the specified path.

        Args:
            visualization: The visualization object to save
            path: Path where to save the visualization

        Returns:
            str: The path where the visualization was saved
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """
        Get the name of this visualizer.

        Returns:
            str: The visualizer name
        """
        pass

    @property
    @abstractmethod
    def description(self):
        """
        Get a description of this visualizer.

        Returns:
            str: The visualizer description
        """
        pass