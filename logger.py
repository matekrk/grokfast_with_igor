from collections import defaultdict

class DataLogger:
    def __init__(self, id=None, **kwargs):
        self.logs = defaultdict(lambda: defaultdict(list))
        self.id = id


    def log_data(self, category, key, value):
        """
        Log data under a specific category and key.

        :param category: The category or subset of data logs.
        :param key: The key within the category to store the value.
        :param value: The value to log.
        """
        self.logs[category][key].append(value)

    def update_category(self, category, data_dict):
        """
        Add or update a whole sub-dictionary for a specific category.

        :param category: The category to update.
        :param data_dict: The dictionary containing data to add or update.
        """
        for key in data_dict:
            self.log_data(category, key, data_dict[key])

    def update_category_means(self, category, data_dict):
        self.update_category_with_lists(category, data_dict)

    def value_in_category_key(self, category, key, value):
        return True if value in self.logs[category][key] else False

    def value_in_category(self, category, value):
        return True in self.logs[category].values()

    def update_category_with_lists(self, category, data_dict):
        for key in data_dict:
            self.log_data(category, key, data_dict[key])

    def get_length(self, category, key):
        return len(self.logs[category][key])

    def get_all_logs(self):
        return self.logs.copy()

    def get_logs(self, category=None):
        """
        Retrieve logs for a specific category or all logs if no category is specified.

        :param category: The category to retrieve logs for. If None, retrieve all logs.
        :return: The logs for the specified category or all logs.
        """
        if category:
            return self.logs.get(category, {})
        return self.logs

    def get_last_value(self, category, key):
        category_dict = self.logs.get(category, {})
        if key in category_dict:
            return category_dict[key][-1]
        return None

    def clear_logs(self, category=None):
        """
        Clear logs for a specific category or all logs if no category is specified.

        :param category: The category to clear logs for. If None, clear all logs.
        """
        if category:
            if category in self.logs:
                del self.logs[category]
        else:
            self.logs.clear()
