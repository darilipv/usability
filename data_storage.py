#!/usr/bin/env python
"""
Data Storage Module
Handles persistent storage and retrieval of test results.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class DataStorage:
    """
    Abstract base class for data storage operations.
    Defines the interface for storing and retrieving test data.
    """
    
    def save_test_result(self, test_data: Dict[str, Any]) -> None:
        """Save a test result. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement save_test_result")
    
    def load_test_results(self) -> List[Dict[str, Any]]:
        """Load all test results. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_test_results")
    
    def clear_data(self) -> None:
        """Clear all stored data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement clear_data")


class JSONDataStorage(DataStorage):
    """
    Concrete implementation of DataStorage using JSON files.
    Stores test results in a structured JSON format.
    """
    
    def __init__(self, storage_dir: str = "test_data"):
        """
        Initialize JSON data storage.
        
        Args:
            storage_dir: Directory where test data will be stored
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._data_file = self._storage_dir / "test_results.json"
        self._ensure_data_file()
    
    def _ensure_data_file(self) -> None:
        """Ensure the data file exists, create if it doesn't."""
        if not self._data_file.exists():
            with open(self._data_file, 'w') as f:
                json.dump([], f)
    
    def save_test_result(self, test_data: Dict[str, Any]) -> None:
        """
        Save a test result to the JSON file.
        
        Args:
            test_data: Dictionary containing test result data
        """
        # Add timestamp if not present
        if 'timestamp' not in test_data:
            test_data['timestamp'] = datetime.now().isoformat()
        
        # Load existing data
        existing_data = self.load_test_results()
        
        # Append new test result
        existing_data.append(test_data)
        
        # Save back to file
        with open(self._data_file, 'w') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
    
    def load_test_results(self) -> List[Dict[str, Any]]:
        """
        Load all test results from the JSON file.
        
        Returns:
            List of test result dictionaries
        """
        if not self._data_file.exists():
            return []
        
        try:
            with open(self._data_file, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            return []
    
    def clear_data(self) -> None:
        """Clear all stored test data."""
        with open(self._data_file, 'w') as f:
            json.dump([], f)
    
    def get_storage_path(self) -> str:
        """Get the path to the storage directory."""
        return str(self._storage_dir)
    
    def get_data_file_path(self) -> str:
        """Get the path to the data file."""
        return str(self._data_file)

