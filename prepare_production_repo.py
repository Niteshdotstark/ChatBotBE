#!/usr/bin/env python3
"""
Prepare repository for production push - ensure only production files are included
"""

import os
import glob
import shutil

def create_production_file_list():
    """Create a list of production-ready files that should be in the repo"""
    
    production_files = {
        # Core application files
        "core": [
            "main.py",
            "schemas.py", 
            "models.py",
            "database.py",
            "crud.py",
            "requirements.txt",
            ".gitignore",
            "README.md",
            "PRODUCTION_SYSTEM_S