#!/usr/bin/env python3
"""
Script to fix all alignment issues in the codebase
"""

import os
import re
from pathlib import Path

def fix_print_statements(file_path):
    """Replace print() with logger.info()"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if logging is imported
    if 'import logging' not in content and 'from logging import' not in content:
        # Add import at the beginning after other imports
        lines = content.split('\n')
        import_index = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_index = i + 1
            elif import_index > 0 and not line.startswith(('import ', 'from ')):
                break

        # Add logging import
        if import_index > 0:
            lines.insert(import_index, 'import logging')
            lines.insert(import_index + 1, '')
            content = '\n'.join(lines)

    # Check if logger is defined
    if 'logger = logging.getLogger' not in content:
        # Add logger after imports
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import logging'):
                # Find next non-import line
                for j in range(i+1, len(lines)):
                    if not lines[j].startswith(('import ', 'from ', '#')) and lines[j].strip():
                        lines.insert(j, 'logger = logging.getLogger(__name__)\n')
                        break
                break
        content = '\n'.join(lines)

    # Replace print statements
    # Handle print(f"...") format
    content = re.sub(r'print\(f"([^"]+)"\)', r'logger.info(f"\1")', content)
    content = re.sub(r"print\(f'([^']+)'\)", r"logger.info(f'\1')", content)

    # Handle print("...") format
    content = re.sub(r'print\("([^"]+)"\)', r'logger.info("\1")', content)
    content = re.sub(r"print\('([^']+)'\)", r"logger.info('\1')", content)

    # Handle print(...) with variables
    content = re.sub(r'print\(([^)]+)\)', r'logger.info(\1)', content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return True

def fix_hardcoded_paths():
    """Fix hardcoded model paths to use environment variables"""

    # Fix src/config.py
    config_file = Path("src/config.py")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace hardcoded model path
        content = content.replace(
            'model: str = "gpt-oss-20b"  # Default to GPT-OSS',
            'model: str = os.getenv("MODEL_NAME", "gpt-oss-20b")  # From environment'
        )

        # Add os import if needed
        if 'import os' not in content:
            content = 'import os\n' + content

        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed hardcoded paths in {config_file}")

def add_logging_config():
    """Add centralized logging configuration"""

    logging_config = '''#!/usr/bin/env python3
"""
Centralized logging configuration for the entire project
"""

import logging
import os
from pathlib import Path

def setup_logging():
    """Configure logging for the entire application"""

    # Get log level from environment
    log_level = os.getenv('LOG_LEVEL', 'INFO')

    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_dir / 'app.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers
    )

    # Disable noisy loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    return logging.getLogger(__name__)

# Initialize logging when module is imported
logger = setup_logging()
'''

    # Write logging configuration
    logging_file = Path("src/utils/logging_config.py")
    logging_file.parent.mkdir(exist_ok=True)
    with open(logging_file, 'w') as f:
        f.write(logging_config)
    print(f"Created logging configuration at {logging_file}")

def main():
    """Main function to fix all alignment issues"""

    print("Starting alignment fixes...")

    # List of files with print statements
    files_with_prints = [
        "src/moe/extensions/flash_attention.py",
        "src/moe/optimization_safety/optimization_control_center.py",
        "src/moe/extensions/quantization_manager.py",
        "src/moe/extensions/torch_compile_wrapper.py",
        "src/moe/cuda_kernels.py",
        "src/moe/moe_config.py",
        "src/moe/tiered_cache.py",
        "src/moe/optimization_safety/optimization_monitor.py",
        "src/moe/dynamic_batch_manager.py",
        "src/moe/multi_gpu_moe.py",
        "src/moe/async_expert_loader.py",
        "src/moe/native_moe_complete.py",
        "src/moe/expert_cache.py"
    ]

    # Fix print statements
    fixed_count = 0
    for file_path in files_with_prints:
        if Path(file_path).exists():
            if fix_print_statements(file_path):
                print(f"Fixed print statements in {file_path}")
                fixed_count += 1

    print(f"Fixed print statements in {fixed_count} files")

    # Fix hardcoded paths
    fix_hardcoded_paths()

    # Add logging configuration
    add_logging_config()

    print("\nAlignment fixes completed!")
    print("Next steps:")
    print("1. Review the changes")
    print("2. Run tests to ensure nothing broke")
    print("3. Commit the fixes")

if __name__ == "__main__":
    main()