import unittest
import sys
from pathlib import Path

def main():
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(tests_dir), pattern='test_*.py')
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main()