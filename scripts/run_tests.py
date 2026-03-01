import sys
import os
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    print("="*60)
    print("RUNNING POKER ENGINE TESTS")
    print("="*60)
    
    result = subprocess.run(
        ['pytest', 'tests/', '-v', '--tb=short'],
        cwd=os.path.join(os.path.dirname(__file__), '..')
    )
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
